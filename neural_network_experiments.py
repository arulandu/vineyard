
"""
neural_network_experiments.py
=============================

• Experiment 1  — depth sweep        (width = 9, depth ∈ [1, 5])
• Experiment 2  — learning-rate sweep(lr ∈ {1e-3, 3e-3, 1e-2, 3e-2})
• Experiment 3  — batch-size sweep  (batch ∈ {8,16,32,64,128,200})

For every configuration & trial we record

    ─ Decision-surface snapshots  (compressed .npz per trial)
    ─ Full H₀ / H₁ vineyards       (JSON-serialised)
    ─ Per-epoch train / val metrics

Author : ChatGPT (o3) – 2025-05-28
"""

# ─────────────────────────  imports  ────────────────────────── #
import json, random, warnings, os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import StratifiedShuffleSplit

import gudhi

warnings.filterwarnings("ignore")


# ───────────────────────  helper utilities  ─────────────────────── #
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def onehot(labels: List[int]) -> np.ndarray:
    k = max(labels) + 1
    m = np.zeros((len(labels), k), dtype=np.float32)
    m[np.arange(len(labels)), labels] = 1
    return m


# ────────────────────────────  dataset  ─────────────────────────── #
class BlobAndCirclesDataset(Dataset):
    """100 Gaussian blob points + 100 inner-circle points."""
    def __init__(self, seed: int = 2024):
        np.random.seed(seed)
        X0 = np.random.normal(scale=0.10, size=(100, 2))   # class 0
        y0 = [0] * 100

        X1 = make_circles(
            n_samples=200, noise=0.05, factor=0.5,
            shuffle=False, random_state=seed
        )[0][100:]
        y1 = [1] * 100

        self.X = np.vstack([X0, X1]).astype(np.float32)
        self.y = onehot(y0 + y1)

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):  return self.X[idx], self.y[idx]


# ─────────────────────  persistence helpers  ───────────────────── #
def cubical_intervals(A: np.ndarray, dim: int) -> np.ndarray:
    cc = gudhi.CubicalComplex(top_dimensional_cells=A)
    cc.compute_persistence()
    return cc.persistence_intervals_in_dimension(dim)


def pd_to_json(pd: np.ndarray) -> List[List[float]]:
    if pd.size == 0:
        return []
    return [
        [None if np.isinf(b) else float(b),
         None if np.isinf(d) else float(d)]
        for b, d in pd
    ]


# ───────────────────────────  model  ──────────────────────────── #
class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: List[int], activation: str = "relu"):
        super().__init__()
        sizes = [in_dim] + hidden
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(hidden))]
            + [nn.Linear(sizes[-1], out_dim)]
        )
        self.act = activation.lower()

    def _phi(self, x):  # activation
        return torch.sigmoid(x) if self.act == "sigmoid" else torch.relu(x)

    def forward(self, x):
        for lyr in self.layers[:-1]:
            x = self._phi(lyr(x))
        return torch.softmax(self.layers[-1](x), dim=1)


# ────────────────────────  global params  ─────────────────────── #
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
TRIALS_PER_CONFIG = 50
EPOCHS            = 100
SNAP_INTERVAL     = 1
GRID_RES          = 300
SEED              = 2024

NEURONS_PER_LAYER = 8
DEPTH_GRID        = [1, 3, 5, 10]            # Exp 1
LR_GRID           = []   # Exp 2
BATCH_GRID        = [8, 16, 32, 64, 128]  # Exp 3

ARCH_FIXED        = [NEURONS_PER_LAYER] * 3    # used in Exp 2 & 3
LR_FIXED          = 1e-2                      # used in Exp 1 & 3

RESULTS_DIR = Path("vineyard_results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ─────────────────────  evaluation grid  ─────────────────────── #
def make_test_grid(n: int = GRID_RES, margin: float = 1.5) -> np.ndarray:
    xs = np.linspace(-margin, margin, n)
    ys = np.linspace(-margin, margin, n)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)


GRID_TENSOR = torch.from_numpy(make_test_grid()).to(DEVICE)
T_STEPS     = EPOCHS // SNAP_INTERVAL + 1   # snapshots per trial


# ─────────────────────  core training loop  ───────────────────── #
def run_trial(
    hidden_layers: List[int],
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    grid_save_path: Path,
) -> Dict[str, Any]:
    """
    Train one network.  Returns dict with vineyards & metrics.
    Also saves decision-surface snapshots to `grid_save_path` (.npz).
    """
    model = Network(2, 2, hidden_layers).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    lossF = nn.MSELoss()

    vineyard_H0, vineyard_H1 = [], []
    metrics = []
    grids   = np.empty((T_STEPS, GRID_RES, GRID_RES), dtype=np.float16)  # snapshots

    # ------------- helper: store current decision grid ------------- #
    snap_idx = 0
    def snapshot():
        nonlocal snap_idx
        with torch.no_grad():
            decision = (
                model(GRID_TENSOR)[:, 0]
                .detach().cpu().numpy()
                .reshape(GRID_RES, GRID_RES)
                .astype(np.float16)
            )
        grids[snap_idx] = decision
        pd0 = pd_to_json(cubical_intervals(decision, 0))
        pd1 = pd_to_json(cubical_intervals(decision, 1))
        vineyard_H0.append(pd0)
        vineyard_H1.append(pd1)
        snap_idx += 1

    # ----- epoch 0 snapshot -----
    snapshot()

    # ------------- training loop ------------- #
    for epoch in range(1, EPOCHS + 1):
        # --- training phase ---
        model.train()
        tr_loss, tr_correct, n_tr = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            l = lossF(pred, y)
            l.backward()
            opt.step()

            tr_loss   += l.item() * len(x)
            tr_correct += (pred.argmax(1) == y.argmax(1)).sum().item()
            n_tr += len(x)
        tr_loss /= n_tr
        tr_acc   = tr_correct / n_tr

        # --- validation phase ---
        model.eval()
        val_loss, val_correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                l = lossF(pred, y)
                val_loss   += l.item() * len(x)
                val_correct += (pred.argmax(1) == y.argmax(1)).sum().item()
                n_val += len(x)
        val_loss /= n_val
        val_acc   = val_correct / n_val

        metrics.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc":  tr_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })

        # snapshot every SNAP_INTERVAL
        if epoch % SNAP_INTERVAL == 0:
            snapshot()

    # save decision grids
    np.savez_compressed(grid_save_path, grids=grids)

    return {"H0": vineyard_H0, "H1": vineyard_H1, "metrics": metrics,
            "grid_file": str(grid_save_path.name)}


# ────────────────────  helper: run trials & save  ──────────────────── #
def collect_trials(hidden_layers, lr, batch_size, cfg_tag):
    base_dir = RESULTS_DIR / cfg_tag
    base_dir.mkdir(parents=True, exist_ok=True)

    full_ds = BlobAndCirclesDataset()
    labels  = np.argmax(full_ds.y, axis=1)
    splitter = StratifiedShuffleSplit(n_splits=TRIALS_PER_CONFIG,
                                      test_size=0.20,
                                      random_state=SEED)

    trials = []
    for trial_idx, (tr_idx, val_idx) in enumerate(splitter.split(full_ds.X, labels)):
        set_all_seeds(SEED + 7000 + trial_idx)
        tr_ds  = torch.utils.data.Subset(full_ds, tr_idx.tolist())
        val_ds = torch.utils.data.Subset(full_ds, val_idx.tolist())
        tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        grid_path = base_dir / f"trial_{trial_idx}_grids.npz"
        out = run_trial(hidden_layers, lr, tr_loader, val_loader, grid_path)

        # collect everything under one per-trial dict
        trials.append({
            "trial":      trial_idx,
            "H0":         out["H0"],
            "H1":         out["H1"],
            "metrics":    out["metrics"],
            "grid_file":  out["grid_file"],
        })

    return {
        "hidden_layers": hidden_layers,
        "lr":            lr,
        "batch_size":    batch_size,
        "trials":        trials
    }
   

def save_json(obj: Dict, path: Path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

# ─────────────────────────────  experiments  ────────────────────────────── #
def experiment_depth():
    print("\n=== Experiment 1 : depth sweep ===")
    for depth in DEPTH_GRID:
        layers = [NEURONS_PER_LAYER] * depth
        tag    = f"depth_{depth}"
        print(f" → {tag}")
        cfg = collect_trials(layers, lr=LR_FIXED, batch_size=64, cfg_tag=tag)
        save_json(cfg, RESULTS_DIR / f"{tag}.json")
        print(f"   saved → {tag}.json + grids")


def experiment_lr():
    print("\n=== Experiment 2 : learning-rate sweep ===")
    for lr in LR_GRID:
        tag = f"lr_{lr:.0e}"
        print(f" → {tag}")
        cfg = collect_trials(ARCH_FIXED, lr=lr, batch_size=64, cfg_tag=tag)
        save_json(cfg, RESULTS_DIR / f"{tag}.json")
        print(f"   saved → {tag}.json + grids")


def experiment_batchsize():
    print("\n=== Experiment 3 : batch-size sweep ===")
    for bs in BATCH_GRID:
        tag = f"bs_{bs}"
        print(f" → {tag}")
        cfg = collect_trials(ARCH_FIXED, lr=LR_FIXED, batch_size=bs, cfg_tag=tag)
        save_json(cfg, RESULTS_DIR / f"{tag}.json")
        print(f"   saved → {tag}.json + grids")


# ────────────────────────────────  main  ──────────────────────────────── #
if __name__ == "__main__":
    start = datetime.now()
    set_all_seeds(SEED)

    experiment_depth()
    experiment_lr()
    experiment_batchsize()

    dt = datetime.now() - start
    print(f"\nAll experiments finished in {dt}.  Results in → {RESULTS_DIR.resolve()}")


