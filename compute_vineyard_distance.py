#!/usr/bin/env python3
"""
compute_vineyard_distance.py
------------------------------------------------------------
Compute epoch-wise cumulative vineyard distances (NO H0+H1 SUMS)
for saved vineyards (H0, H1) produced by your experiment runner.

- Uses GUDHI's optimal matching: W_{∞,1} with matching=True.
- Handles essentials by capping death=None -> CAP (default 1.0).
- Optional: drop the longest H0 bar per snapshot (global component).
- Outputs one CSV per trial with cumulative curves, keeping H0 and H1 separate.

Columns:
  epoch, cum_unif_H0, cum_std_H0, cum_unif_H1, cum_std_H1

Usage:
  python compute_vineyard_distance.py \
      --results_dir vineyard_results \
      --out_dir vineyard_metrics \
      --cap 1.0 \
      [--drop_longest_h0]

Requires: pip install gudhi numpy
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gudhi.wasserstein import wasserstein_distance

# --------------------------- weights ---------------------------

def w_uniform(_p: np.ndarray) -> float:
    """Uniform weight: 1."""
    return 1.0

def w_standard(p: np.ndarray) -> float:
    """Standard weight: Euclidean distance to the diagonal."""
    return abs(p[1] - p[0]) / math.sqrt(2.0)

# ------------- L∞ distances (between points / to diag) --------

def linf(a: np.ndarray, b: np.ndarray) -> float:
    """L∞ distance between two BD points."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))

def linf_to_diag(p: np.ndarray) -> float:
    """L∞ distance from a BD point to the diagonal y=x."""
    return float(abs(p[1] - p[0]) / 2.0)

# ---------------- snapshot loading / cleaning ------------------

def _to_float_or_none(v):
    """JSON null -> None; strings 'inf'/'-inf' handled; else float."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"inf", "+inf", "infinity"}:
            return math.inf
        if s in {"-inf", "-infinity"}:
            return -math.inf
    return float(v)

def snapshot_to_array_cap_infinity(snap: List[List[float]], cap_value: float, birth_floor: float = 0.0) -> np.ndarray:
    """
    Convert snapshot [[birth, death], ...] to ndarray (k x 2), replacing
    death=None (∞) -> cap_value, and birth=None (-∞) -> birth_floor.
    Filters non-finite or invalid rows (requires death >= birth).
    """
    out = []
    for b, d in snap:
        b = _to_float_or_none(b)
        d = _to_float_or_none(d)
        bb = birth_floor if b is None or not np.isfinite(b) else float(b)
        dd = cap_value   if d is None or not np.isfinite(d) else float(d)
        if np.isfinite(bb) and np.isfinite(dd) and dd >= bb:
            out.append((bb, dd))
    if not out:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(out, dtype=float)

def drop_longest_bar(diag: np.ndarray) -> np.ndarray:
    """Remove the single longest bar (often the global H0 component)."""
    if diag.size == 0:
        return diag
    lifetimes = diag[:, 1] - diag[:, 0]
    keep = np.ones(len(diag), dtype=bool)
    keep[int(np.argmax(lifetimes))] = False
    return diag[keep]

def load_series(diag_series_json: List[List[List[float]]], cap_value: float, drop_longest: bool = False) -> List[np.ndarray]:
    """List of snapshots -> list of (k_t x 2) arrays with capping applied."""
    series = [snapshot_to_array_cap_infinity(s, cap_value=cap_value) for s in diag_series_json]
    if drop_longest:
        series = [drop_longest_bar(d) for d in series]
    return series

# ---------------------- per-step computation -------------------

def step_weighted(A: np.ndarray, B: np.ndarray, weight_fn) -> float:
    """
    One-step weighted vineyard distance between diagrams A (prev) and B (curr),
    using optimal W_{∞,1} matching (with diagonal) from GUDHI.
    Weights are evaluated at the SOURCE point (right Riemann sum).
    """
    m, n = len(A), len(B)

    # Fast paths for empty diagrams
    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        # births only: diag -> B
        w_diag = 0.0 if weight_fn is w_standard else 1.0
        return float(np.sum([w_diag * linf_to_diag(B[j]) for j in range(n)]))
    if n == 0:
        # deaths only: A -> diag
        return float(np.sum([weight_fn(A[i]) * linf_to_diag(A[i]) for i in range(m)]))

    # Both non-empty: use GUDHI's matching
    _, matching = wasserstein_distance(
        A, B, matching=True, order=1.0, internal_p=np.inf  # sum with L∞ ground metric
    )

    total = 0.0
    for i, j in matching:
        if i >= 0 and j >= 0:
            c = linf(A[i], B[j])
            w = weight_fn(A[i])  # weight at previous point
            total += w * c
        elif i >= 0 and j < 0:
            total += weight_fn(A[i]) * linf_to_diag(A[i])  # death
        elif i < 0 and j >= 0:
            total += (0.0 if weight_fn is w_standard else 1.0) * linf_to_diag(B[j])  # birth
    return float(total)

def cumulative_series(series: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cum_unif, cum_std), both arrays of length T."""
    T = len(series)
    cum_u = np.zeros(T, dtype=float)
    cum_s = np.zeros(T, dtype=float)
    for t in range(1, T):
        du = step_weighted(series[t - 1], series[t], w_uniform)
        ds = step_weighted(series[t - 1], series[t], w_standard)
        cum_u[t] = cum_u[t - 1] + du
        cum_s[t] = cum_s[t - 1] + ds
    return cum_u, cum_s

# --------------------------- driver ----------------------------

def process_config(json_path: Path, out_root: Path, cap_value: float, drop_longest_h0: bool):
    cfg = json.loads(json_path.read_text())
    tag = json_path.stem

    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for tr in cfg.get("trials", []):
        tid = tr["trial"]

        # Load series (apply capping); optionally drop longest H0 bar
        H0 = load_series(tr["H0"], cap_value=cap_value, drop_longest=drop_longest_h0)
        H1 = load_series(tr["H1"], cap_value=cap_value, drop_longest=False)

        # Compute cumulative curves per homology (NO SUMS ACROSS H0/H1)
        cum_unif_H0, cum_std_H0 = cumulative_series(H0)
        cum_unif_H1, cum_std_H1 = cumulative_series(H1)

        epochs = np.arange(len(cum_unif_H0), dtype=int)

        # Save CSV (no H0+H1 totals)
        out_file = out_dir / f"trial_{tid}_vineyard.csv"
        header = "epoch,cum_unif_H0,cum_std_H0,cum_unif_H1,cum_std_H1"
        data = np.column_stack([epochs, cum_unif_H0, cum_std_H0, cum_unif_H1, cum_std_H1])
        np.savetxt(out_file, data, delimiter=",", header=header, comments="", fmt="%.10g")
        print(f"[{tag}] wrote {out_file.name} (T={len(epochs)})")

def main():
    ap = argparse.ArgumentParser(description="Compute cumulative vineyard distances (H0 and H1 kept separate).")
    ap.add_argument("--results_dir", type=Path, default=Path("vineyard_results_1"), help="Directory with *.json experiment files.")
    ap.add_argument("--out_dir",     type=Path, default=Path("vineyard_metrics_higher_cap_1"), help="Output directory for CSVs.")
    ap.add_argument("--cap",         type=float, default=8.0, help="Cap for death=None (∞) -> CAP (same units as filtration).")
    ap.add_argument("--drop_longest_h0", action="store_true", help="Drop the longest H0 bar per snapshot (global component).")
    args = ap.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(args.results_dir.glob("*.json"))
    if not json_files:
        print(f"No *.json files found in {args.results_dir}")
        return

    for jp in json_files:
        process_config(jp, args.out_dir, cap_value=args.cap, drop_longest_h0=args.drop_longest_h0)

if __name__ == "__main__":
    main()
