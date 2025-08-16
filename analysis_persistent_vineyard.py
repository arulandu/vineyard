#!/usr/bin/env python3
# analysis_persistent_vineyard.py
#
# Build visuals for the MOST-PERSISTENT vine (PV) signal that you computed into
# vineyard_metrics_persistent_vine/, using per-epoch metrics and raw snapshots
# from vineyard_results/. Outputs are written under most_persistent_vine_analysis/.
#
# What this script makes (H0/H1 kept strictly separate — never added):
#  1) Trajectory ribbons (median ± IQR) of PV cumulative VD by epoch, per tag.
#  2) PV path in the birth–death (BD) plane: exemplar + per-epoch median path.
#  3) Time-to-threshold vs PV-VD@threshold scatter with OLS fit (H1_std, H0_std).
#  4) Heatmaps (if possible) of median PV-VD@threshold and median epoch_100 across factors.
#  5) Phase portraits: PV-VD@threshold(H1,std) vs PV-VD@threshold(H0,std) colored by epoch_100.
#  6) Increment distributions (early vs late epochs) for PV increments.
#  7) Efficiency curves: val_acc(t) vs PV cumulative VD(t) (per trial faint + median).
#  8) Dominance map: fraction of trials where PV-VD@threshold(H1,std) > PV-VD@threshold(H0,std).
#  9) Calibration: bin PV-VD@threshold(H1,std) and plot mean final accuracy per bin.
# 10) Per-tag “profile card” small multiples.
# 11) Cross-tag ranks: median PV-VD@threshold(H1,std) vs median epoch_100.
# 12) Outlier trails: BD paths for slowest/weakest vs tag median BD path.
#
# Usage (typical):
#   python analysis_persistent_vineyard.py \
#       --metrics_dir vineyard_metrics_persistent_vine \
#       --results_dir vineyard_results \
#       --out_dir most_persistent_vine_analysis \
#       --acc_threshold 1.0 \
#       --inf_policy tag_global
#
# Notes:
# - Assumes your persistent-vine metrics CSVs exist per trial:
#       vineyard_metrics_persistent_vine/<tag>/trial_<k>_vineyard.csv
#   with columns: epoch,cum_unif_H0,cum_std_H0,cum_unif_H1,cum_std_H1
# - Reads vineyard_results/<tag>.json for per-epoch val_acc and raw diagrams (H0/H1 snapshots).
# - Infinity handling for BD-plane paths is configurable (fixed / tag_global / epoch_max).
# - Figures are saved under <out_dir>/figs_pv/... and summary CSVs under <out_dir>/tables/...
#
# Author: you ✨

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------- utils: tag parsing & dirs -------------------------

def parse_tag(tag: str):
    if tag.startswith("depth_"):
        try: return ("depth", float(tag.split("_",1)[1]))
        except: return ("depth", np.nan)
    if tag.startswith("lr_"):
        v = tag.split("_",1)[1]
        try: return ("lr", float(v))
        except:
            try: return ("lr", float(eval(v)))
            except: return ("lr", np.nan)
    if tag.startswith("bs_"):
        try: return ("batch", float(tag.split("_",1)[1]))
        except: return ("batch", np.nan)
    return ("other", np.nan)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ------------------------- weights & norms -------------------------

def w_uniform(_p: Tuple[float,float]) -> float:
    return 1.0

def w_standard(p: Tuple[float,float]) -> float:
    # Euclidean distance to diagonal y=x
    return abs(p[1] - p[0]) / math.sqrt(2.0)

def linf(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))


# ------------------------- data loading -------------------------

def load_trial_metrics_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists(): return None
    try:
        df = pd.read_csv(csv_path)
        # expected columns: epoch, cum_unif_H0, cum_std_H0, cum_unif_H1, cum_std_H1
        need = {"epoch","cum_unif_H0","cum_std_H0","cum_unif_H1","cum_std_H1"}
        if not need.issubset(df.columns):
            return None
        return df.sort_values("epoch").reset_index(drop=True)
    except Exception:
        return None

def load_results_json(json_path: Path) -> Optional[Dict]:
    if not json_path.exists(): return None
    try:
        return json.loads(json_path.read_text())
    except Exception:
        return None

def first_epoch_at_acc(metrics_list: List[Dict], threshold: float) -> Optional[int]:
    if not metrics_list: return None
    for idx, m in enumerate(metrics_list, start=1):  # epochs are 1..E
        try:
            if float(m.get("val_acc", 0.0)) >= threshold:
                return idx
        except Exception:
            pass
    return None


# ------------------------- infinity mapping for BD snapshots -------------------------

def to_float_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        s = str(v).strip().lower()
        if s in {"inf","+inf","infinity"}:  return math.inf
        if s in {"-inf","-infinity"}:       return -math.inf
        return float(v)
    except Exception:
        return None

def collect_finites(series_json: List[List[List[float]]]) -> Tuple[List[float], List[float]]:
    births, deaths = [], []
    for snap in series_json:
        for bd in snap:
            if not isinstance(bd, (list,tuple)) or len(bd) != 2:
                continue
            b, d = to_float_or_none(bd[0]), to_float_or_none(bd[1])
            if b is not None and np.isfinite(b): births.append(float(b))
            if d is not None and np.isfinite(d): deaths.append(float(d))
    return births, deaths

def tag_caps(series_json: List[List[List[float]]], fixed_cap: float, fixed_floor: float):
    b, d = collect_finites(series_json)
    d_cap = max(d) if d else fixed_cap
    b_floor = min(b) if b else fixed_floor
    return float(d_cap), float(b_floor)

def epoch_caps(series_json: List[List[List[float]]], fixed_cap: float, fixed_floor: float):
    T = len(series_json)
    d_caps = np.full(T, fixed_cap, dtype=float)
    b_floors = np.full(T, fixed_floor, dtype=float)
    for t, snap in enumerate(series_json):
        dd, bb = [], []
        for bd in snap:
            if not isinstance(bd, (list,tuple)) or len(bd) != 2:
                continue
            b, d = to_float_or_none(bd[0]), to_float_or_none(bd[1])
            if d is not None and np.isfinite(d): dd.append(float(d))
            if b is not None and np.isfinite(b): bb.append(float(b))
        if dd: d_caps[t] = max(dd)
        if bb: b_floors[t] = min(bb)
    return d_caps, b_floors

def map_bd(b: Optional[float], d: Optional[float], d_cap: float, b_floor: float) -> Optional[Tuple[float,float]]:
    if b is None or d is None: return None
    bb = b_floor if not np.isfinite(b) else float(b)
    dd = d_cap   if not np.isfinite(d) else float(d)
    if dd < bb: return None
    return (bb, dd)

def pick_top_bar_in_snapshot(snap: List[List[float]], d_cap: float, b_floor: float) -> Optional[Tuple[float,float]]:
    best, best_pers = None, -1.0
    for bd in snap:
        if not isinstance(bd, (list,tuple)) or len(bd) != 2:
            continue
        b, d = to_float_or_none(bd[0]), to_float_or_none(bd[1])
        mapped = map_bd(b, d, d_cap, b_floor)
        if mapped is None: continue
        bb, dd = mapped
        per = dd - bb
        if per > best_pers:
            best_pers = per
            best = (bb, dd)
    return best

def pv_bd_path(series_json: List[List[List[float]]], inf_policy: str, fixed_cap: float, fixed_floor: float) -> List[Optional[Tuple[float,float]]]:
    """Return a list of per-epoch (b,d) for the most persistent bar (after mapping)."""
    T = len(series_json)
    if inf_policy == "fixed":
        d_caps = np.full(T, fixed_cap, dtype=float)
        b_floors = np.full(T, fixed_floor, dtype=float)
    elif inf_policy == "tag_global":
        d_cap, b_floor = tag_caps(series_json, fixed_cap, fixed_floor)
        d_caps = np.full(T, d_cap, dtype=float)
        b_floors = np.full(T, b_floor, dtype=float)
    elif inf_policy == "epoch_max":
        d_caps, b_floors = epoch_caps(series_json, fixed_cap, fixed_floor)
    else:
        raise ValueError(f"Unknown inf_policy: {inf_policy}")

    pts = []
    for t in range(T):
        pts.append(pick_top_bar_in_snapshot(series_json[t], d_caps[t], b_floors[t]))
    return pts


# ------------------------- dataset stitching -------------------------

def build_trial_index(metrics_dir: Path, results_dir: Path, acc_threshold: float) -> pd.DataFrame:
    """
    Build a per-trial index: one row per (tag, trial) with:
      factor, value, epochs, epoch_100, final_acc, best_acc,
      paths to CSV and JSON; existence flags.
    """
    rows = []
    for jpath in sorted(results_dir.glob("*.json")):
        cfg = load_results_json(jpath)
        if not cfg: continue
        tag = jpath.stem
        factor, value = parse_tag(tag)
        trials = cfg.get("trials", [])
        for tr in trials:
            tid = int(tr.get("trial", -1))
            metrics = tr.get("metrics", [])
            epoch_100 = first_epoch_at_acc(metrics, acc_threshold)
            final_acc = float(metrics[-1]["val_acc"]) if metrics else np.nan
            best_acc  = float(max(m["val_acc"] for m in metrics)) if metrics else np.nan
            n_epochs  = len(metrics)
            csv_path  = metrics_dir / tag / f"trial_{tid}_vineyard.csv"
            rows.append({
                "tag": tag, "factor": factor, "value": value, "trial": tid,
                "epochs": n_epochs, "epoch_100": epoch_100,
                "final_acc": final_acc, "best_acc": best_acc,
                "csv_path": str(csv_path), "json_path": str(jpath)
            })
    df = pd.DataFrame(rows)
    return df


# ------------------------- plotting helpers -------------------------

def save_fig(fig: plt.Figure, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def median_iqr(arr: np.ndarray, axis=0):
    med = np.nanmedian(arr, axis=axis)
    q25 = np.nanpercentile(arr, 25, axis=axis)
    q75 = np.nanpercentile(arr, 75, axis=axis)
    return med, (q75 - q25)

def ribbon(ax, x, mat, label, ylabel):
    med, iqr = median_iqr(mat, axis=0)
    lo = med - 0.5*iqr
    hi = med + 0.5*iqr
    ax.plot(x, med, marker="o", label=label)
    ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True)


# ------------------------- analyses -------------------------

def analysis_1_ribbons(index_df: pd.DataFrame, out_dir: Path):
    """Trajectory ribbons per tag for {H0,H1}×{uniform,standard}."""
    for tag, sub in index_df.groupby("tag"):
        mats = {"H0_unif": [], "H0_std": [], "H1_unif": [], "H1_std": []}
        T = None
        for _, row in sub.iterrows():
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty: continue
            if T is None: T = int(dfm["epoch"].max()+1)
            mats["H0_unif"].append(dfm["cum_unif_H0"].values)
            mats["H0_std"].append(dfm["cum_std_H0"].values)
            mats["H1_unif"].append(dfm["cum_unif_H1"].values)
            mats["H1_std"].append(dfm["cum_std_H1"].values)
        if T is None: continue
        x = np.arange(T)
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)
        for ax, key, title in [
            (ax1,"H0_unif","H0 – uniform"),
            (ax2,"H0_std","H0 – standard"),
            (ax3,"H1_unif","H1 – uniform"),
            (ax4,"H1_std","H1 – standard"),
        ]:
            if len(mats[key]) == 0:
                ax.set_title(f"{title} (no data)"); ax.axis("off"); continue
            mat = np.vstack(mats[key])
            ribbon(ax, x, mat, label="median ± IQR", ylabel="PV cumulative VD")
            ax.set_title(title)
        fig.suptitle(f"PV cumulative VD trajectories — {tag}")
        save_fig(fig, out_dir / "ribbons" / f"{tag}_pv_ribbons.png")


def analysis_2_bd_paths(index_df: pd.DataFrame, out_dir: Path, inf_policy: str, fixed_cap: float, fixed_floor: float):
    """PV path in BD plane: exemplar + per-epoch median for H1 and H0."""
    for tag, sub in index_df.groupby("tag"):
        # Load raw snapshots once per tag
        cfg = load_results_json(Path(sub.iloc[0]["json_path"]))
        if not cfg: continue
        x_trials = cfg.get("trials", [])
        # assume all trials share epochs length
        # choose first non-empty trial as exemplar for H1/H0
        for Hk in ["H1", "H0"]:
            # Collect paths across trials
            paths: List[List[Optional[Tuple[float,float]]]] = []
            T = None
            exemplar = None
            for tr in x_trials:
                series = tr.get(Hk, [])
                if not series: continue
                if T is None: T = len(series)
                path = pv_bd_path(series, inf_policy=inf_policy, fixed_cap=fixed_cap, fixed_floor=fixed_floor)
                paths.append(path)
                if exemplar is None and all(p is not None for p in path):
                    exemplar = path
            if T is None or len(paths)==0:
                continue

            # Median path
            b_mat, d_mat = [], []
            for t in range(T):
                b_list, d_list = [], []
                for p in paths:
                    pt = p[t]
                    if pt is None: continue
                    b_list.append(pt[0]); d_list.append(pt[1])
                if b_list:
                    b_mat.append(np.median(b_list)); d_mat.append(np.median(d_list))
                else:
                    b_mat.append(np.nan); d_mat.append(np.nan)
            b_arr = np.array(b_mat); d_arr = np.array(d_mat)

            # Plot
            fig = plt.figure(figsize=(10,4))
            ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)

            # Exemplar (first fully-defined if available; else first)
            if exemplar is None: exemplar = paths[0]
            ex_b = [p[0] for p in exemplar if p is not None]
            ex_d = [p[1] for p in exemplar if p is not None]
            ax1.plot(ex_b, ex_d, marker="o")
            m = max(ex_b+ex_d+[0]); ax1.plot([0,m],[0,m], linestyle="--", alpha=0.5)  # diagonal
            ax1.set_xlabel("birth"); ax1.set_ylabel("death")
            ax1.set_title(f"{tag} — {Hk} PV BD path (exemplar)")
            ax1.grid(True)

            # Median path
            mask = ~np.isnan(b_arr) & ~np.isnan(d_arr)
            if np.any(mask):
                ax2.plot(b_arr[mask], d_arr[mask], marker="o")
                mm = max(np.nanmax(b_arr), np.nanmax(d_arr), 0.0)
                ax2.plot([0,mm],[0,mm], linestyle="--", alpha=0.5)
            ax2.set_xlabel("birth"); ax2.set_ylabel("death")
            ax2.set_title(f"{tag} — {Hk} PV BD path (median across trials)")
            ax2.grid(True)

            save_fig(fig, out_dir / "bd_paths" / f"{tag}_pv_bdpath_{Hk}.png")


def analysis_3_epoch100_scatter(index_df: pd.DataFrame, out_dir: Path, metrics_dir: Path):
    """epoch_100 vs PV-VD@100 (H1_std & H0_std), per tag scatter with OLS line."""
    def fit_line(x, y):
        x1 = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
        yhat = x1 @ beta
        return beta, yhat

    for tag, sub in index_df.groupby("tag"):
        rows = []
        for _, row in sub.iterrows():
            e100 = row["epoch_100"]
            if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty: continue
            if e100 not in dfm["epoch"].values: continue
            r = dfm[dfm["epoch"]==e100].iloc[0]
            rows.append({
                "trial": row["trial"],
                "epoch_100": float(e100),
                "H1_std_at100": float(r["cum_std_H1"]),
                "H0_std_at100": float(r["cum_std_H0"]),
            })
        if not rows: continue
        dat = pd.DataFrame(rows)

        for col, title in [("H1_std_at100","H1 (standard)"),
                           ("H0_std_at100","H0 (standard)")]:
            x = dat[col].values; y = dat["epoch_100"].values
            if len(x) < 3: continue
            beta, yhat = fit_line(x, y)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x, y, alpha=0.7)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xs, beta[0] + beta[1]*xs, linestyle="-")
            ax.set_xlabel(f"PV-VD@100 ({title})")
            ax.set_ylabel("epoch_100 (lower is better)")
            ax.set_title(f"{tag}: epoch_100 vs PV-VD@100 — {title}")
            ax.grid(True)
            save_fig(fig, out_dir / "epoch100_scatter" / f"{tag}_epoch100_vs_{col}.png")


def analysis_4_heatmaps(index_df: pd.DataFrame, out_dir: Path, metrics_dir: Path):
    """
    Heatmaps of median PV-VD@100 (H1_std) and median epoch_100 across factor values.
    Only draws when we can pivot (e.g., depth×batch). Missing combinations appear blank.
    """
    # Build a table: tag, factor, value, median_H1std_at100, median_epoch100
    rows = []
    for tag, sub in index_df.groupby("tag"):
        factor, value = parse_tag(tag)
        if factor == "other": continue
        at100_vals = []
        e100_vals  = []
        for _, row in sub.iterrows():
            e100 = row["epoch_100"]
            if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty or (e100 not in dfm["epoch"].values): continue
            v = float(dfm[dfm["epoch"]==e100]["cum_std_H1"])
            at100_vals.append(v); e100_vals.append(float(e100))
        if at100_vals:
            rows.append({
                "tag": tag, "factor": factor, "value": value,
                "median_H1std_at100": float(np.median(at100_vals)),
                "median_epoch_100":   float(np.median(e100_vals))
            })
    if not rows: return
    tab = pd.DataFrame(rows)

    # Try to build two-factor pivots opportunistically
    def attempt_pivot(primary: str, secondary: str, value_col: str, title: str, fname: str):
        a = tab[tab["factor"]==primary].copy()
        # we need a second factor: parse from tag names
        a["sec_factor"], a["sec_value"] = zip(*a["tag"].map(parse_tag))
        # keep where sec_factor==secondary (this only happens if tag names include both; often they don't)
        a = a[a["sec_factor"]==secondary]
        if a.empty: return
        piv = a.pivot_table(index="value", columns="sec_value", values=value_col, aggfunc="median")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(piv.values, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{v:g}" for v in piv.index])
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{v:g}" for v in piv.columns], rotation=45)
        ax.set_xlabel(secondary); ax.set_ylabel(primary); ax.set_title(title)
        fig.colorbar(im, ax=ax)
        save_fig(fig, out_dir / "heatmaps" / f"{fname}.png")

    # These will only render if your tags encode two axes (often not the case). Safe to try.
    attempt_pivot("depth","batch","median_H1std_at100","Median PV-VD@100 (H1,std): depth × batch","heatmap_H1std_depth_by_batch")
    attempt_pivot("depth","lr","median_H1std_at100","Median PV-VD@100 (H1,std): depth × lr","heatmap_H1std_depth_by_lr")
    attempt_pivot("depth","batch","median_epoch_100","Median epoch_100: depth × batch","heatmap_epoch100_depth_by_batch")
    attempt_pivot("depth","lr","median_epoch_100","Median epoch_100: depth × lr","heatmap_epoch100_depth_by_lr")


def analysis_5_phase_portrait(index_df: pd.DataFrame, out_dir: Path, metrics_dir: Path):
    """PV-VD@100(H1,std) vs PV-VD@100(H0,std), colored by epoch_100, per tag and pooled."""
    pooled = []
    for tag, sub in index_df.groupby("tag"):
        X, Y, C = [], [], []
        for _, row in sub.iterrows():
            e100 = row["epoch_100"]
            if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty or (e100 not in dfm["epoch"].values): continue
            r = dfm[dfm["epoch"]==e100].iloc[0]
            X.append(float(r["cum_std_H1"]))
            Y.append(float(r["cum_std_H0"]))
            C.append(float(e100))
            pooled.append((tag, float(r["cum_std_H1"]), float(r["cum_std_H0"]), float(e100)))
        if len(X) >= 3:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sc = ax.scatter(X, Y, c=C)
            ax.set_xlabel("PV-VD@100 (H1,std)")
            ax.set_ylabel("PV-VD@100 (H0,std)")
            ax.set_title(f"{tag}: Phase portrait colored by epoch_100")
            fig.colorbar(sc, ax=ax, label="epoch_100")
            ax.grid(True)
            save_fig(fig, out_dir / "phase" / f"{tag}_phase_H1std_vs_H0std.png")

    # pooled
    if pooled:
        arr = np.array(pooled, dtype=object)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(arr[:,1].astype(float), arr[:,2].astype(float), c=arr[:,3].astype(float))
        ax.set_xlabel("PV-VD@100 (H1,std)"); ax.set_ylabel("PV-VD@100 (H0,std)")
        ax.set_title("Pooled: Phase portrait colored by epoch_100")
        fig.colorbar(sc, ax=ax, label="epoch_100")
        ax.grid(True)
        save_fig(fig, out_dir / "phase" / f"pooled_phase_H1std_vs_H0std.png")


def analysis_6_increment_distributions(index_df: pd.DataFrame, out_dir: Path, early_k: int = 10):
    """Boxplots of PV increments (ΔVD) for early vs late epochs, per tag & signal."""
    for tag, sub in index_df.groupby("tag"):
        incs = {"H0_unif_early":[], "H0_unif_late":[],
                "H0_std_early":[],  "H0_std_late":[],
                "H1_unif_early":[], "H1_unif_late":[],
                "H1_std_early":[],  "H1_std_late":[]}
        T = None
        for _, row in sub.iterrows():
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty: continue
            if T is None: T = int(dfm["epoch"].max()+1)
            for key, col in [("H0_unif","cum_unif_H0"), ("H0_std","cum_std_H0"),
                             ("H1_unif","cum_unif_H1"), ("H1_std","cum_std_H1")]:
                v = dfm[col].values
                dv = np.diff(v)  # length T-1
                k = min(early_k, len(dv)//2 if len(dv)>0 else 0)
                if k == 0: continue
                incs[f"{key}_early"].extend(dv[:k])
                incs[f"{key}_late"].extend(dv[-k:])

        if T is None: continue
        # plot 8 boxes in two rows
        labels = list(incs.keys())
        data = [incs[k] for k in labels if len(incs[k])>0]
        if not data: continue
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        ax.boxplot(data, labels=[k.replace("_"," ") for k in labels if len(incs[k])>0], vert=True)
        ax.set_ylabel("PV increment (ΔVD per epoch)")
        ax.set_title(f"{tag}: PV increments early vs late")
        ax.grid(True, axis="y", linestyle=":")
        plt.xticks(rotation=30)
        save_fig(fig, out_dir / "increments" / f"{tag}_pv_increments_early_late.png")


def analysis_7_efficiency_curves(index_df: pd.DataFrame, out_dir: Path):
    """val_acc(t) vs PV cumulative VD(t) with per-trial lines (faint) and median line."""
    for tag, sub in index_df.groupby("tag"):
        # load metrics json once
        cfg = load_results_json(Path(sub.iloc[0]["json_path"]))
        if not cfg: continue
        trials = cfg.get("trials", [])
        # We'll do H1_std and H0_std
        for sig, col in [("H1_std","cum_std_H1"), ("H0_std","cum_std_H0")]:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            all_pairs = []
            for _, row in sub.iterrows():
                dfm = load_trial_metrics_csv(Path(row["csv_path"]))
                if dfm is None or dfm.empty: continue
                # find this trial's val_acc series
                trial_obj = next((t for t in trials if int(t.get("trial",-999))==int(row["trial"])), None)
                if trial_obj is None: continue
                acc = [float(m["val_acc"]) for m in trial_obj.get("metrics", [])]
                x = dfm[col].values
                # align lengths (assume dfm epoch starts at 0 and includes T entries; metrics length T)
                T = min(len(x), len(acc))
                if T < 2: continue
                all_pairs.append((x[:T], np.array(acc[:T], dtype=float)))
                ax.plot(x[:T], acc[:T], alpha=0.15)
            if not all_pairs:
                plt.close(fig); continue
            # median curve across trials at common x-quantiles is tricky; we approximate by epoch-wise median via reindexing by epoch-length
            # As a simpler proxy: take median across trials at each epoch position (using re-sampled x via np.interp on a common grid)
            # Build common grid from 0..max median x
            # First, interpolate each trial’s acc over its x to a common grid of 100 points
            xmax = np.nanmedian([xp[-1] for xp,_ in all_pairs])
            grid = np.linspace(0, xmax, 100)
            acc_interp = []
            for xp, yp in all_pairs:
                # ensure xp is non-decreasing
                order = np.argsort(xp)
                xp2 = xp[order]; yp2 = yp[order]
                # clip/unique
                mask = np.concatenate([[True], np.diff(xp2)>1e-12])
                xp2 = xp2[mask]; yp2 = yp2[mask]
                if len(xp2) < 2: continue
                yi = np.interp(grid, xp2, yp2, left=yp2[0], right=yp2[-1])
                acc_interp.append(yi)
            if acc_interp:
                acc_interp = np.vstack(acc_interp)
                med = np.nanmedian(acc_interp, axis=0)
                ax.plot(grid, med, linewidth=2.0, label="median", alpha=0.9)
                ax.legend()
            ax.set_xlabel(f"PV cumulative VD ({sig.replace('_',' / ')})")
            ax.set_ylabel("validation accuracy")
            ax.set_title(f"{tag}: Efficiency curve (PV-only)")
            ax.grid(True)
            save_fig(fig, out_dir / "efficiency" / f"{tag}_efficiency_{sig}.png")


def analysis_8_dominance(index_df: pd.DataFrame, out_dir: Path):
    """Per-tag fraction of trials where PV-VD@100(H1,std) > PV-VD@100(H0,std)."""
    rows = []
    for tag, sub in index_df.groupby("tag"):
        wins = 0; total = 0
        for _, row in sub.iterrows():
            e100 = row["epoch_100"]
            if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty or (e100 not in dfm["epoch"].values): continue
            r = dfm[dfm["epoch"]==e100].iloc[0]
            h1 = float(r["cum_std_H1"]); h0 = float(r["cum_std_H0"])
            if np.isfinite(h1) and np.isfinite(h0):
                total += 1
                if h1 > h0: wins += 1
        if total > 0:
            frac = wins/total
            rows.append({"tag": tag, "frac_H1std_gt_H0std": frac, "n": total})

    if not rows: return
    tab = pd.DataFrame(rows).sort_values("tag")
    # bar plot
    fig = plt.figure(figsize=(max(6, 0.25*len(tab)),4))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(tab)), tab["frac_H1std_gt_H0std"].values)
    ax.set_xticks(np.arange(len(tab)))
    ax.set_xticklabels(tab["tag"].tolist(), rotation=45, ha="right")
    ax.set_ylim(0,1)
    ax.set_ylabel("fraction of trials (H1,std > H0,std)")
    ax.set_title("Dominance of H1 (standard) at 100%")
    ax.grid(True, axis="y", linestyle=":")
    save_fig(fig, out_dir / "dominance" / "dominance_bar.png")

    ensure_dir(out_dir / "tables")
    tab.to_csv(out_dir / "tables" / "dominance_table.csv", index=False)


def analysis_9_calibration(index_df: pd.DataFrame, out_dir: Path, bins: int = 10):
    """Calibration: bin PV-VD@100(H1,std), plot mean final acc per bin (pooled)."""
    vals, accs = [], []
    for _, row in index_df.iterrows():
        e100 = row["epoch_100"]
        if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
        dfm = load_trial_metrics_csv(Path(row["csv_path"]))
        if dfm is None or dfm.empty or (e100 not in dfm["epoch"].values): continue
        r = dfm[dfm["epoch"]==e100].iloc[0]
        v = float(r["cum_std_H1"]); a = float(row["final_acc"])
        if np.isfinite(v) and np.isfinite(a):
            vals.append(v); accs.append(a)
    if len(vals) < max(5, bins): return
    vals = np.array(vals); accs = np.array(accs)
    qs = np.quantile(vals, np.linspace(0,1,bins+1))
    centers = 0.5*(qs[:-1]+qs[1:])
    means, stds, ns = [], [], []
    for i in range(bins):
        mask = (vals >= qs[i]) & (vals <= qs[i+1] if i==bins-1 else vals < qs[i+1])
        if not np.any(mask):
            means.append(np.nan); stds.append(np.nan); ns.append(0); continue
        means.append(float(np.mean(accs[mask])))
        stds.append(float(np.std(accs[mask])))
        ns.append(int(np.sum(mask)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(centers, means, yerr=stds, fmt="o-")
    ax.set_xlabel("PV-VD@100 (H1,std) — quantile centers")
    ax.set_ylabel("mean final accuracy")
    ax.set_title("Calibration of PV-VD@100 (H1,std) vs final accuracy (pooled)")
    ax.grid(True)
    save_fig(fig, out_dir / "calibration" / "calibration_h1std_finalacc.png")

    ensure_dir(out_dir / "tables")
    pd.DataFrame({"bin_center":centers, "mean_final_acc":means, "std_final_acc":stds, "n":ns}).to_csv(
        out_dir / "tables" / "calibration_h1std_finalacc.csv", index=False
    )


def analysis_10_profile_cards(index_df: pd.DataFrame, out_dir: Path, inf_policy: str, fixed_cap: float, fixed_floor: float):
    """Per-tag 2×2 small-multiple card: (H1-std ribbon, H1 BD median path, acc-vs-VD scatter, histogram PV-VD@100)."""
    for tag, sub in index_df.groupby("tag"):
        # (a) H1-std ribbon
        mat = []
        # (b) H1 BD median path
        cfg = load_results_json(Path(sub.iloc[0]["json_path"]))
        H1_paths = []
        # (c) val_acc vs VD over epochs points pooled
        all_pairs = []
        # (d) histogram PV-VD@100
        vd100 = []
        if cfg:
            T_raw = None
            for tr in cfg.get("trials", []):
                series = tr.get("H1", [])
                if not series: continue
                if T_raw is None: T_raw = len(series)
                H1_paths.append(pv_bd_path(series, inf_policy, fixed_cap, fixed_floor))

        trials_cfg = cfg.get("trials", []) if cfg else []
        for _, row in sub.iterrows():
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty: continue
            mat.append(dfm["cum_std_H1"].values)
            # acc vs VD
            tr_obj = next((t for t in trials_cfg if int(t.get("trial",-999))==int(row["trial"])), None)
            if tr_obj is not None:
                acc = np.array([float(m["val_acc"]) for m in tr_obj.get("metrics", [])], dtype=float)
                T = min(len(acc), len(dfm))
                all_pairs.append((dfm["cum_std_H1"].values[:T], acc[:T]))
            # histogram at 100
            e100 = row["epoch_100"]
            if e100 is not None and not (isinstance(e100,float) and np.isnan(e100)) and e100 in dfm["epoch"].values:
                vd100.append(float(dfm[dfm["epoch"]==e100]["cum_std_H1"]))

        if not mat: continue
        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)

        # (a) ribbon
        x = np.arange(len(mat[0]))
        ribbon(ax1, x, np.vstack(mat), label="median ± IQR", ylabel="PV cumulative VD (H1,std)")
        ax1.set_title(f"{tag}: H1-std PV trajectory")

        # (b) median BD path
        if H1_paths:
            # per-epoch medians
            T = max(len(p) for p in H1_paths)
            b_med, d_med = [], []
            for t in range(T):
                bs, ds = [], []
                for p in H1_paths:
                    if t < len(p) and p[t] is not None:
                        bs.append(p[t][0]); ds.append(p[t][1])
                if bs:
                    b_med.append(float(np.median(bs))); d_med.append(float(np.median(ds)))
            if b_med:
                ax2.plot(b_med, d_med, marker="o")
                m = max(max(b_med), max(d_med), 0.0)
                ax2.plot([0,m],[0,m], linestyle="--", alpha=0.5)
            ax2.set_xlabel("birth"); ax2.set_ylabel("death"); ax2.set_title("H1 PV median BD path")
            ax2.grid(True)
        else:
            ax2.text(0.5,0.5,"no BD data",ha="center",va="center"); ax2.axis("off")

        # (c) acc vs VD pairs
        for xp, yp in all_pairs:
            ax3.plot(xp, yp, alpha=0.15)
        if all_pairs:
            xmax = np.nanmedian([xp[-1] for xp,_ in all_pairs])
            grid = np.linspace(0, xmax, 100)
            interp = []
            for xp,yp in all_pairs:
                ord = np.argsort(xp)
                xp2 = xp[ord]; yp2 = yp[ord]
                mask = np.concatenate([[True], np.diff(xp2)>1e-12])
                xp2 = xp2[mask]; yp2 = yp2[mask]
                if len(xp2) >= 2:
                    interp.append(np.interp(grid, xp2, yp2, left=yp2[0], right=yp2[-1]))
            if interp:
                med = np.nanmedian(np.vstack(interp), axis=0)
                ax3.plot(grid, med, linewidth=2.0)
        ax3.set_xlabel("PV cumulative VD (H1,std)"); ax3.set_ylabel("val acc"); ax3.set_title("Efficiency (PV-only)")
        ax3.grid(True)

        # (d) histogram
        if vd100:
            ax4.hist(vd100, bins=12)
        ax4.set_xlabel("PV-VD@100 (H1,std)"); ax4.set_ylabel("count"); ax4.set_title("Histogram @100%")
        ax4.grid(True)

        fig.suptitle(f"Profile card — {tag}")
        save_fig(fig, out_dir / "profiles" / f"{tag}_profile.png")


def analysis_11_cross_tag_rank(index_df: pd.DataFrame, out_dir: Path, metrics_dir: Path):
    """Rank tags by median PV-VD@100(H1,std) and overlay median epoch_100."""
    rows = []
    for tag, sub in index_df.groupby("tag"):
        vals = []; e100s = []
        for _, row in sub.iterrows():
            e100 = row["epoch_100"]
            if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
            dfm = load_trial_metrics_csv(Path(row["csv_path"]))
            if dfm is None or dfm.empty or (e100 not in dfm["epoch"].values): continue
            v = float(dfm[dfm["epoch"]==e100]["cum_std_H1"])
            vals.append(v); e100s.append(float(e100))
        if vals:
            rows.append({"tag":tag, "median_H1std_at100":float(np.median(vals)), "median_epoch_100":float(np.median(e100s))})
    if not rows: return
    tab = pd.DataFrame(rows).sort_values("median_H1std_at100")
    x = np.arange(len(tab))
    fig = plt.figure(figsize=(max(6, 0.3*len(tab)),4))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, tab["median_H1std_at100"].values, marker="o", label="median PV-VD@100 (H1,std)")
    ax1.set_ylabel("PV-VD@100 (H1,std)")
    ax1.set_xticks(x); ax1.set_xticklabels(tab["tag"].tolist(), rotation=45, ha="right")
    ax1.grid(True, axis="y", linestyle=":")
    # overlay epoch_100 on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, tab["median_epoch_100"].values, marker="s", linestyle="--", label="median epoch_100", color="C1")
    ax2.set_ylabel("epoch_100")
    fig.legend(loc="upper left")
    fig.suptitle("Cross-tag ranks: PV-VD@100 (H1,std) and epoch_100")
    save_fig(fig, out_dir / "cross_tag" / "rank_vd100_epoch100.png")

    ensure_dir(out_dir / "tables")
    tab.to_csv(out_dir / "tables" / "cross_tag_rank.csv", index=False)


def analysis_12_outlier_trails(index_df: pd.DataFrame, out_dir: Path, inf_policy: str, fixed_cap: float, fixed_floor: float, worst_frac: float = 0.1):
    """Overlay BD paths for slowest (largest epoch_100) trials vs tag median path (H1)."""
    for tag, sub in index_df.groupby("tag"):
        # collect epoch_100 and BD paths
        cfg = load_results_json(Path(sub.iloc[0]["json_path"]))
        if not cfg: continue
        trial_map = {int(tr["trial"]): tr for tr in cfg.get("trials", [])}
        rows = []
        for _, row in sub.iterrows():
            e100 = row["epoch_100"]
            if e100 is None or (isinstance(e100,float) and np.isnan(e100)): continue
            tr = trial_map.get(int(row["trial"]))
            if tr is None: continue
            series = tr.get("H1", [])
            if not series: continue
            path = pv_bd_path(series, inf_policy, fixed_cap, fixed_floor)
            rows.append({"trial": int(row["trial"]), "epoch_100": float(e100), "path": path})
        if len(rows) < 3: continue
        # pick worst decile
        rows_sorted = sorted(rows, key=lambda r: r["epoch_100"], reverse=True)
        k = max(1, int(len(rows_sorted)*worst_frac))
        worst = rows_sorted[:k]
        # median path
        T = max(len(r["path"]) for r in rows)
        b_med, d_med = [], []
        for t in range(T):
            bs, ds = [], []
            for r in rows:
                if t < len(r["path"]) and r["path"][t] is not None:
                    bs.append(r["path"][t][0]); ds.append(r["path"][t][1])
            if bs:
                b_med.append(float(np.median(bs))); d_med.append(float(np.median(ds)))
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # worst trails
        for w in worst:
            pts = [p for p in w["path"] if p is not None]
            if len(pts) >= 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color="gray", alpha=0.35)
        # median
        if b_med:
            ax.plot(b_med, d_med, color="C1", linewidth=2.0, label="median path (all trials)")
        m = 0.0
        if b_med: m = max(m, max(b_med))
        if d_med: m = max(m, max(d_med))
        ax.plot([0,m],[0,m], linestyle="--", alpha=0.5)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.set_title(f"{tag}: Outlier H1 PV BD trails (slowest decile) vs median path")
        ax.grid(True); ax.legend()
        save_fig(fig, out_dir / "outliers" / f"{tag}_outlier_trails.png")


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Most-persistent-vine (PV) analysis suite.")
    ap.add_argument("--metrics_dir", type=Path, default=Path("vineyard_metrics_persistent_vine"))
    ap.add_argument("--results_dir", type=Path, default=Path("vineyard_results"))
    ap.add_argument("--out_dir",     type=Path, default=Path("most_persistent_vine_analysis"))
    ap.add_argument("--acc_threshold", type=float, default=1.0, help="Accuracy threshold for epoch_100 (e.g., 1.0 or 0.975).")
    ap.add_argument("--inf_policy", choices=["fixed","tag_global","epoch_max"], default="fixed",
                    help="How to map infinite births/deaths when drawing BD paths.")
    ap.add_argument("--fixed_cap",   type=float, default=1.0, help="Death cap (if inf_policy uses caps).")
    ap.add_argument("--fixed_floor", type=float, default=0.0, help="Birth floor (if inf_policy uses floors).")
    ap.add_argument("--early_k",     type=int, default=10, help="Epoch window for 'early'/'late' increment summaries.")
    ap.add_argument("--bins",        type=int, default=10, help="Bins for calibration curve.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Build index (per tag/trial)
    index_df = build_trial_index(args.metrics_dir, args.results_dir, args.acc_threshold)
    if index_df.empty:
        print(f"No trials found under {args.results_dir} and {args.metrics_dir}.")
        return

    # Save index for reference
    ensure_dir(Path(args.out_dir) / "tables")
    index_df.to_csv(Path(args.out_dir) / "tables" / "trial_index.csv", index=False)

    # Analyses
    analysis_1_ribbons(index_df, Path(args.out_dir))
    analysis_2_bd_paths(index_df, Path(args.out_dir), args.inf_policy, args.fixed_cap, args.fixed_floor)
    analysis_3_epoch100_scatter(index_df, Path(args.out_dir), args.metrics_dir)
    analysis_4_heatmaps(index_df, Path(args.out_dir), args.metrics_dir)
    analysis_5_phase_portrait(index_df, Path(args.out_dir), args.metrics_dir)
    analysis_6_increment_distributions(index_df, Path(args.out_dir), early_k=args.early_k)
    analysis_7_efficiency_curves(index_df, Path(args.out_dir))
    analysis_8_dominance(index_df, Path(args.out_dir))
    analysis_9_calibration(index_df, Path(args.out_dir), bins=args.bins)
    analysis_10_profile_cards(index_df, Path(args.out_dir), args.inf_policy, args.fixed_cap, args.fixed_floor)
    analysis_11_cross_tag_rank(index_df, Path(args.out_dir), args.metrics_dir)
    analysis_12_outlier_trails(index_df, Path(args.out_dir), args.inf_policy, args.fixed_cap, args.fixed_floor)

    print(f"All PV analyses written to: {Path(args.out_dir).resolve()}")

if __name__ == "__main__":
    main()
