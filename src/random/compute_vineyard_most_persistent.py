#!/usr/bin/env python3
"""
compute_persistent_vine_vd_infty.py

Compute cumulative vineyard distance (uniform & standard) for the
single MOST-PERSISTENT vine (per epoch, no matching), while mapping
infinite births/deaths to finite caps/floors instead of dropping them.

Inputs (read-only):
  vineyard_results/<tag>.json
    trials[*].H0, trials[*].H1 : list over epochs of lists of [birth, death]
      where 'null' denotes ±inf as serialized by your pipeline.

Outputs:
  vineyard_metrics_persistent_vine/<tag>/trial_<id>_vineyard.csv
    columns: epoch,cum_unif_H0,cum_std_H0,cum_unif_H1,cum_std_H1

Infinity handling (choose with --inf_policy):
  - fixed         : death -> fixed cap (default 1.0), birth -> fixed floor (default 0.0)
  - tag_global    : per tag & homology, death -> max finite death seen in that tag;
                    birth -> min finite birth seen in that tag (fallback to fixed if none)
  - epoch_max     : per epoch & homology, death -> max finite death in that epoch;
                    birth -> min finite birth in that epoch (fallback to fixed if none)

Notes
-----
- We select the top-persistence bar *after* mapping infinities.
- Distance in BD-plane is L∞; standard weight = Euclidean dist to diagonal.
- H0 and H1 are NEVER added together.
"""

import argparse, json, math
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import numpy as np

# ------------- norms & weights -------------

def linf(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

def w_uniform(_p: Tuple[float,float]) -> float:
    return 1.0

def w_standard(p: Tuple[float,float]) -> float:
    # Euclidean distance to the diagonal y=x
    return abs(p[1] - p[0]) / math.sqrt(2.0)

# ------------- parsing helpers -------------

def to_float_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        s = str(v).strip().lower()
        if s in {"inf", "+inf", "infinity"}: return math.inf
        if s in {"-inf", "-infinity"}:       return -math.inf
        return float(v)
    except Exception:
        return None

# ------------- infinity mapping -------------

def collect_finites(series_json: List[List[List[float]]]) -> Tuple[List[float], List[float]]:
    """Collect finite births and finite deaths across the whole series."""
    births, deaths = [], []
    for snap in series_json:
        for bd in snap:
            if not isinstance(bd, (list, tuple)) or len(bd) != 2: 
                continue
            b, d = to_float_or_none(bd[0]), to_float_or_none(bd[1])
            if b is not None and np.isfinite(b): births.append(float(b))
            if d is not None and np.isfinite(d): deaths.append(float(d))
    return births, deaths

def compute_caps_for_tag(series_json: List[List[List[float]]],
                         fixed_cap: float, fixed_floor: float) -> Tuple[float, float]:
    """Return (death_cap, birth_floor) for tag_global policy; fallback to fixed if no finites."""
    births, deaths = collect_finites(series_json)
    d_cap = max(deaths) if deaths else fixed_cap
    b_floor = min(births) if births else fixed_floor
    return float(d_cap), float(b_floor)

def compute_caps_per_epoch(series_json: List[List[List[float]]],
                           fixed_cap: float, fixed_floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays per epoch for epoch_max policy."""
    T = len(series_json)
    d_caps = np.full(T, fixed_cap, dtype=float)
    b_floors = np.full(T, fixed_floor, dtype=float)
    for t, snap in enumerate(series_json):
        d_list, b_list = [], []
        for bd in snap:
            if not isinstance(bd, (list, tuple)) or len(bd) != 2:
                continue
            b, d = to_float_or_none(bd[0]), to_float_or_none(bd[1])
            if d is not None and np.isfinite(d): d_list.append(float(d))
            if b is not None and np.isfinite(b): b_list.append(float(b))
        if d_list: d_caps[t] = max(d_list)
        if b_list: b_floors[t] = min(b_list)
    return d_caps, b_floors

def map_bd(b: Optional[float], d: Optional[float],
           d_cap: float, b_floor: float) -> Optional[Tuple[float,float]]:
    """Map (b,d) to finite using caps/floors; return None if cannot."""
    if b is None or d is None:
        return None
    bb = b_floor if not np.isfinite(b) else float(b)
    dd = d_cap   if not np.isfinite(d) else float(d)
    if dd < bb:
        return None
    return (bb, dd)

# ------------- top-vine selection & accumulation -------------

def pick_top_bar_in_snapshot(
    snap: List[List[float]],
    d_cap: float,
    b_floor: float
) -> Optional[Tuple[float,float]]:
    """Pick (b,d) with max lifetime (after mapping)."""
    best = None
    best_pers = -1.0
    for bd in snap:
        if not isinstance(bd, (list, tuple)) or len(bd) != 2:
            continue
        b, d = to_float_or_none(bd[0]), to_float_or_none(bd[1])
        mapped = map_bd(b, d, d_cap, b_floor)
        if mapped is None:
            continue
        bb, dd = mapped
        pers = dd - bb
        if pers > best_pers:
            best_pers = pers
            best = (bb, dd)
    return best

def cumulative_for_top_vine(
    series_json: List[List[List[float]]],
    inf_policy: str,
    fixed_cap: float,
    fixed_floor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative VD for the most persistent vine (per epoch) for both weights.
    inf_policy in {'fixed','tag_global','epoch_max'}.
    """
    T = len(series_json)
    cum_u = np.zeros(T, dtype=float)
    cum_s = np.zeros(T, dtype=float)

    # Determine caps/floors
    if inf_policy == "fixed":
        d_caps = np.full(T, fixed_cap, dtype=float)
        b_floors = np.full(T, fixed_floor, dtype=float)
    elif inf_policy == "tag_global":
        d_cap, b_floor = compute_caps_for_tag(series_json, fixed_cap, fixed_floor)
        d_caps = np.full(T, d_cap, dtype=float)
        b_floors = np.full(T, b_floor, dtype=float)
    elif inf_policy == "epoch_max":
        d_caps, b_floors = compute_caps_per_epoch(series_json, fixed_cap, fixed_floor)
    else:
        raise ValueError(f"Unknown inf_policy: {inf_policy}")

    prev = None
    for t in range(1, T):
        p = pick_top_bar_in_snapshot(series_json[t], d_caps[t], b_floors[t])
        if prev is None:
            prev = pick_top_bar_in_snapshot(series_json[t-1], d_caps[t-1], b_floors[t-1])
        if prev is not None and p is not None:
            step = linf(prev, p)
            cum_u[t] = cum_u[t-1] + step * w_uniform(prev)
            cum_s[t] = cum_s[t-1] + step * w_standard(prev)
        else:
            cum_u[t] = cum_u[t-1]
            cum_s[t] = cum_s[t-1]
        if p is not None:
            prev = p  # lock onto the most persistent vine path
    return cum_u, cum_s

# ------------- per-tag processing -------------

def process_tag(json_path: Path, out_dir: Path, inf_policy: str, fixed_cap: float, fixed_floor: float):
    cfg = json.loads(json_path.read_text())
    tag = json_path.stem
    trials = cfg.get("trials", [])
    tag_dir = out_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    for tr in trials:
        tid = tr["trial"]
        H0_series = tr.get("H0", [])
        H1_series = tr.get("H1", [])
        if not H0_series and not H1_series:
            continue

        if H0_series:
            cum_H0_u, cum_H0_s = cumulative_for_top_vine(H0_series, inf_policy, fixed_cap, fixed_floor)
        else:
            cum_H0_u = cum_H0_s = np.zeros(len(H1_series), dtype=float)

        if H1_series:
            cum_H1_u, cum_H1_s = cumulative_for_top_vine(H1_series, inf_policy, fixed_cap, fixed_floor)
        else:
            cum_H1_u = cum_H1_s = np.zeros(len(H0_series), dtype=float)

        T = max(len(H0_series), len(H1_series))
        with open(tag_dir / f"trial_{tid}_vineyard.csv", "w") as fh:
            fh.write("epoch,cum_unif_H0,cum_std_H0,cum_unif_H1,cum_std_H1\n")
            for i in range(T):
                u0 = cum_H0_u[i] if i < len(cum_H0_u) else 0.0
                s0 = cum_H0_s[i] if i < len(cum_H0_s) else 0.0
                u1 = cum_H1_u[i] if i < len(cum_H1_u) else 0.0
                s1 = cum_H1_s[i] if i < len(cum_H1_s) else 0.0
                fh.write(f"{i},{u0},{s0},{u1},{s1}\n")

    print(f"✓ Processed {tag} → {tag_dir}")

def main():
    ap = argparse.ArgumentParser(description="Persistent-vine cumulative VD with finite ∞ mapping (no matching).")
    ap.add_argument("--results_dir", type=Path, default=Path("vineyard_results"))
    ap.add_argument("--out_dir",     type=Path, default=Path("vineyard_metrics_persistent_vine"))
    ap.add_argument("--inf_policy",  choices=["fixed","tag_global","epoch_max"], default="fixed",
                    help="How to map ∞ deaths/births to finite values.")
    ap.add_argument("--fixed_cap",   type=float, default=1.0, help="Death cap when inf_policy=fixed or fallback.")
    ap.add_argument("--fixed_floor", type=float, default=0.0, help="Birth floor when inf_policy=fixed or fallback.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.results_dir.glob("*.json"))
    if not files:
        print(f"No JSON found in {args.results_dir}")
        return

    for jp in files:
        process_tag(jp, args.out_dir, args.inf_policy, args.fixed_cap, args.fixed_floor)

if __name__ == "__main__":
    main()