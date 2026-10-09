"""Descriptive sensitivity of heuristic alignment to non-negative weight ratios."""

import argparse
import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

RANDOM_STATE = 42
TIER_ORDINAL = {"CRITICAL": 5, "HIGH": 4, "MODERATE": 3, "LOW": 2, "MINIMAL": 1, "UNKNOWN": 0}

N_DRAWS = 20000
PAPER_W = np.array([3.0, 1.5, 1.0, 0.5])
NAMES = ["z(pro_bia)", "z(inferred_tier)", "z(history_fail)", "is_foundational"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new destination")
    df = pd.read_csv(args.features)
    gt = df["gt_rank"].values.astype(float)

    def z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 0 else a * 0.0

    X = np.column_stack([
        z(df["pro_bia_score"].values.astype(float)),
        z(df["bia_risk_label"].fillna("UNKNOWN").map(TIER_ORDINAL).values.astype(float)),
        z(df["history_fail_rate"].values.astype(float)),
        df["is_foundational"].values.astype(float),
    ])

    def rho(w):
        value = spearmanr(-gt, X @ w).statistic
        return float(np.nan_to_num(value))

    rng = np.random.default_rng(RANDOM_STATE)
    W = rng.dirichlet(np.ones(4), size=N_DRAWS)          # uniform over the simplex
    rhos = np.array([rho(w) for w in W])

    paper = rho(PAPER_W)
    equal = rho(np.ones(4))
    pct = float((rhos < paper).mean() * 100)
    best_i = int(np.argmax(rhos))

    print(f"n = {len(gt)}, {N_DRAWS} random weightings on the simplex\n")
    print(f"paper weights 3:1.5:1:0.5   rho = {paper:.4f}   ({pct:.1f}th percentile of random weightings)")
    print(f"equal weights 1:1:1:1       rho = {equal:.4f}")
    print("random weightings           rho = "
          f"{rhos.mean():.4f} mean, [{np.percentile(rhos, 5):.4f}, {np.percentile(rhos, 95):.4f}] 5-95%, "
          f"min {rhos.min():.4f}")
    print(f"best sampled weighting      rho = {rhos[best_i]:.4f}  at "
          + ":".join(f"{v:.2f}" for v in W[best_i] / W[best_i].max()))
    print("\nsingle signals alone:")
    for i, nm in enumerate(NAMES):
        print(f"  {nm:<18} rho = {rho(np.eye(4)[i]):.4f}")

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump({
            "n": int(len(gt)), "n_draws": N_DRAWS, "random_state": RANDOM_STATE,
            "signals": NAMES,
            "paper_weights": PAPER_W.tolist(), "paper_rho": paper, "paper_percentile": pct,
            "equal_rho": equal,
            "random_rho": {"mean": float(rhos.mean()), "std": float(rhos.std()),
                           "p5": float(np.percentile(rhos, 5)), "p50": float(np.percentile(rhos, 50)),
                           "p95": float(np.percentile(rhos, 95)), "min": float(rhos.min()),
                           "max": float(rhos.max())},
            "best_weights_normalized": (W[best_i] / W[best_i].max()).tolist(),
            "single_signal_rho": {nm: rho(np.eye(4)[i]) for i, nm in enumerate(NAMES)},
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
