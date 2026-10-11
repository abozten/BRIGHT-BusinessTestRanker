#!/usr/bin/env python3
"""Pair-bootstrap uncertainty of the Bradley--Terry consequence reference."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from analysis_helpers import fit_bradley_terry_map

RNG_SEED = 20260926
N_BT_BOOT = 2_000
warnings.filterwarnings("ignore", category=ConstantInputWarning)


def rho(target, score):
    value = spearmanr(target, score).statistic
    return 0.0 if np.isnan(value) else float(value)


def ci(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]



def plot_bt_uncertainty(item, output):
    top = item.sort_values("rank").head(15)
    lines = [r"\begin{tikzpicture}[x=1mm,y=1mm,font=\scriptsize]",
             r"\draw[gray!45] (47,3) -- (47,-73);",
             r"\draw[red!65,dashed] (60.3,3) -- (60.3,-73);",
             r"\node[anchor=south] at (69,3) {rank (95\% pair-bootstrap interval)};",
             r"\node[anchor=south] at (124,3) {degree};"]
    for row, case in enumerate(top.itertuples()):
        y = -5 * row
        lo, rank, hi = 46 + 1.4 * case.rank_lo, 46 + 1.4 * case.rank, 46 + 1.4 * case.rank_hi
        # case text renamed
        label = f"Scenario {case.rank:02d}"
        lines.extend([
            rf"\node[anchor=east] at (44,{y}) {{{label}}};",
            rf"\draw[gray!70] ({lo:.1f},{y}) -- ({hi:.1f},{y});",
            rf"\draw[gray!70] ({lo:.1f},{y-1}) -- ({lo:.1f},{y+1});",
            rf"\draw[gray!70] ({hi:.1f},{y-1}) -- ({hi:.1f},{y+1});",
            rf"\fill[blue!65!black] ({rank:.1f},{y}) circle (1.1pt);",
            rf"\fill[blue!45] (112,{y-1.4}) rectangle ({112 + 4 * case.degree},{y+1.4});",
            rf"\node[anchor=west] at ({113 + 4 * case.degree},{y}) {{{case.degree}}};",
        ])
    lines.extend([r"\foreach \r in {1,10,20,30} {",
                  r"  \pgfmathsetmacro{\x}{46+1.4*\r}",
                  r"  \draw (\x,-74) -- (\x,-72.5) node[below] {\r};",
                  r"}", r"\end{tikzpicture}"])
    (output / "bt_target_uncertainty.tex").write_text("\n".join(lines) + "\n")


def bt_uncertainty(rng, cases_path, pairs_path, verdicts_path, reference_path, output):
    cases = pd.read_csv(cases_path)
    pairs = pd.read_csv(pairs_path).merge(pd.read_csv(verdicts_path), on="pair_id")
    n = len(cases)
    seed = cases.bia_score.fillna(cases.bia_score.median()).to_numpy(float)
    prior = (seed - seed.mean()) / (seed.std() + 1e-9)

    def fit(rows):
        wins = np.zeros((n, n))
        for row in rows.itertuples():
            loser = row.j if row.winner == row.i else row.i
            wins[row.winner, loser] += 1
        strength = fit_bradley_terry_map(n, wins, prior, lam=1.0)
        return (-strength).argsort().argsort() + 1

    original = fit(pairs)
    saved = cases[["case_key"]].merge(
        pd.read_csv(reference_path)[["case_key", "gt_rank"]], on="case_key"
    ).gt_rank.to_numpy()
    assert np.array_equal(original, saved), "Bradley--Terry refit does not reproduce the saved target"
    ranks = np.empty((N_BT_BOOT, n), dtype=np.int16)
    correlations = np.empty(N_BT_BOOT)
    top_overlap = np.empty(N_BT_BOOT)
    top = set(np.flatnonzero(original <= 10))
    for b in range(N_BT_BOOT):
        sample = pairs.iloc[rng.integers(0, len(pairs), len(pairs))]
        ranks[b] = fit(sample)
        correlations[b] = rho(original, ranks[b])
        top_overlap[b] = len(top & set(np.flatnonzero(ranks[b] <= 10))) / 10

    degree = np.zeros(n, dtype=int)
    for row in pairs.itertuples():
        degree[row.i] += 1
        degree[row.j] += 1
    item = cases[["case_key", "mapped_feature_name", "test_scenario"]].copy()
    item["rank"] = original
    item["degree"] = degree
    item["rank_lo"] = np.floor(np.percentile(ranks, 2.5, axis=0)).astype(int)
    item["rank_hi"] = np.ceil(np.percentile(ranks, 97.5, axis=0)).astype(int)
    item["top10_probability"] = (ranks <= 10).mean(axis=0)
    item.sort_values("rank").to_csv(output / "bt_rank_uncertainty.csv", index=False)
    plot_bt_uncertainty(item, output)
    return {"n_comparisons": len(pairs), "possible_pairs": n * (n - 1) // 2,
            "coverage_fraction": len(pairs) / (n * (n - 1) / 2),
            "mean_degree": float(degree.mean()), "n_bootstrap": N_BT_BOOT,
            "spearman_to_original": {"median": float(np.median(correlations)), "ci95": ci(correlations)},
            "top10_overlap_fraction": {"median": float(np.median(top_overlap)), "ci95": ci(top_overlap)},
            "top10_membership_probability_mean": float(item.loc[item["rank"] <= 10, "top10_probability"].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "pairs", "verdicts", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new destination")
    args.output.mkdir(parents=True)
    report = {"random_seed": RNG_SEED,
              "bradley_terry_uncertainty": bt_uncertainty(np.random.default_rng(RNG_SEED),
                  args.cases, args.pairs, args.verdicts, args.reference, args.output)}
    (args.output / "target_uncertainty.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
