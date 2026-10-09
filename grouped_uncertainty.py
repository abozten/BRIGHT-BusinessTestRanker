#!/usr/bin/env python3
"""Grouped model-comparison and Bradley--Terry uncertainty analyses."""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.model_selection import GroupKFold

from analysis_helpers import fit_bradley_terry_map

RNG_SEED = 20260925
N_BOOT = 10_000
N_BT_BOOT = 2_000
TIER = {"CRITICAL": 5, "HIGH": 4, "MODERATE": 3, "LOW": 2, "MINIMAL": 1, "UNKNOWN": 0}
warnings.filterwarnings("ignore", category=ConstantInputWarning)


def rho(target, score):
    value = spearmanr(target, score).statistic
    return 0.0 if np.isnan(value) else float(value)


def ci(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def z(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def fold_weighted_rho(frame, score, sampled_indices=None):
    total = weighted = 0.0
    for fold, part in frame.groupby("fold"):
        if sampled_indices is None:
            target, values = -part.gt_rank.to_numpy(), part[score].to_numpy()
        else:
            idx = sampled_indices[fold]
            target, values = -frame.gt_rank.to_numpy()[idx], frame[score].to_numpy()[idx]
        weight = len(target)
        weighted += weight * rho(target, values)
        total += weight
    return weighted / total


def grouped_model_inference(rng, predictions):
    pred = pd.read_csv(predictions)
    pred = pred.sort_values("gt_rank").reset_index(drop=True)
    groups = pd.factorize(pred.feature_name)[0]
    pred["fold"] = -1
    for fold, (_, test) in enumerate(GroupKFold(5).split(pred, pred.gt_rank, groups)):
        pred.loc[test, "fold"] = fold

    tier = pred.bia_risk_label.fillna("UNKNOWN").map(TIER).to_numpy(float)
    found = pred.is_foundational.to_numpy(float)
    pred["tier"] = tier
    pred["equal"] = z(pred.pro_bia_score) + z(tier) + z(pred.history_fail_rate) + found
    pred["hand"] = 3 * z(pred.pro_bia_score) + 1.5 * z(tier) + z(pred.history_fail_rate) + 0.5 * found
    group_by_fold = {fold: [part.index[part.feature_name == name].to_numpy()
                            for name in part.feature_name.unique()]
                     for fold, part in pred.groupby("fold")}

    comparisons = {}
    for baseline in ("equal", "hand"):
        observed = fold_weighted_rho(pred, "oof_pred_score") - fold_weighted_rho(pred, baseline)
        draws = np.empty(N_BOOT)
        for b in range(N_BOOT):
            sampled = {fold: np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                       for fold, blocks in group_by_fold.items()}
            draws[b] = (fold_weighted_rho(pred, "oof_pred_score", sampled)
                        - fold_weighted_rho(pred, baseline, sampled))
        comparisons[baseline] = {"delta_rho": observed, "ci95": ci(draws)}

    signal_differences = {}
    signal_intervals = {}
    all_groups = [pred.index[pred.feature_name == name].to_numpy()
                  for name in pred.feature_name.unique()]
    signal_draws = {score: np.empty(N_BOOT) for score in
                    ("llm_bia_score", "pro_bia_score", "history_fail_rate", "tier", "equal", "hand")}
    for b in range(N_BOOT):
        idx = np.concatenate([all_groups[i] for i in rng.integers(0, len(all_groups), len(all_groups))])
        sample = pred.iloc[idx]
        for score in signal_draws:
            signal_draws[score][b] = rho(-sample.gt_rank, sample[score])
    for score, draws in signal_draws.items():
        signal_intervals[score] = {"rho": rho(-pred.gt_rank, pred[score]), "ci95": ci(draws)}
    for other in ("history_fail_rate", "tier"):
        observed = rho(-pred.gt_rank, pred.pro_bia_score) - rho(-pred.gt_rank, pred[other])
        draws = signal_draws["pro_bia_score"] - signal_draws[other]
        signal_differences[other] = {"delta_rho": observed, "ci95": ci(draws)}

    return {"resampling_unit": "feature group", "n_groups": int(pred.feature_name.nunique()),
            "n_bootstrap": N_BOOT, "ridge_minus_heuristic": comparisons,
            "single_signal_rho": signal_intervals,
            "pro_llm_minus_signal": signal_differences}


def plot_bt_uncertainty(item, output):
    def tex(value):
        return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")

    top = item.sort_values("rank").head(15)
    lines = [r"\begin{tikzpicture}[x=1mm,y=1mm,font=\scriptsize]",
             r"\draw[gray!45] (47,3) -- (47,-73);",
             r"\draw[red!65,dashed] (60.3,3) -- (60.3,-73);",
             r"\node[anchor=south] at (69,3) {rank (95\% pair-bootstrap interval)};",
             r"\node[anchor=south] at (124,3) {degree};"]
    for row, case in enumerate(top.itertuples()):
        y = -5 * row
        lo, rank, hi = 46 + 1.4 * case.rank_lo, 46 + 1.4 * case.rank, 46 + 1.4 * case.rank_hi
        label = tex(f"{case.rank}. {case.mapped_feature_name[:38]}")
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
    for name in ("predictions", "cases", "pairs", "verdicts", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output exists; choose a new destination")
    args.output.mkdir(parents=True)
    report = {"random_seeds": {"model_inference": RNG_SEED, "bradley_terry": RNG_SEED + 1},
              "model_inference": grouped_model_inference(np.random.default_rng(RNG_SEED), args.predictions),
              "bradley_terry_uncertainty": bt_uncertainty(np.random.default_rng(RNG_SEED + 1),
                  args.cases, args.pairs, args.verdicts, args.reference, args.output)}
    (args.output / "grouped_uncertainty.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
