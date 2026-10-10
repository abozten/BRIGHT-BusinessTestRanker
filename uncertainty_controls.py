"""Paired model uncertainty, seed controls, and alternative-reference scoring.

Bootstrap intervals condition on the fitted OOF models and selected hyperparameters.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analysis_helpers import add_foundational_feature, calibrate_scores_by_train_ecdf, fit_bradley_terry_map
from expert_review import parse_answers
from evaluation import TIER, SCOPE, fit_predict, metrics


def correlations(target, scores):
    """Spearman for score columns; constant columns use the study's zero policy."""
    x = rankdata(target).astype(float)
    y = rankdata(np.asarray(scores), axis=0).astype(float)
    x -= x.mean()
    y -= y.mean(axis=0)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y, axis=0))
    return np.divide(x @ y, denominator, out=np.zeros(y.shape[1]), where=denominator > 0)


def interval(draws):
    return np.percentile(draws, [2.5, 97.5], axis=0).tolist()


def wilson(successes, total):
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def fold_average(target, scores, indices):
    values = [correlations(target[idx], scores[idx]) for idx in indices]
    return np.average(values, axis=0, weights=[len(idx) for idx in indices])


def feature_bootstrap(frame, target, scores, folds, n_boot, seed):
    rng = np.random.default_rng(seed)
    blocks = [[idx[frame.feature_name.iloc[idx].to_numpy() == name]
               for name in frame.feature_name.iloc[idx].unique()] for _, idx in folds]
    draws = np.empty((n_boot, scores.shape[1]))
    for b in range(n_boot):
        sampled = [np.concatenate([group[i] for i in rng.integers(len(group), size=len(group))])
                   for group in blocks]
        draws[b] = fold_average(target, scores, sampled)
    return draws


def pair_agreement(scores, pairs):
    correct = ties = 0
    for a, b, winner in pairs:
        difference = scores[a] - scores[b]
        ties += difference == 0
        correct += (difference > 0 and winner == a) or (difference < 0 and winner == b)
    return {"correct": int(correct), "ties": int(ties), "n": len(pairs),
            "agreement_half_credit_ties": float((correct + 0.5 * ties) / len(pairs))}


def run(args):
    if args.output.exists():
        raise ValueError("Output exists; choose a new destination")
    frozen = json.loads(args.metrics.read_text())
    df = pd.read_csv(args.cases).sort_values("gt_rank").reset_index(drop=True)
    seed_cases = pd.read_csv(args.seed_cases)
    assert seed_cases.case_key.is_unique and seed_cases.bia_score.notna().all()
    seed_values = seed_cases.set_index("case_key").bia_score
    df["seed_score"] = seed_values.loc[df.case_key].to_numpy(float)
    for path, column in [(args.flash, "llm_bia_score"), (args.pro, "pro_bia_score")]:
        df = df.merge(pd.read_csv(path)[["case_key", column]], on="case_key", validate="one_to_one")
    df = add_foundational_feature(df)
    df["feature_centrality"] = df.mapped_feature_name.map(df.mapped_feature_name.value_counts())
    df["bia_ordinal"] = df.bia_risk_label.map(TIER)
    df["impact_ordinal"] = df.impact_scope.map(SCOPE)
    groups = pd.factorize(df.feature_name)[0]
    folds = list(GroupKFold(5).split(df, groups=groups))
    saved = pd.read_json(args.oof).set_index("case_key")
    assert len(df) == 70 and df.gt_rank.nunique() == 70 and len(saved) == 70
    np.testing.assert_array_equal(saved.gt_rank.loc[df.case_key], df.gt_rank)
    names = ["seed_score", "history", "cached_retrieval_score", "bia_tier", "llm_flash", "llm_pro",
             "equal_heuristic", "hand_heuristic", "ridge",
             "ridge_without_impact_llms", "ridge_without_any_llm_derived_signal", "embedding_plus_ridge"]
    columns = ["seed_score", "history_fail_rate", "semantic_score", "bia_ordinal", "llm_bia_score", "pro_bia_score"]
    raw = np.zeros((len(df), len(names)))
    raw[:, :len(columns)] = df[columns].to_numpy(float)
    pooled = raw.copy()
    # Reconstruct raw fold predictions: saved training-ECDF scores can introduce ties.
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    snapshot = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    encoder = SentenceTransformer(snapshot, local_files_only=True, device="cpu")
    embeddings = encoder.encode((df.feature_name + " " + df.test_scenario).tolist(),
                                normalize_embeddings=True, show_progress_bar=False)
    embedding_columns = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    df = pd.concat([df, pd.DataFrame(embeddings, columns=embedding_columns)], axis=1)
    for fold, (tr, te) in enumerate(folds):
        train, test = df.iloc[tr], df.iloc[te]
        heuristic_columns = ["pro_bia_score", "bia_ordinal", "history_fail_rate"]
        scaler = StandardScaler().fit(train[heuristic_columns])
        for name, weights in [("equal_heuristic", [1, 1, 1, 1]), ("hand_heuristic", [3, 1.5, 1, .5])]:
            p = scaler.transform(test[heuristic_columns]) @ weights[:3] + weights[3] * test.is_foundational
            ptr = scaler.transform(train[heuristic_columns]) @ weights[:3] + weights[3] * train.is_foundational
            col = names.index(name)
            raw[te, col] = p
            pooled[te, col] = calibrate_scores_by_train_ecdf(ptr, p)
        for name in names[8:]:
            spec = frozen["variant_features"][name]
            p, ptr = fit_predict(train, test, spec["numeric"], spec["categorical"],
                                 frozen["selected_alphas"][name][fold])
            col = names.index(name)
            raw[te, col] = p
            pooled[te, col] = calibrate_scores_by_train_ecdf(ptr, p)
            np.testing.assert_allclose(pooled[te, col], saved[name].loc[test.case_key], atol=1e-9, rtol=0)
    target = -df.gt_rank.to_numpy(float)
    observed = fold_average(target, raw, [te for _, te in folds])
    for name, value in zip(names[1:], observed[1:]):
        np.testing.assert_allclose(value, frozen["fold_weighted"][name]["spearman"], atol=1e-10)
    if args.scenario_scores:
        for label in ["flash", "pro"]:
            control = pd.read_csv(args.scenario_scores / f"{label}_scores.csv").set_index("case_key")
            assert len(control) == len(df) and control.index.is_unique
            values = control[f"{label}_scenario_score"].loc[df.case_key].to_numpy(float)
            names.append(f"{label}_scenario_only")
            raw = np.column_stack([raw, values])
            pooled = np.column_stack([pooled, values])
        observed = fold_average(target, raw, [te for _, te in folds])
    draws = feature_bootstrap(df, target, raw, folds, args.bootstrap, args.seed)
    intervals = np.asarray(interval(draws))
    ridge = names.index("ridge")
    evaluation = {}
    for col, name in enumerate(names):
        delta = draws[:, ridge] - draws[:, col]
        evaluation[name] = {"rho": float(observed[col]), "rho_ci95": intervals[:, col].tolist(),
                         "ridge_minus_strategy": float(observed[ridge] - observed[col]),
                         "ridge_minus_strategy_ci95": interval(delta)}
        if name in frozen["fold_weighted"]:
            evaluation[name]["metrics"] = frozen["fold_weighted"][name]
        else:
            parts = [metrics(df.gt_rank.iloc[te], raw[te, col]) for _, te in folds]
            evaluation[name]["metrics"] = {key: float(np.average([part[key] for part in parts],
                                                            weights=[len(te) for _, te in folds]))
                                        for key in parts[0]}
    contrasts = {}
    comparisons = [("llm_pro", "history"), ("llm_pro", "bia_tier")]
    if args.scenario_scores:
        comparisons += [("llm_flash", "flash_scenario_only"), ("llm_pro", "pro_scenario_only")]
    for left, right in comparisons:
        a, b = names.index(left), names.index(right)
        contrasts[f"{left}_minus_{right}"] = {"delta_rho": float(observed[a] - observed[b]),
                                                "ci95": interval(draws[:, a] - draws[:, b])}
    # Alternative reference changes evaluation only; it does not refit the rankers.
    pairs = pd.read_csv(args.pairs).merge(pd.read_csv(args.verdicts), on="pair_id", validate="one_to_one")
    wins = np.zeros((len(seed_cases), len(seed_cases)))
    for row in pairs.itertuples():
        assert row.winner in {row.i, row.j}
        wins[row.winner, row.j if row.winner == row.i else row.i] += 1
    seeds = seed_cases.bia_score.to_numpy(float)
    strength = fit_bradley_terry_map(len(seeds), wins, (seeds - seeds.mean()) / (seeds.std() + 1e-9))
    np.testing.assert_array_equal((-strength).argsort().argsort() + 1,
                                 df.set_index("case_key").gt_rank.loc[seed_cases.case_key])
    neutral_strength = fit_bradley_terry_map(len(seeds), wins, np.zeros(len(seeds)))
    neutral_rank = pd.Series((-neutral_strength).argsort().argsort() + 1, index=seed_cases.case_key)
    neutral_target = -neutral_rank.loc[df.case_key].to_numpy(float)
    neutral_observed = fold_average(neutral_target, raw, [te for _, te in folds])
    neutral_draws = feature_bootstrap(df, neutral_target, raw, folds, args.bootstrap, args.seed)
    neutral_intervals = np.asarray(interval(neutral_draws))
    neutral = {name: {"rho": float(neutral_observed[col]), "rho_ci95": neutral_intervals[:, col].tolist(),
                      "metrics": {k: float(np.average([metrics(-neutral_target[te], raw[te, col])[k]
                                                     for _, te in folds], weights=[len(te) for _, te in folds]))
                                  for k in ["kendall", "ndcg10", "overlap10"]}}
               for col, name in enumerate(names)}
    manifest = json.loads(args.expert_key.read_text())
    answers = parse_answers(args.expert_review)
    records = []
    mapping = {key: i for i, key in enumerate(df.case_key)}
    llm_agreement = 0
    for number, row in enumerate(manifest["records"], 1):
        if row["kind"] != "primary":
            continue
        answer = answers.get(number)
        assert answer in {"A", "B"}, "Primary expert review is incomplete or non-decisive"
        winner = row[answer.lower()]
        llm_agreement += winner == row["llm_winner"]
        a, b, winner = [mapping[seed_cases.case_key.iloc[index]] for index in [row["a"], row["b"], winner]]
        records.append((a, b, winner))
    expert = {name: pair_agreement(pooled[:, col], records) for col, name in enumerate(names)}
    expert["seed_centered_reference"] = pair_agreement(target, records)
    expert["neutral_prior_reference"] = pair_agreement(neutral_target, records)
    expert["original_llm_verdict"] = {"correct": llm_agreement, "n": len(records),
                                      "agreement": llm_agreement / len(records),
                                      "wilson_ci95": wilson(llm_agreement, len(records))}
    expert["seed_score"]["wilson_ci95"] = wilson(expert["seed_score"]["correct"], len(records))
    replay = json.loads(args.replay.read_text())
    strategies = list(replay["aggregate"])
    measure_names = ["failure_apfd", "failure_apfdc", "first_failure_seconds"]
    values = np.array([[[job["measures"][strategy][measure] for measure in measure_names]
                        for strategy in strategies] for job in replay["jobs"]], dtype=float)
    assert np.isfinite(values).all() and len(values) == replay["n_eligible_failure_jobs"]
    rng = np.random.default_rng(args.seed + 1)
    job_draws = values[rng.integers(len(values), size=(args.bootstrap, len(values)))].mean(axis=1)
    job_ci = np.asarray(interval(job_draws))
    replay_ridge = strategies.index("ridge_oof")
    job_report = {}
    for col, strategy in enumerate(strategies):
        job_report[strategy] = {}
        for m, measure in enumerate(measure_names):
            mean = float(values[:, col, m].mean())
            np.testing.assert_allclose(mean, replay["aggregate"][strategy][measure]["mean"], atol=1e-10)
            job_report[strategy][measure] = {"mean": mean, "ci95": job_ci[:, col, m].tolist(),
                "strategy_minus_ridge": float(mean - values[:, replay_ridge, m].mean()),
                "strategy_minus_ridge_ci95": interval(job_draws[:, col, m] - job_draws[:, replay_ridge, m])}
    inputs = [args.cases, args.seed_cases, args.flash, args.pro, args.metrics, args.oof,
              args.pairs, args.verdicts, args.expert_key, args.expert_review, args.replay]
    if args.scenario_scores:
        inputs += [args.scenario_scores / f"{label}_scores.csv" for label in ["flash", "pro"]]
    report = {"seed": args.seed, "bootstrap_draws": args.bootstrap,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "protocol": {"feature_bootstrap": "paired feature blocks within each fixed outer fold; size-weighted rho",
                           "model_uncertainty": "conditional on frozen fitted models and selected hyperparameters; no retuning",
                           "job_bootstrap": "paired whole eligible jobs; fixed saved random means",
                           "alternative_reference": "fixed seed-trained predictions re-scored; no alternative-reference training",
                           "expert_prediction": "training-ECDF pooled OOF scores for learned models and fold-scaled heuristics; ties receive half credit",
                           "wilson": "nominal binomial interval; shared cases violate independence"},
              "seed_whole_suite_rho": float(spearmanr(df.seed_score, target).statistic),
              "evaluation": evaluation, "paired_contrasts": contrasts,
              "neutral_prior": neutral, "expert": expert, "replay": job_report}
    args.output.mkdir(parents=True)
    (args.output / "controls.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(raw, columns=names).assign(case_key=df.case_key).to_csv(args.output / "raw_fold_scores.csv", index=False)
    print(json.dumps({"seed_rho": report["seed_whole_suite_rho"], "evaluation": evaluation,
                      "neutral_rho": {k: v["rho"] for k, v in neutral.items()}, "expert": expert}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["cases", "seed-cases", "flash", "pro", "metrics", "oof", "pairs", "verdicts",
                 "expert-key", "expert-review", "replay", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scenario-scores", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("bootstrap must be at least 2")
    run(args)
