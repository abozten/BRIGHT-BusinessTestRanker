"""Nested feature-group evaluation of Ridge, ablations, and matched baselines."""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_helpers import add_foundational_feature, calibrate_scores_by_train_ecdf

ALPHAS = [0.1, 1, 10, 50, 100, 500, 1000]
TIER = {"MINIMAL": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4, "UNKNOWN": 2}
SCOPE = {"LOCAL": 0, "MODULE": 1, "GLOBAL": 2, "UNKNOWN": 1}
warnings.filterwarnings("ignore", category=ConstantInputWarning)


def metrics(ranks, scores, rng=None, repeats=100):
    ranks, scores = np.asarray(ranks), np.asarray(scores)
    n, k = len(ranks), min(10, len(ranks))
    # Use the suite-wide relevance scale in folds and pooled results.
    rel = 71 - ranks
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sum(np.sort(rel)[::-1][:k] * discount)
    truth = set(np.argsort(ranks)[:k])
    ndcg, overlap = [], []
    rng = rng if rng is not None else np.random.default_rng(42)
    for _ in range(repeats if len(np.unique(scores)) < n else 1):
        order = np.lexsort((rng.random(n), -scores))[:k]
        ndcg.append(float(np.sum(rel[order] * discount) / ideal))
        overlap.append(len(truth & set(order)) / k)
    rho = spearmanr(-ranks, scores).statistic if np.unique(scores).size > 1 else 0.0
    tau = kendalltau(-ranks, scores).statistic if np.unique(scores).size > 1 else 0.0
    return dict(spearman=float(np.nan_to_num(rho)), kendall=float(np.nan_to_num(tau)),
                ndcg10=float(np.mean(ndcg)), overlap10=float(np.mean(overlap)))


def design(train, test, numeric, categorical):
    columns = list(numeric)
    train, test = train.copy(), test.copy()
    if "disagreement" in columns:
        scale = StandardScaler().fit(train[["llm_bia_score", "pro_bia_score"]])
        for frame in (train, test):
            z = scale.transform(frame[["llm_bia_score", "pro_bia_score"]])
            frame["disagreement"] = np.abs(z[:, 0] - z[:, 1])
    transforms = [("numeric", StandardScaler(), columns)]
    if categorical:
        transforms.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    transformer = ColumnTransformer(transforms)
    return transformer.fit_transform(train), transformer.transform(test)


def fit_predict(train, test, numeric, categorical, alpha):
    xtrain, xtest = design(train, test, numeric, categorical)
    ranks = train.gt_rank.to_numpy(float)
    weight = 1 + 2 * (ranks.max() - ranks + 1) / ranks.max()
    model = Ridge(alpha=alpha).fit(xtrain, -ranks, sample_weight=weight)
    return model.predict(xtest), model.predict(xtrain)


def run(output, cases, flash, pro):
    if output.exists():
        raise ValueError("Output exists; choose a new destination")
    paths = [cases, flash, pro]
    df = pd.read_csv(paths[0]).sort_values("gt_rank").reset_index(drop=True)
    for path, column in zip(paths[1:], ("llm_bia_score", "pro_bia_score")):
        df = df.merge(pd.read_csv(path)[["case_key", column]], on="case_key", validate="one_to_one")
    assert len(df) == 70 and df.gt_rank.nunique() == 70
    df = add_foundational_feature(df)
    df["feature_centrality"] = df.mapped_feature_name.map(df.mapped_feature_name.value_counts())
    df["bia_ordinal"] = df.bia_risk_label.map(TIER)
    df["impact_ordinal"] = df.impact_scope.map(SCOPE)
    groups = pd.factorize(df.feature_name)[0]
    df["_group"] = groups
    folds = list(GroupKFold(5).split(df, groups=groups))
    rng = np.random.default_rng(20261007)
    signals = {"history": "history_fail_rate", "cached_retrieval_score": "semantic_score",
               "foundational_text_only": "is_foundational", "centrality": "feature_centrality",
               "scope": "impact_ordinal", "bia_tier": "bia_ordinal",
               "llm_flash": "llm_bia_score", "llm_pro": "pro_bia_score"}
    names = list(signals) + ["domain_train_mean", "equal_heuristic", "hand_heuristic", "random"]
    results = {name: [] for name in names}
    sizes = [len(te) for _, te in folds]
    full = ["history_fail_rate", "llm_bia_score", "pro_bia_score", "disagreement", "feature_centrality", "bia_ordinal", "impact_ordinal"]
    variants = {
        "ridge": (full, ["domain"]),
        "ridge_without_impact_llms": ([x for x in full if x not in {"llm_bia_score", "pro_bia_score", "disagreement"}], ["domain"]),
        "ridge_without_any_llm_derived_signal": (["history_fail_rate", "feature_centrality", "is_foundational"], []),
        "ridge_without_taxonomy": ([x for x in full if x != "impact_ordinal"], []),
        "ridge_without_retrieval": ([x for x in full if x != "bia_ordinal"], ["domain"]),
        "ridge_without_history": ([x for x in full if x != "history_fail_rate"], ["domain"]),
        "ridge_without_structural": ([x for x in full if x != "feature_centrality"], ["domain"]),
    }
    # This baseline is available only from an already-cached model; no downloads or APIs.
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    snapshot = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    encoder = SentenceTransformer(snapshot, local_files_only=True, device="cpu")
    text = (df.feature_name + " " + df.test_scenario).tolist()
    embedded = encoder.encode(text, normalize_embeddings=True, show_progress_bar=False)
    embedding_columns = [f"embedding_{i}" for i in range(embedded.shape[1])]
    df = pd.concat([df, pd.DataFrame(embedded, columns=embedding_columns)], axis=1)
    variants["embedding_plus_ridge"] = (embedding_columns, [])
    oof = {name: np.zeros(len(df)) for name in variants}
    selected = {name: [] for name in variants}
    for name in variants:
        results[name] = []
    for train_idx, test_idx in folds:
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        ranks = test.gt_rank.to_numpy()
        for name, column in signals.items():
            results[name].append(metrics(ranks, test[column], rng))
        means = train.groupby("domain").gt_rank.mean()
        results["domain_train_mean"].append(metrics(ranks, -test.domain.map(means).fillna(train.gt_rank.mean()), rng))
        columns = ["pro_bia_score", "bia_ordinal", "history_fail_rate"]
        z = StandardScaler().fit(train[columns]).transform(test[columns])
        for name, weights in (("equal_heuristic", [1, 1, 1, 1]), ("hand_heuristic", [3, 1.5, 1, 0.5])):
            scores = z @ np.array(weights[:3]) + weights[3] * test.is_foundational.to_numpy()
            results[name].append(metrics(ranks, scores, rng))
        # Random score vectors are continuous, so there are no ties.
        draws = [metrics(ranks, rng.random(len(test)), rng) for _ in range(1000)]
        results["random"].append({key: float(np.mean([d[key] for d in draws])) for key in draws[0]})
        for name, (numeric, categorical) in variants.items():
            inner = list(GroupKFold(3).split(train, groups=train._group))
            candidate = []
            for alpha in ALPHAS:
                values = []
                for itr, ite in inner:
                    p, _ = fit_predict(train.iloc[itr], train.iloc[ite], numeric, categorical, alpha)
                    values.append((len(ite), metrics(train.iloc[ite].gt_rank, p, rng)["spearman"]))
                candidate.append(sum(n * v for n, v in values) / sum(n for n, _ in values))
            alpha = ALPHAS[int(np.argmax(candidate))]
            selected[name].append(alpha)
            p, ptr = fit_predict(train, test, numeric, categorical, alpha)
            results[name].append(metrics(ranks, p, rng))
            oof[name][test_idx] = calibrate_scores_by_train_ecdf(ptr, p)
    weighted = {name: {key: float(np.average([r[key] for r in values], weights=sizes))
                       for key in values[0]} for name, values in results.items()}
    pooled = {name: metrics(df.gt_rank, values, rng, repeats=1000) for name, values in oof.items()}
    random_pooled = [metrics(df.gt_rank, rng.random(len(df)), rng) for _ in range(1000)]
    pooled["random"] = {key: float(np.mean([r[key] for r in random_pooled])) for key in random_pooled[0]}
    report = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "seed": 20261007, "n_cases": len(df), "n_groups": len(set(groups)), "fold_sizes": sizes,
              "alpha_grid": ALPHAS, "selected_alphas": selected,
              "variant_features": {name: {"numeric": numeric, "categorical": categorical} for name, (numeric, categorical) in variants.items()},
              "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_snapshot_revision": Path(snapshot).name,
              "embedding_text": "feature_name + test_scenario",
              "group_size_distribution": {str(k): int(v) for k, v in df.groupby("feature_name").size().value_counts().sort_index().items()},
              "fold_weighted": weighted, "calibrated_pooled_oof": pooled,
              "protocol": "5 outer / 3 inner GroupKFold; training-partition scaling and disagreement; one-hot domain; training-ECDF pooled calibration",
              "evaluation_context": {"target": "seed-centered pairwise consequence reference",
                                     "history": "cached failure rates",
                                     "documentation": "retrieval/NLI impact score and derived tier",
                                     "foundational_input": "feature and scenario text",
                                     "no_llm_inputs": "failure history, structural centrality, foundational indicator"}}
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    df[["case_key", "gt_rank"]].assign(**oof).to_json(output / "oof_predictions.json", orient="records", indent=2)
    labels = {"cached_retrieval_score": "Cached retrieval/NLI score", "foundational_text_only": "Foundational rule (text only)",
              "domain_train_mean": "Domain (training mean)", "equal_heuristic": "Equal-weight heuristic (text-only rule)",
              "hand_heuristic": "Hand-weighted heuristic (text-only rule)", "ridge": "Ridge",
              "ridge_without_impact_llms": "Ridge without pointwise LLM estimates", "ridge_without_any_llm_derived_signal": "Ridge without any LLM-derived signal",
              "ridge_without_taxonomy": "Ridge without taxonomy", "ridge_without_retrieval": "Ridge without retrieval/NLI",
              "ridge_without_history": "Ridge without history", "ridge_without_structural": "Ridge without structural counts", "embedding_plus_ridge": "MiniLM embeddings + Ridge (raw text)"}
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize", r"\caption{Ridge, ablations, and baselines on the consequence reference. Results are size-weighted averages of the same five outer feature-group folds. Ridge uses three inner folds for regularization selection; transforms are fitted on the training partition.}",
             r"\label{tab:matched-baselines}", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrr}\toprule",
             r"Strategy & Spearman & Kendall & NDCG@10 & Top-10 overlap \\\midrule"]
    for name, row in weighted.items():
        label = labels.get(name, name.replace("_", " ").capitalize())
        lines.append(f"{label} & {row['spearman']:.3f} & {row['kendall']:.3f} & {row['ndcg10']:.3f} & {row['overlap10']:.3f} " + r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{adjustbox}", r"\end{table}"])
    (output / "matched_baselines.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "fold_weighted": weighted, "pooled": pooled}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--flash", type=Path, required=True)
    parser.add_argument("--pro", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.cases, args.flash, args.pro)
