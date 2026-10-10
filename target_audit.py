"""Compare seed-centered and neutral-centered Bradley--Terry references."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from analysis_helpers import fit_bradley_terry_map



def run(output, cases_path, pairs_path, verdicts_path, reference_path):
    if output.exists():
        raise ValueError("Output exists; choose a new destination")
    cases = pd.read_csv(cases_path)
    pairs = pd.read_csv(pairs_path).merge(pd.read_csv(verdicts_path), on="pair_id", validate="one_to_one")
    n = len(cases)
    wins = np.zeros((n, n))
    seed = cases.bia_score.to_numpy(float)
    mu = (seed - seed.mean()) / (seed.std() + 1e-9)
    agreements = []
    for row in pairs.itertuples():
        assert row.winner in {row.i, row.j}
        loser = row.j if row.winner == row.i else row.i
        wins[row.winner, loser] += 1
        if seed[row.i] != seed[row.j]:
            agreements.append(seed[row.winner] > seed[loser])
    reference = pd.read_csv(reference_path).set_index("case_key").gt_rank.loc[cases.case_key].to_numpy()
    seeded = fit_bradley_terry_map(n, wins, mu)
    seeded_rank = (-seeded).argsort().argsort() + 1
    assert np.array_equal(seeded_rank, reference), "Refit does not reproduce the frozen reference"
    neutral = fit_bradley_terry_map(n, wins, np.zeros(n))
    neutral_rank = (-neutral).argsort().argsort() + 1
    components, _ = connected_components(wins, directed=True, connection="strong")
    report = {"n_cases": n, "n_pairs": len(pairs), "n_pairs_with_unequal_seed_scores": len(agreements),
              "llm_verdict_agreement_with_seed_preference": float(np.mean(agreements)),
              "directed_win_graph_strong_components": int(components),
              "finite_unregularized_mle_identifiable": components == 1,
              "neutral_centered_gaussian_prior": {"lambda": 1,
                  "spearman_with_seed_centered_reference": float(spearmanr(reference, neutral_rank).statistic),
                  "top10_overlap": len(set(np.flatnonzero(reference <= 10)) & set(np.flatnonzero(neutral_rank <= 10))) / 10},
              "reference_context": {"comparisons": "fixed seed-informed pair sampling and observed verdicts",
                                    "neutral_fit": "zero prior center with Gaussian regularization lambda=1",
                                    "finite_unregularized_mle": "requires a strongly connected directed win graph"}}
    output.mkdir(parents=True)
    (output / "target_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("cases", "pairs", "verdicts", "reference"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.cases, args.pairs, args.verdicts, args.reference)
