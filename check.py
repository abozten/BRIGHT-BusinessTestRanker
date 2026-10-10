"""Small numerical and contract checks; no industrial records or fixtures."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_helpers import calibrate_scores_by_train_ecdf, fit_bradley_terry_map
from evaluation import design, metrics, run
from expert_review import summarize
from ci_replay import failure_metrics, collect_jobs
from uncertainty_controls import correlations, feature_bootstrap, pair_agreement, wilson
from scenario_only_control import parse_score


def main():
    assert parse_score([{"text": '{"bia_score": 75}'}]) == 75
    for invalid in ['{"bia_score": true}', '{"bia_score": 101}', '{"bia_score": 1.5}']:
        try:
            parse_score(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid provider score was accepted")
    np.testing.assert_allclose(correlations([1, 2, 3], [[3, 7], [2, 7], [1, 7]]), [-1, 0])
    assert pair_agreement(np.array([1., 1., 3.]), [(0, 1, 0), (0, 2, 2)]) == {
        "correct": 1, "ties": 1, "n": 2, "agreement_half_credit_ties": .75}
    np.testing.assert_allclose(wilson(26, 40), [.49505881, .77865471], atol=1e-6)
    frame = pd.DataFrame({"feature_name": ["a", "a", "b", "c", "c", "d"]})
    scores = np.column_stack([np.arange(6), np.arange(6)])
    draws = feature_bootstrap(frame, np.arange(6), scores,
                              [(None, np.arange(3)), (None, np.arange(3, 6))], 20, 42)
    np.testing.assert_array_equal(draws[:, 0] - draws[:, 1], 0)
    train = pd.DataFrame({"llm_bia_score": [10, 20, 40], "pro_bia_score": [12, 18, 50],
                          "domain": ["A", "A", "B"]})
    heldout = pd.DataFrame({"llm_bia_score": [999], "pro_bia_score": [-999], "domain": ["UNSEEN"]})
    cols = ["llm_bia_score", "pro_bia_score", "disagreement"]
    xtrain, xtest = design(train, heldout, cols, ["domain"])
    changed, _ = design(train, heldout.assign(llm_bia_score=0), cols, ["domain"])
    np.testing.assert_allclose(xtrain, changed)
    np.testing.assert_allclose(xtest[0, -2:], 0)
    assert metrics([1, 4, 70], [3, 2, 1])["ndcg10"] == 1
    tied = metrics(np.arange(1, 71), np.ones(70), repeats=1000)
    assert tied["spearman"] == 0 and .12 < tied["overlap10"] < .17
    np.testing.assert_allclose(calibrate_scores_by_train_ecdf([1, 1, 3], [1, 2, 3]), [1/3, 2/3, 5/6])
    strengths = fit_bradley_terry_map(2, np.array([[0., 2.], [0., 0.]]), np.zeros(2))
    assert strengths[0] > strengths[1] and abs(strengths.mean()) < 1e-12
    report = summarize({"records": [
        dict(a=0, b=1, llm_winner=0, kind="primary"),
        dict(a=2, b=3, llm_winner=3, kind="primary"),
        dict(a=1, b=0, llm_winner=0, kind="swap", original=1),
    ]}, {1: "A", 2: "B", 3: "B"})
    assert report["agreement_with_llm"] == report["cohens_kappa_binary"] == 1
    assert report["consistency_checks"]["swap"]["consistency"] == 1
    assert failure_metrics([dict(status="failed", seconds=2), dict(status="passed", seconds=8)]) == {
        "failure_apfd": .75, "failure_apfdc": .9, "first_failure_seconds": 2}
    assert failure_metrics([dict(status="passed", seconds=8)])["failure_apfd"] is None
    try:
        collect_jobs([{}], set())
    except ValueError:
        pass
    else:
        raise AssertionError("Missing job identifier was accepted")
    with tempfile.TemporaryDirectory() as folder:
        try:
            run(Path(folder), None, None, None)
        except ValueError as error:
            assert "Output exists" in str(error)
        else:
            raise AssertionError("Existing output was accepted")
    print("Numerical and contract checks passed")


if __name__ == "__main__":
    main()
