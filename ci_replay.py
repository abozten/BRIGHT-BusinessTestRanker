"""Read-only retrospective failure replay; never treat failed tests as distinct faults."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

def normalized(value):
    return " ".join(str(value or "").lower().split())


def case_key(feature, scenario):
    return (normalized(feature), normalized(scenario))


def parse_date(value):
    if not value:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def failure_metrics(order):
    """Area metrics use one failure event per failed test, not defect identities."""
    n = len(order)
    failed = [i for i, row in enumerate(order, 1) if row["status"] == "failed"]
    if not failed:
        return {"failure_apfd": None, "failure_apfdc": None, "first_failure_seconds": None}
    apfd = 1 - sum(failed) / (n * len(failed)) + 1 / (2 * n)
    costs = [row["seconds"] for row in order]
    if any(not np.isfinite(c) or c <= 0 for c in costs):
        return {"failure_apfd": apfd, "failure_apfdc": None, "first_failure_seconds": None}
    elapsed = np.cumsum(costs)
    total = sum(costs)
    apfdc = sum(total - elapsed[i - 1] + costs[i - 1] / 2 for i in failed) / (total * len(failed))
    return {"failure_apfd": apfd, "failure_apfdc": apfdc,
            "first_failure_seconds": float(elapsed[failed[0] - 1])}


def collect_jobs(documents, keys):
    jobs = defaultdict(lambda: {"rows": {}, "dates": [], "retries": 0, "skipped": 0, "unmatched": 0})
    for doc in documents:
        if doc.get("job_id") is None:
            raise ValueError("CI document has no job_id")
        job = jobs[str(doc["job_id"])]
        for feature in doc.get("processed_data", []) or []:
            date = parse_date(feature.get("created_at"))
            if date:
                job["dates"].append(date)
            for scenario in feature.get("scenarios", []) or []:
                key = case_key(feature.get("feature_name"), scenario.get("scenario_name"))
                if key not in keys:
                    job["unmatched"] += 1
                    continue
                status = normalized(scenario.get("scenario_status"))
                if status not in {"passed", "failed"}:
                    job["skipped"] += 1
                    continue
                if key in job["rows"]:
                    job["retries"] += 1
                    continue
                job["rows"][key] = {"key": key, "status": status,
                                    "seconds": float(scenario.get("scenario_execution_time") or 0)}
    return jobs


def run(output, predictions, cases_path, pro_path, jobs_path):
    if output.exists():
        raise ValueError("Output exists; choose a new destination")
    df = pd.read_csv(cases_path)
    df = df.merge(pd.read_csv(pro_path)[["case_key", "pro_bia_score"]], on="case_key", validate="one_to_one")
    pred = pd.read_json(predictions)
    df = df.merge(pred[["case_key", "ridge_corrected"]], on="case_key", validate="one_to_one")
    assert len(df) == 70
    keys = [case_key(r.feature_name, r.test_scenario) for r in df.itertuples()]
    assert len(set(keys)) == 70
    pro = dict(zip(keys, df.pro_bia_score))
    ridge = dict(zip(keys, df.ridge_corrected))
    tiers = dict(zip(keys, df.bia_risk_label.map({"MINIMAL": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4})))
    found = {case_key(r.feature_name, r.test_scenario): int(any(word in normalized(r.feature_name + " " + r.test_scenario)
              for word in ("login", "auth", "server", "cli", "execute a test cycle", "access"))) for r in df.itertuples()}
    # Caller supplies records in first-stored-attempt order; no database connection.
    documents = json.loads(jobs_path.read_text())
    if not isinstance(documents, list):
        raise ValueError("Jobs input must be a JSON array of records")
    jobs = collect_jobs(documents, set(keys))
    history = defaultdict(list)
    rng = random.Random(20261007)
    summaries = []
    ordered_jobs = sorted(jobs.items(), key=lambda pair: (min(pair[1]["dates"]) if pair[1]["dates"] else datetime.max.replace(tzinfo=timezone.utc), pair[0]))
    for job_id, job in ordered_jobs:
        timestamp = min(job["dates"]) if job["dates"] else None
        # Rates use only strictly earlier timestamps inside 180 days, never current outcomes.
        rates = {}
        for key in keys:
            past = [status for date, status in history[key]
                    if timestamp and timestamp - timedelta(days=180) <= date < timestamp]
            rates[key] = sum(status == "failed" for status in past) / len(past) if past else 0.0
        rows = list(job["rows"].values())
        strategies = {"stored_order": rows,
                      "history_180d": sorted(rows, key=lambda r: (-rates[r["key"]], r["key"])),
                      "pointwise_llm_pro": sorted(rows, key=lambda r: (-pro[r["key"]], r["key"])),
                      "revised_ridge_oof": sorted(rows, key=lambda r: (-ridge[r["key"]], r["key"])),
                      "foundational_first": sorted(rows, key=lambda r: (-found[r["key"]], r["key"]))}
        signals = np.array([[pro[k], tiers[k], rates[k]] for k in keys])
        deviations = signals.std(axis=0)
        z = (signals - signals.mean(axis=0)) / np.where(deviations > 0, deviations, 1)
        for name, weights in (("equal_heuristic", [1, 1, 1, 1]), ("hand_heuristic", [3, 1.5, 1, 0.5])):
            scores = dict(zip(keys, z @ weights[:3] + weights[3] * np.array([found[k] for k in keys])))
            strategies[name] = sorted(rows, key=lambda r: (-scores[r["key"]], r["key"]))
        if len(rows) >= 2 and any(r["status"] == "failed" for r in rows):
            measures = {name: failure_metrics(order) for name, order in strategies.items()}
            draws = []
            for _ in range(1000):
                shuffled = rows.copy()
                rng.shuffle(shuffled)
                draws.append(failure_metrics(shuffled))
            measures["random"] = {name: float(np.mean([r[name] for r in draws])) if draws[0][name] is not None else None
                                  for name in draws[0]}
            summaries.append({"job_hash": hashlib.sha256(job_id.encode()).hexdigest()[:12],
                              "n_observed_cases": len(rows), "n_failed_cases": sum(r["status"] == "failed" for r in rows),
                              "timestamp_available": timestamp is not None, "measures": measures})
        if timestamp:
            for row in rows:
                history[row["key"]].append((timestamp, row["status"]))
    aggregate = {}
    if summaries:
        for strategy in summaries[0]["measures"]:
            aggregate[strategy] = {}
            for metric in summaries[0]["measures"][strategy]:
                values = [j["measures"][strategy][metric] for j in summaries if j["measures"][strategy][metric] is not None]
                aggregate[strategy][metric] = {"n_jobs": len(values), "mean": float(np.mean(values)) if values else None,
                                               "median": float(np.median(values)) if values else None}
    report = {"n_jobs_in_database": len(jobs), "n_jobs_with_matched_cases": sum(bool(j["rows"]) for j in jobs.values()),
              "n_eligible_failure_jobs": len(summaries), "n_jobs_with_all_70_executed": sum(len(j["rows"]) == 70 for j in jobs.values()),
              "executed_case_count_distribution": {str(n): sum(len(j["rows"]) == n for j in jobs.values()) for n in sorted({len(j["rows"]) for j in jobs.values()})},
              "duplicate_attempts_excluded": sum(j["retries"] for j in jobs.values()), "skipped_or_unknown_excluded": sum(j["skipped"] for j in jobs.values()),
              "unmatched_executions_excluded": sum(j["unmatched"] for j in jobs.values()),
              "source_prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
              "seed": 20261007, "random_replays_per_job": 1000, "aggregate": aggregate, "jobs": summaries,
              "limitations": ["Current database snapshot may differ from the paper's original 88-job extraction.",
                              "Uses first stored passed/failed attempt; stored document order is not verified execution order.",
                              "Failed cases are failure events, not identified distinct faults; area measures are APFD-style proxies.",
                              "Cost and time metrics assume serial, order-invariant scenario durations in seconds, not measured wall clock.",
                              "Test dependencies and order-induced outcomes are not modeled; partial jobs omit unobserved cases.",
                              "History uses feature.created_at as recorded; its meaning as execution time needs confirmation.",
                              "Static text features and group-OOF Ridge were generated after historical jobs: retrospective diagnostic, not temporal deployment validation.",
                              "No calibrated severity weights, actual cancellation savings, energy, or carbon estimates.",
                              "No-failure jobs have undefined area metrics and are excluded from failure-effectiveness means."]}
    output.mkdir(parents=True)
    (output / "replay.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n_jobs_in_database", "n_jobs_with_matched_cases", "n_eligible_failure_jobs", "n_jobs_with_all_70_executed", "aggregate")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    for name in ("cases", "pro", "jobs"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output, args.predictions, args.cases, args.pro, args.jobs)
    except Exception as error:
        parser.exit(2, f"CI replay failed ({type(error).__name__}); no raw connection details printed.\n")
