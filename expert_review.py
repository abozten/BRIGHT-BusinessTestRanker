"""Prepare and score a blind expert audit without replacing the frozen target."""

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(output, cases_path, pairs_path, verdicts_path, count=40, seed=20261007):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new destination to preserve reviews")
    paths = [cases_path, pairs_path, verdicts_path]
    cases, pairs, verdicts = map(read_csv, paths)
    winners = {int(row["pair_id"]): int(row["winner"]) for row in verdicts}
    if not 16 <= count <= len(pairs):
        raise ValueError("Choose at least 16 pairs and no more than the supplied pair count")
    rng = random.Random(seed)
    selected = rng.sample(pairs, count)
    records = []
    for pair in selected:
        a, b = int(pair["i"]), int(pair["j"])
        winner = winners[int(pair["pair_id"])]
        if winner not in (a, b):
            raise ValueError("LLM winner is not a member of its pair")
        if rng.randrange(2):
            a, b = b, a
        records.append(dict(pair_id=int(pair["pair_id"]), a=a, b=b,
                            llm_winner=winner, kind="primary"))
    checks = rng.sample(records, 16)
    for index, original in enumerate(checks):
        record = {**original, "kind": "swap" if index < 8 else "repeat",
                  "original": records.index(original) + 1}
        if record["kind"] == "swap":
            record["a"], record["b"] = record["b"], record["a"]
        records.append(record)
    followup = records[count:]
    rng.shuffle(followup)
    records = records[:count] + followup
    output.mkdir(parents=True, exist_ok=True)
    for start, end, filename in ((0, count, "review.md"), (count, len(records), "followup.md")):
        lines = ["# Blind expert consequence review", "",
                 "Reviewer name/role: ", "Years working with this platform: ",
                 "Involvement in seed scores, prompts, or original verdicts: ",
                 "Prior exposure to final ordering: ", "Review date: ",
                 "Business context, deployment assumptions, and relevant SLAs: ", "",
                 "Assume each described capability fails in the same deployment context. "
                 "Which failure has greater business consequence? Consider blocked work, "
                 "affected users, delivery disruption, and recovery/workarounds. "
                 "Do not judge failure probability or implementation complexity.", "",
                 "Use A, B, TIE, or UNKNOWN. UNKNOWN means missing context prevents a decision. "
                 "Record a short reason and assumptions for every answer. Do not inspect "
                 "the scoring key, seed scores, or LLM ordering while judging. "
                 "Complete review.md before followup.md, preferably in a later session "
                 "without consulting earlier answers. Prior familiarity must be disclosed.", "",
                 "This sheet contains only feature and scenario descriptions; existing "
                 "business-impact rationales are omitted because their provenance is uncertain.", ""]
        for number in range(start, end):
            row = records[number]
            lines.extend([f"## Q{number + 1:03d}", ""])
            for side in ("a", "b"):
                case = cases[row[side]]
                lines.extend([f"**{side.upper()}: {case['mapped_feature_name']}**",
                              case["test_scenario"], ""])
            lines.extend(["Answer: ", "Reason / assumptions: ", "Confidence (low/medium/high): ", ""])
        (output / filename).write_text("\n".join(lines))
    provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = {"seed": seed, "sampling": "uniform without replacement from supplied judged pairs",
                "n_primary": count, "n_cases_covered": len({r[s] for r in records[:count] for s in ("a", "b")}),
                "source_sha256": provenance, "records": records}
    with (output / "scoring_key.DO_NOT_OPEN_BEFORE_REVIEW.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    return {"primary_pairs": count, "followup_checks": 16,
            "cases_covered": manifest["n_cases_covered"], "output": str(output)}


def parse_answers(path):
    text = path.read_text()
    answers = {}
    for number, block in re.findall(r"^## Q(\d+)\s*\n(.*?)(?=^## Q|\Z)", text, re.M | re.S):
        match = re.search(r"^Answer:[ \t]*([^\n]*)", block, re.M)
        answer = match.group(1).strip().upper() if match else ""
        if answer not in {"", "A", "B", "TIE", "UNKNOWN"}:
            raise ValueError(f"Q{number}: answer must be A, B, TIE, or UNKNOWN")
        if int(number) in answers:
            raise ValueError(f"Duplicate question Q{number}")
        answers[int(number)] = answer
    return answers


def summarize(manifest, answers):
    records = manifest["records"]
    if set(answers) - set(range(1, len(records) + 1)):
        raise ValueError("Answers contain unknown question IDs")
    human, llm, canonical = [], [], {}
    counts = {"A": 0, "B": 0, "TIE": 0, "UNKNOWN": 0, "unanswered": 0}
    for number, row in enumerate(records, 1):
        answer = answers.get(number, "")
        canonical[number] = row[answer.lower()] if answer in {"A", "B"} else answer
        if row["kind"] == "primary":
            counts[answer or "unanswered"] += 1
            if answer in {"A", "B"}:
                human.append(answer == "A")
                llm.append(row["llm_winner"] == row["a"])
    n = len(human)
    agreement = sum(a == b for a, b in zip(human, llm)) / n if n else None
    ph, pl = (sum(human) / n, sum(llm) / n) if n else (0, 0)
    chance = ph * pl + (1 - ph) * (1 - pl)
    kappa = (agreement - chance) / (1 - chance) if n and chance < 1 else None
    checks = {}
    for kind in ("swap", "repeat"):
        usable = [(canonical[number], canonical[row["original"]])
                  for number, row in enumerate(records, 1) if row["kind"] == kind
                  and canonical[number] not in {"", "UNKNOWN"}
                  and canonical[row["original"]] not in {"", "UNKNOWN"}]
        checks[kind] = {"n_evaluable": len(usable),
                        "consistency": sum(a == b for a, b in usable) / len(usable) if usable else None}
    return {"complete_primary": counts["unanswered"] == 0, "primary_answer_counts": counts,
            "n_decisive_primary": n, "agreement_with_llm": agreement, "cohens_kappa_binary": kappa,
            "consistency_checks": checks,
            "limitations": ["Single reviewer; prior involvement/exposure is recorded separately.",
                            "Agreement is conditional on sampled existing pairs.",
                            "Pairs share cases; descriptive agreement is not an independent-sample noise ceiling.",
                            "Ties and unknowns are reported and excluded from binary kappa.",
                            "Follow-up measures human consistency, not LLM repeat-run stability.",
                            "Sparse sample does not define a validated whole-suite human ordering."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("--output", type=Path, required=True)
    for name in ("cases", "pairs", "verdicts"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    if args.command == "prepare":
        if any(getattr(args, name) is None for name in ("cases", "pairs", "verdicts")):
            parser.error("prepare requires --cases, --pairs, and --verdicts")
        result = prepare(args.output, args.cases, args.pairs, args.verdicts, args.count)
    else:
        manifest = json.loads((args.output / "scoring_key.DO_NOT_OPEN_BEFORE_REVIEW.json").read_text())
        answers = {}
        for name in ("review.md", "followup.md"):
            parsed = parse_answers(args.output / name)
            if answers.keys() & parsed.keys():
                raise ValueError("Duplicate question IDs across review sessions")
            answers.update(parsed)
        result = summarize(manifest, answers)
        with (args.output / "agreement_report.json").open("x") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
