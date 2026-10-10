"""Score scenario text with the paper's pointwise impact rubrics."""

import argparse
import hashlib
import json
import logging
import os
import re
import time
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def load_rubric(source):
    prompts = (Path(__file__).parent / "prompts.md").read_text()
    rubrics = re.findall(r"```text\n(.*?)```", prompts, re.S)
    return rubrics[0 if source == "llm_bia_labeler.py" else 1].strip()


def create_llm(model, temperature):
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is required for the explicitly invoked hosted scorer")
    return ChatGoogleGenerativeAI(model=model, api_key=key, temperature=temperature,
                                 top_k=40, top_p=.8, include_thoughts=False,
                                 thinking_budget=0 if "flash" in model else None,
                                 thinking_level="low" if "pro" in model else None)


def parse_score(content):
    if isinstance(content, list):
        content = " ".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    start, end = content.find("{"), content.rfind("}")
    value = json.loads(content[start:end + 1])["bia_score"]
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise ValueError("Invalid score")
    return value


def run(cases_path, output, limit=None, scorer="both"):
    if output.exists():
        raise ValueError("Output exists; choose a new destination")
    df = pd.read_csv(cases_path)[["case_key", "test_scenario"]]
    if df.empty or not df.case_key.is_unique or df.test_scenario.isna().any():
        raise ValueError("Cases must be nonempty with unique keys and complete scenario text")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if limit is not None:
        df = df.head(limit)
    # Suppress request logging: only anonymized progress is sent to the terminal.
    logging.getLogger().setLevel(logging.ERROR)
    output.mkdir(parents=True)
    settings = [("flash", "gemini-3.1-flash-lite", 0.0, 1, "llm_bia_labeler.py"),
                ("pro", "gemini-3.1-pro-preview", 0.3, 2, "gemini_pro_bia_pointwise.py")]
    settings = [item for item in settings if scorer == "both" or item[0] == scorer]
    manifest = {"input_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                "input_fields": ["test_scenario"], "rationale_included": False,
                "feature_name_included": False, "settings": [], "complete": False}
    for label, model, temperature, samples, source in settings:
        llm = create_llm(model, temperature)
        rubric = load_rubric(source)
        manifest["settings"].append({"model": model, "temperature": temperature,
                                     "samples_per_case": samples, "rubric_source": source,
                                     "thinking_budget": 0 if label == "flash" else None,
                                     "thinking_level": "low" if label == "pro" else None,
                                     "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest()})
        (output / "status.json").write_text(json.dumps(manifest, indent=2))
        records = []
        def score_case(row):
            prompt = f"{rubric}\n\nTest scenario: {row.test_scenario}\n\nJSON:"
            values = []
            responses = []
            for sample in range(samples):
                for attempt in range(4):
                    try:
                        response = llm.invoke(prompt)
                        score = parse_score(response.content)
                        record = {"case_key": row.case_key, "sample": sample,
                                  "score": score, "response": response.content,
                                  "response_metadata": response.response_metadata,
                                  "usage_metadata": response.usage_metadata}
                        responses.append(record)
                        values.append(score)
                        break
                    except Exception as error:
                        if attempt == 3:
                            (output / "status.json").write_text(json.dumps({**manifest,
                                "failed_model": model,
                                "error_type": type(error).__name__}, indent=2))
                            raise RuntimeError(f"Scoring failed for {label}; see private output") from None
                        time.sleep(2 * (attempt + 1))
            return {"case_key": row.case_key, f"{label}_scenario_score": sum(values) / samples}, responses
        rows = df.itertuples()
        # Validate one complete case before submitting the rest to the provider.
        first = score_case(next(rows))
        with ThreadPoolExecutor(max_workers=6) as executor:
            for index, (record, responses) in enumerate(chain([first], executor.map(score_case, rows))):
                records.append(record)
                with (output / f"{label}_responses.jsonl").open("a") as handle:
                    for response in responses:
                        handle.write(json.dumps(response, default=str) + "\n")
                pd.DataFrame(records).to_csv(output / f"{label}_scores.csv", index=False)
                if (index + 1) % 10 == 0 or index + 1 == len(df):
                    print(f"{label}: {index + 1}/{len(df)} complete", flush=True)
    manifest["complete"] = True
    (output / "status.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scorer", choices=["both", "flash", "pro"], default="both")
    args = parser.parse_args()
    run(args.cases, args.output, args.limit, args.scorer)
