# BRIGHT paper code companion

Code and technical details for *Business-Consequence-Aware Regression Test
Prioritization: An Industrial Case Study*.

Companion repository: [BRIGHT-BusinessTestRanker](https://github.com/abozten/BRIGHT-BusinessTestRanker).

The package implements nested feature-group Ridge evaluation, matched baselines
and ablations, paired bootstrap intervals, Bradley–Terry reference sensitivity,
expert-pair agreement, retrospective CI replay, and scenario-only impact scoring.
Industrial inputs and fitted outputs remain private. Supply your own inputs to
run the analyses.

[METHOD_DETAILS.md](METHOD_DETAILS.md) documents the algorithms, feature encodings,
metrics, review protocol, replay rules, and timing boundaries.
[config.json](config.json) records experiment settings; runners define their
constants in source. [prompts.md](prompts.md) contains the scoring rubrics and
input templates. [provenance.json](provenance.json) records source provenance.

## Setup

Use Python 3.13 and the listed dependency versions:

```sh
uv venv --python 3.13
uv pip install -r requirements.txt
.venv/bin/python check.py
```

The evaluation baseline loads a cached `sentence-transformers/all-MiniLM-L6-v2`
model with `local_files_only=True`; prepare the cache before running it.
`scenario_only_control.py` calls Gemini and reads `GEMINI_API_KEY` from the
environment. The other analyses use local inputs.

## Inputs and commands

Commands run from this directory. Input and output paths are placeholders for
locations outside the repository. Each runner requires a new output destination.

### Nested evaluation

The cases CSV requires `case_key`, `gt_rank`, `feature_name`,
`mapped_feature_name`, `test_scenario`, `history_fail_rate`, `semantic_score`,
`bia_risk_label`, `impact_scope`, and `domain`. Flash and Pro CSVs require
`case_key` with `llm_bia_score` and `pro_bia_score`, respectively.

`evaluation.py` implements the paper's 70-case protocol: five outer and three
inner feature-group folds, training-partition preprocessing, and global linear
relevance `71 - gt_rank`. It writes `metrics.json`, `oof_predictions.json`, and
`matched_baselines.tex`. The main model is named `ridge` in both JSON outputs.

```sh
.venv/bin/python evaluation.py --cases /private-input/cases.csv \
  --flash /private-input/flash.csv --pro /private-input/pro.csv \
  --output /private-output/evaluation
```

### Bradley–Terry sensitivity

Cases require `case_key` and `bia_score` in source-row order. Pairs require
`pair_id,i,j`; verdicts require `pair_id,winner`, with zero-based case indices.
The reference CSV requires `case_key,gt_rank`. The audit verifies the
seed-centered refit and compares a neutral prior center at the same strength.

```sh
.venv/bin/python target_audit.py --cases /private-input/seed-cases.csv \
  --pairs /private-input/pairs.csv --verdicts /private-input/verdicts.csv \
  --reference /private-input/reference.csv --output /private-output/target-audit
```

### Reference uncertainty

`target_uncertainty.py` resamples judged pairs for 2,000 Bradley–Terry refits.
It uses the same cases, pairs, verdicts, and reference schemas as the audit.
Cases additionally require `mapped_feature_name` and `test_scenario` for the
rank-interval CSV and LaTeX figure. The summary is `target_uncertainty.json`.

```sh
.venv/bin/python target_uncertainty.py --cases /private-input/seed-cases.csv \
  --pairs /private-input/pairs.csv --verdicts /private-input/verdicts.csv \
  --reference /private-input/reference.csv --output /private-output/target-uncertainty
```

### Expert review

Review cases require `case_key`, `mapped_feature_name`, and `test_scenario` in
source-row order. Use the pair/verdict schemas above. Preparation samples 40
pairs, randomizes sides, and adds eight order-swapped and eight same-order checks.
Scoring reads the completed worksheets and their saved scoring key.

```sh
.venv/bin/python expert_review.py prepare --cases /private-input/seed-cases.csv \
  --pairs /private-input/pairs.csv --verdicts /private-input/verdicts.csv \
  --output /private-output/review
.venv/bin/python expert_review.py score --output /private-output/review
```

### Retrospective replay

Supply the cases and Pro CSVs, the evaluation's OOF JSON, and a JSON array of CI
records in first-stored-attempt order. Each record has `job_id` and
`processed_data`; each feature has `feature_name`, `created_at`, and `scenarios`;
each scenario has `scenario_name`, `scenario_status`, and
`scenario_execution_time` in seconds. Normalize feature/scenario aliases
consistently before running.

Replay reconstructs 180-day history from earlier recorded timestamps. Area
metrics count failed cases, and time measures assume serial, order-invariant
costs. The Ridge strategy is named `ridge_oof` in `replay.json`.

```sh
.venv/bin/python ci_replay.py --cases /private-input/cases.csv \
  --pro /private-input/pro.csv --predictions /private-output/evaluation/oof_predictions.json \
  --jobs /private-input/jobs.json --output /private-output/replay
```

### Paired uncertainty and alternative references

`uncertainty_controls.py` reconstructs raw fold scores at the saved selected
alphas and verifies them against the evaluation metrics and calibrated OOF
scores. It performs 10,000 paired feature-group draws within fixed outer folds
and paired whole-job replay draws. It evaluates the same fitted predictions
against neutral-prior ranks and expert-labelled pairs. Intervals condition on
the saved fits and hyperparameters. Cross-fold expert comparisons use
training-percentile scores for learned models and fold-scaled heuristics;
score ties receive half credit.

```sh
.venv/bin/python uncertainty_controls.py --cases /private-input/cases.csv \
  --seed-cases /private-input/seed-cases.csv --flash /private-input/flash.csv \
  --pro /private-input/pro.csv --metrics /private-output/evaluation/metrics.json \
  --oof /private-output/evaluation/oof_predictions.json --pairs /private-input/pairs.csv \
  --verdicts /private-input/verdicts.csv \
  --expert-key /private-output/review/scoring_key.DO_NOT_OPEN_BEFORE_REVIEW.json \
  --expert-review /private-output/review/review.md \
  --replay /private-output/replay/replay.json \
  --scenario-scores /private-output/scenario-only --output /private-output/controls
```

The optional `--scenario-scores` directory contains `flash_scores.csv` and
`pro_scores.csv`, with `case_key` and `flash_scenario_score` or
`pro_scenario_score`. Seed cases preserve source-row indices used in the pair
and expert records and require complete `bia_score` values. `controls.json`
groups the primary comparison results under `evaluation`.

### Heuristic weight sensitivity

`weight_sensitivity.py` reads `gt_rank`, `pro_bia_score`, `bia_risk_label`,
`history_fail_rate`, and `is_foundational`. It samples 20,000 non-negative weight
vectors uniformly from the simplex. This descriptive whole-suite analysis uses
whole-suite standardization and the supplied foundational indicator. Its tier
mapping is recorded in `config.json`. The matched evaluation above uses
training-partition standardization and a feature/scenario text rule.

```sh
.venv/bin/python weight_sensitivity.py --features /private-input/weight-features.csv \
  --output /private-output/weight-sensitivity.json
```

### Scenario-only scoring

The scorer sends `test_scenario` text to the configured Gemini models and stores
responses and usage metadata in the output directory. It uses the rubrics in
`prompts.md`, with one Flash sample and two Pro samples. Pro uses low thinking.
This comparison varies both the input context and Pro's thinking setting.

```sh
.venv/bin/python scenario_only_control.py --cases /private-input/cases.csv \
  --output /private-output/scenario-only
```

## Package contents

Use this directory as the companion repository root. The `.gitignore` lists the
source, configuration, prompt, provenance, and documentation files included in
the package. Keep datasets, review records, fitted models, and generated outputs
outside the repository. Reproducing the paper's numerical results requires the
study inputs. A distribution license has not yet been assigned.
