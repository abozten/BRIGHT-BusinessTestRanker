# BRIGHT paper code companion

Methodological source for *Business-Consequence-Aware Regression Test Prioritization:
An Industrial Case Study*.

This standalone package contains revised Ridge evaluation and ablations,
Bradley–Terry seed sensitivity, expert-pair agreement, retrospective failure-event
replay, experiment settings, and exact feature-scoring prompts. It contains **no
industrial test cases or project data**: no descriptions, feature caches, seed
assessments, judgments, reviewer answers, job records, case-level predictions,
model bundles, or paper figures. The original RankNet/LambdaMART settings are
listed in `config.json`; their training runners are outside this package.

## Electronic appendix

[METHOD_DETAILS.md](METHOD_DETAILS.md) supplies the detailed feature definitions,
backend input/encoding table, ranking losses, evaluation conventions, expert-review
protocol, replay mechanics, and timing boundaries moved out of the paper.
[config.json](config.json) lists exact settings and grids;
[prompts.md](prompts.md) preserves both exact scoring rubrics and their input template.
These supplements contain methodological details only, without industrial case data.

## Setup

Use Python 3.13 and the listed dependency versions:

```sh
uv venv --python 3.13
uv pip install -r requirements.txt
.venv/bin/python check.py
```

`config.json` documents the experiment settings; constants used by each runner
are visible in its source. The file is a configuration record, not a dynamic
configuration loader. `provenance.json` records source-code checksums and packaging
changes. No hosted inference or database connection is made by these runners.
The raw-text baseline requires a previously cached MiniLM model and uses
`local_files_only=True`; obtain the model separately before running that analysis.

## Inputs and commands

All inputs are supplied privately by the caller. Paths below are placeholders for
files kept **outside this repository**; no study input or sample case dataset is
bundled. Existing output destinations are preserved rather than overwritten.

### Revised matched evaluation

The cases CSV requires `case_key`, `gt_rank`, `feature_name`,
`mapped_feature_name`, `test_scenario`, `history_fail_rate`, `semantic_score`,
`bia_risk_label`, `impact_scope`, and `domain`. Flash and Pro CSVs require
`case_key` with `llm_bia_score` and `pro_bia_score`, respectively. The runner
implements the paper's 70-case, 5-outer/3-inner-fold protocol and global relevance
`71 - gt_rank`. It writes metrics, calibrated OOF scores, and a LaTeX table to
an explicitly supplied private output directory.

```sh
.venv/bin/python revision_metrics.py --cases /private-input/cases.csv \
  --flash /private-input/flash.csv --pro /private-input/pro.csv \
  --output /private-output/revised
```

### Bradley–Terry sensitivity

Cases require `case_key` and `bia_score` in the original positional order.
Pairs require `pair_id,i,j`; verdicts require `pair_id,winner` with a zero-based
case index. The reference CSV requires `case_key,gt_rank`. The audit verifies the
seed-centered refit and compares neutral-centered regularization at the same
prior strength.

```sh
.venv/bin/python target_audit.py --cases /private-input/seed-cases.csv \
  --pairs /private-input/pairs.csv --verdicts /private-input/verdicts.csv \
  --reference /private-input/reference.csv --output /private-output/target-audit
```

### Expert review

The same pairs/verdicts schema is used. Review cases additionally require
`mapped_feature_name` and `test_scenario`. Preparation writes private review
worksheets and a scoring key; scoring reads completed worksheets. The command
randomizes sides and includes order-swapped and same-order repeated checks.

```sh
.venv/bin/python expert_review.py prepare --cases /private-input/seed-cases.csv \
  --pairs /private-input/pairs.csv --verdicts /private-input/verdicts.csv \
  --output /private-output/review
.venv/bin/python expert_review.py score --output /private-output/review
```

### Retrospective replay

Supply the cases and Pro CSVs, the revised OOF predictions JSON, and a private
JSON array of CI documents **in first-stored-attempt order**. Each document has
`job_id` and `processed_data`; each feature record has `feature_name`,
`created_at`, and `scenarios`; each scenario has `scenario_name`,
`scenario_status`, and `scenario_execution_time` in seconds. Normalize aliases
consistently in private inputs before running; project-specific aliases are excluded
from this package. Replay uses prior recorded timestamps for failure history and
serial, order-invariant scenario costs. Area metrics count failed cases, not
identified distinct defects.

```sh
.venv/bin/python ci_replay.py --cases /private-input/cases.csv \
  --pro /private-input/pro.csv --predictions /private-output/revised/oof_predictions.json \
  --jobs /private-input/jobs.json --output /private-output/replay
```

### Bootstrap uncertainty and heuristic weights

`grouped_uncertainty.py` reproduces the archived paired feature-group comparison
and pair-bootstrap rank intervals. Its archived predictions CSV requires
`feature_name`, `gt_rank`, `bia_risk_label`, `is_foundational`, `llm_bia_score`,
`pro_bia_score`, `history_fail_rate`, and `oof_pred_score`. Cases additionally
require `mapped_feature_name` and `test_scenario` for the privately generated
uncertainty figure. The generated figure and case-level rank intervals remain
in the external output directory. This archived comparison uses its original
whole-suite heuristic scaling, distinct from the revised fold-restricted analysis.

```sh
.venv/bin/python grouped_uncertainty.py --predictions /private-input/archived-predictions.csv \
  --cases /private-input/seed-cases.csv --pairs /private-input/pairs.csv \
  --verdicts /private-input/verdicts.csv --reference /private-input/reference.csv \
  --output /private-output/bootstrap
.venv/bin/python weight_sensitivity.py --features /private-input/archived-predictions.csv \
  --output /private-output/weight-sensitivity.json
```

Weight sensitivity reads `gt_rank`, `pro_bia_score`, `bia_risk_label`,
`history_fail_rate`, and `is_foundational`. It retains the archived whole-suite
standardization and reports descriptive correlations over 20,000 weight vectors.

## Publication contents

The `.gitignore` permits only the named source, configuration, prompt, provenance,
and documentation files. Keep generated analyses and input data outside the
repository. Publish this folder as the root of a separate repository, rather than
publishing the backend or research workspace. A repository URL and distribution
license have not yet been assigned. Running the code on the confidential study
inputs is required to reproduce its empirical numbers; the source alone documents
and exposes the methods for use with independently supplied inputs.
