# Methodological supplement

This document serves as the paper's electronic appendix. It records algorithms,
feature definitions, evaluation conventions, and implementation choices.
`config.json` contains exact grids, model identifiers, temperatures, and seeds;
`prompts.md` contains the scoring rubrics and input template;
`README.md` provides input schemas and commands. Industrial inputs, fitted models,
case-level outputs, reviewer answers, and execution records are excluded.

## 1. Representation and feature construction

Each privately supplied input row is a feature–scenario pair. The evaluated
cache joins attributes at scenario level.

- **History**: failed / (passed + failed), excluding skipped attempts; zero when
  there are no completed attempts. The runtime default is a preceding 180-day
  window. Evaluation uses cached rates; replay reconstructs the window from recorded timestamps.
- **Centrality**: number of scenarios sharing the mapped feature, computed from
  suite structure without target ranks.
- **Foundational indicator**: case-folded keyword matching for access-gating
  concepts in feature and scenario text. The exact lexicon
  is visible in `analysis_helpers.py`; replay uses the rule in `ci_replay.py`.
- **Documentation impact**: embedding retrieval finds relevant chunks, and a
  zero-shot NLI model assigns impact bands. If similarity is `s_i` and the band
  weight is `w(label_i)`, the continuous estimate is
  `b = sum_i [s_i / sum_j(s_j)] * w(label_i)`. The cached `semantic_score`
  stores this retrieval/NLI estimate.
- **Impact tier**: map the continuous estimate to Critical, High, Moderate, Low,
  and Minimal using the descending thresholds in `config.json`. The corresponding
  label weights are 0.95, 0.78, 0.55, 0.25, and 0.10; Unknown has weight 0.50.
  Empty retrieval defaults to 0.50, which enters the Moderate tier and combines
  missing evidence with moderate estimates.
- **Pointwise LLM impact**: two model variants score conditional failure damage
  from 0–100. Numerical seed scores and target ranks are omitted from the
  feature-scoring templates; the context includes an impact rationale. The exact
  templates, including their JSON output requests, are in `prompts.md`.
- **LLM disagreement**: for training-standardized Flash and Pro scores `z_l`
  and `z_p`, disagreement = `abs(z_l - z_p)`. Ridge uses the two scores and
  disagreement. This attribute measures the difference between impact estimates.
- **Taxonomy**: an LLM assigns functional domain and impact scope from feature
  and scenario text against an expert-defined allowed vocabulary. Unmapped values
  receive Unknown. The domain vocabulary itself belongs to private input metadata. The exact taxonomy
  model identifier was not retained in the existing records.

## 2. Reference construction

The reference represents consequence conditional on failure. Cases are sorted by
pre-existing BIA seed score; neighbor comparisons use offsets one and two.
Deterministic pseudorandom long-range anchors use the seed and anchor budget in
`config.json`. Presentation sides follow source indices. The judge receives
feature and scenario descriptions, excluding numerical BIA seed scores and
developer-written impact rationales. Seed scores guide pair sampling and center
the Bradley–Terry prior.

For a win-count matrix `W`, let `G = W + W.T`. Bradley–Terry predicts
`P(i > j) = sigmoid(theta_i - theta_j)`. The Gaussian-prior MAP gradient is

```text
grad_i = sum_j W_ij - sum_j G_ij * sigmoid(theta_i - theta_j)
         - lambda * (theta_i - prior_mu_i)
```

The prior mean standardizes the seed assessment. `analysis_helpers.py` uses
mean-centered log strengths, learning rate 0.1, at most 5,000 iterations, and
maximum-update tolerance 1e-8. Sort strengths descending to obtain positive
priority rank `R`; learned models predict `-R`. Numerical seed assessments are
used for reference construction. Ranker inputs are listed below.

`target_audit.py` compares the seed-centered and neutral-centered priors at the
same strength, verifies the frozen-reference refit, and checks strong connectivity
of the directed win graph. Strong connectivity is required for a finite
unregularized Bradley–Terry estimate. Neutral regularization changes the prior
center while retaining the sampled comparisons and their verdicts.
`target_uncertainty.py` resamples comparison rows with replacement, refits with the
seed-centered prior, and records rank intervals, Top-10 membership, and ordering
agreement. Its figure is generated in the caller's output directory.

## 3. Ridge inputs and encoding

| Attribute | Ridge representation |
| --- | --- |
| Failure history | Standardized numeric |
| Flash and Pro estimates | Standardized numeric |
| LLM disagreement | Training-derived, standardized numeric |
| Structural centrality | Standardized numeric |
| Impact tier | Ordinal, standardized numeric |
| Functional domain | One-hot indicators |
| Impact scope | Ordinal, standardized numeric |

The continuous retrieval/NLI score enters through its impact tier. The
foundational indicator is used by heuristics and the strict no-LLM ablation.
Impact tiers map Minimal, Low, Moderate, High, and Critical to 0–4; Unknown maps
to 2. Scope maps Local, Module, and Global to 0–2; Unknown maps to 1. Each fitted
transform uses the current training partition. Unseen domains map to the
all-zero indicator vector.

## 4. Ridge objective and selection

Ridge scores `x` using `w.T @ x + intercept`, with weighted squared loss on `-R`.
For training ranks, importance weight is

```text
importance(R) = 1 + 2 * (R_max + 1 - R) / R_max
R_max = max rank in the training partition
```

Feature groups remain intact in outer and inner GroupKFold partitions. All fitted
transforms and derived disagreement use the current training partition.
Nested tuning selects each outer-fold model using inner held-out Spearman scores.
Non-nested selection is confined to choosing a configuration for full-data refitting,
separate from the primary outer-fold generalization estimate. The complete
variant-to-feature map is visible in `evaluation.py`.

## 5. Evaluation and baselines

Spearman and Kendall compare scores with negated reference rank. Top-10 overlap is
`|reference_top10 ∩ predicted_top10| / 10`, with cutoff reduced to the number of
available rows if fewer than ten are evaluated. Exact score ties use randomized
ordering for head-of-list metrics.

NDCG uses global linear relevance `71 - R` for the paper's ranking protocol:

```text
DCG@10 = sum_{i=1..10} relevance(predicted_i) / log2(i+1)
NDCG@10 = DCG@10 / maximum attainable DCG@10
```

Primary metrics are fold-size-weighted outer-fold averages. For secondary pooled
OOF rankings, a held-out score is mapped to its empirical percentile in its model's
training-score distribution. Midpoint treatment of training-score ties is implemented
in `calibrate_scores_by_train_ecdf`; calibration uses no held-out labels.

Paired inference resamples feature groups, keeping their scenarios together. Within
fixed outer folds, correlations are recomputed and aggregated by sampled fold size;
percentile intervals summarize the paired correlation difference. Pair bootstrap
for the target and feature-group bootstrap for model comparisons have distinct
resampling units: judged pairs and feature groups, respectively.

Transparent heuristic score combines standardized Pro impact, inferred BIA tier,
and failure rate, plus the foundational flag. Equal weights are `(1,1,1,1)`;
the hand weights are `(3,1.5,1,0.5)`. Matched evaluation fits scaling on training
partitions and uses the feature/scenario foundational rule. Descriptive weight
sensitivity standardizes the whole suite, uses the supplied foundational flag,
and samples 20,000 weight ratios from a Dirichlet(1,1,1,1) simplex. Its tier map
is Critical=5, High=4, Moderate=3, Low=2, Minimal=1, Unknown=0.

The no-pointwise-LLM variant retains model-generated taxonomy and retrieval/NLI;
the strict no-LLM variant retains only history, structural counts, and a
foundational rule over feature/scenario text. The raw-text baseline embeds only
feature and scenario with a cached MiniLM model, omitting impact rationales,
then fits nested Ridge. Embedding loads are local-only.

## 6. Expert-pair audit

Uniformly sample existing pairs without replacement and randomize presentation
sides. Review sheets present feature/scenario text while omitting seed scores,
LLM verdicts, and pre-existing impact rationales. Ask for A, B, TIE, or UNKNOWN,
with role, experience, prior involvement/exposure, reasons, assumptions, confidence,
and deployment context. Separate follow-up checks include order swaps and
same-order repeats; counts and sampling seed are in `config.json`.

Primary agreement and binary Cohen's kappa use decisive A/B judgments only.
The study records 26/40 exact agreements (65%; binary Cohen's kappa 0.295).
The authors' qualitative assessment attributes disagreements to fine ordering
among similarly important actions, such as selection clearing and filtering.
Binary agreement and kappa treat every reversed preference equally and do not
measure the size of a consequence difference. Consequence gaps were not rated
separately. Consistency maps presentation sides back to case identities. Follow-up checks
are kept outside the primary agreement denominator and assess human consistency.
The output worksheets, answer records, and scoring key remain outside the repository.

## 7. Retrospective replay and timing

Replay retains the first stored passed/failed attempt per matched case, excluding
skips, unknown statuses, unmatched cases, and subsequent attempts. Historical rates
use strictly earlier recorded timestamps within the configured window. Ordering
uses frozen scores or reconstructed history; deterministic score ties use normalized
case keys. Random permutations form the feedback baseline. All input records and
outputs are supplied and stored outside the public repository.

For `n` observed cases and `m > 0` failed cases at positions `t_i`, failure-event area is

```text
failure_APFD = 1 - sum_i(t_i)/(n*m) + 1/(2*n)
```

The cost-weighted analogue averages the fraction of summed execution cost remaining
after each failed case, adding half its own cost. Time to first failure sums costs
through the first failed case. No-failure jobs have undefined area metrics; invalid
or non-positive costs leave cost/time metrics undefined. These are serial,
order-invariant measures with one event per failed case. Replay uses recorded
feature timestamps and fixed observed outcomes. Partial jobs contribute their
observed cases.

Cached ranking timing covers saved preprocessing, linear scoring, and descending
sorting. Warm runs exclude process startup and model deserialization; cold runs
include loading a saved bundle and its first rank. The repetition counts and thread
setting are in `config.json`. The paper's timing measurements use a saved Ridge
bundle with the same linear scoring form and an earlier feature encoding.
Feature-store retrieval, complete CI integration,
provider generation cost, and energy are outside the timing boundary. Reordering
a mandatory full suite leaves total work unchanged when execution costs are fixed.

## 8. Paired uncertainty and reference controls

`uncertainty_controls.py` reconstructs raw outer-fold scores at the saved
inner-selected alphas, checking their Spearman values and training-ECDF scores
against the frozen outputs. For each of 10,000 draws, resample feature groups with
replacement separately within each fixed outer fold; copy all scenarios in each
selected group. Recompute within-fold Spearman and average by sampled fold size.
Use the same draw for every strategy and form paired score differences before
taking the 2.5th and 97.5th percentiles. These are conditional evaluation intervals,
without model refitting or repeated hyperparameter selection. Constant-score
draws receive zero correlation, consistent with the evaluation policy. The seed
score is evaluated as a diagnostic baseline.

Replay resamples entire eligible jobs with replacement jointly across strategies,
using saved per-job measures. Its intervals condition on the saved random-permutation
means. Jobs may share tests and environmental conditions; intervals condition
on the observed jobs. Both sets of intervals are exploratory and unadjusted for
multiplicity.

Fit the neutral-prior BT reference to the same verdicts with zero prior center and
unchanged regularization. Re-score the fixed seed-trained strategies against its
ranks, retaining the seed-informed pair sampling and the same judged comparisons. Expert
checks map randomized answers back to case identities, using primary
pairs. For cross-fold comparisons, learned and fold-scaled heuristic scores use
training-ECDF calibration; pointwise signals use their common numeric scales.
An exact score tie receives half credit. Report seed agreement and direct
LLM-verdict agreement separately. The Wilson 95% interval uses z=1.95996398454;
it is a nominal binomial interval since sampled pairs share cases.

`scenario_only_control.py` uses only scenario text, the scoring rubrics,
and the configured model identifiers, temperatures, and sample counts. It removes
feature names as well as impact rationale. Pro uses low thinking, a different
setting from the context-rich scores. The comparison measures sensitivity to
input context and runtime settings. Response,
usage, and model metadata are retained privately. Model settings and seeds are
recorded in `config.json`; all case-level inputs and outputs stay external.

The pairwise judge is `claude-opus-4.8`, as identified by the author. The
retained report labels it "Claude (in-session)". Full provider requests,
responses, and the model snapshot were not retained.

## Repository scope

The companion contains source, settings, and scoring templates. Industrial
inputs, fitted models, review answers, execution records, and generated outputs
remain private. Commands and input schemas are in `README.md`.
