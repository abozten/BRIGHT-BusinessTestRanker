# Methodological supplement

This document serves as the paper's electronic appendix. It records algorithms,
feature definitions, evaluation conventions, and implementation choices.
`config.json` contains exact grids, model identifiers, temperatures, and seeds;
`prompts.md` contains the archived scoring rubrics and input template;
`README.md` provides input schemas and commands. Industrial inputs, fitted models,
case-level outputs, reviewer answers, and execution records are excluded.

## 1. Representation and feature construction

Each privately supplied input row is a feature–scenario pair. The evaluated
cache joins attributes at scenario level rather than averaging feature groups.

- **History**: failed / (passed + failed), excluding skipped attempts; zero when
  there are no completed attempts. The runtime default is a preceding 180-day
  window. A runtime default alone does not establish a frozen input's temporal cutoff.
- **Centrality**: number of scenarios sharing the mapped feature, computed from
  suite structure without target ranks.
- **Foundational indicator**: case-folded keyword matching for access-gating
  concepts. The archived rule examines feature, scenario, and rationale; revised
  heuristics and the strict no-LLM baseline omit the rationale. The exact lexicon
  is visible in `analysis_helpers.py`; replay uses the rule in `ci_replay.py`.
- **Documentation impact**: embedding retrieval finds relevant chunks, and a
  zero-shot NLI model assigns impact bands. If similarity is `s_i` and the band
  weight is `w(label_i)`, the continuous estimate is
  `b = sum_i [s_i / sum_j(s_j)] * w(label_i)`. The cached `semantic_score`
  stores this retrieval/NLI estimate, rather than raw cosine similarity.
- **Impact tier**: map the continuous estimate to Critical, High, Moderate, Low,
  and Minimal using the descending thresholds in `config.json`. The corresponding
  label weights are 0.95, 0.78, 0.55, 0.25, and 0.10; Unknown has weight 0.50.
  Empty retrieval defaults to 0.50, which enters the Moderate tier and combines
  missing evidence with moderate estimates. The cache cannot reconstruct a raw
  cosine-only experiment.
- **Pointwise LLM impact**: two model variants score conditional failure damage
  from 0–100. Numerical seed scores and target ranks are omitted from the
  feature-scoring templates; the context includes an impact rationale. The exact
  templates, including their JSON output requests, are in `prompts.md`.
- **Derived LLM signals**: for standardized Flash and Pro scores `z_l` and `z_p`,
  mean = `(z_l + z_p)/2` and disagreement = `abs(z_l - z_p)`. The mean is redundant
  with the two constituent columns. Disagreement is a feature, not a calibrated
  uncertainty estimate.
- **Taxonomy**: an LLM assigns functional domain and impact scope from feature
  and scenario text against an expert-defined allowed vocabulary. Unmapped values
  receive Unknown. The domain vocabulary itself belongs to private input metadata. The exact taxonomy
  model identifier was not retained in the existing records.
- **Optional clusters**: sentence embeddings, or a TF–IDF fallback, are grouped
  using k-means. Cluster count zero disables this candidate attribute.

## 2. Reference construction

The reference represents consequence conditional on failure. Cases are sorted by
pre-existing BIA seed score; neighbor comparisons use offsets one and two.
Deterministic pseudorandom long-range anchors use the seed and anchor budget in
`config.json`. Original presentation sides follow source indices, and the judging
context contains feature text, scenario, rationale, and numerical seed score.
This design is seed-informed rather than uniform all-pairs sampling.

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
used for reference construction, rather than included as feature columns.

`target_audit.py` compares the seed-centered and neutral-centered priors at the
same strength, verifies the frozen-reference refit, and checks strong connectivity
of the directed win graph. Strong connectivity is required for a finite
unregularized Bradley–Terry estimate. Neutral regularization changes the prior
center while retaining the originally sampled comparisons and judging context.
`grouped_uncertainty.py` resamples comparison rows with replacement, refits with the
seed-centered prior, and records rank intervals, Top-10 membership, and ordering
agreement. Its figure is generated only into the caller's external output directory.

## 3. Original backend inputs

This table describes the archived configurations. Ordinal = monotone encoding;
one-hot = categorical indicators; target mean = training-domain mean negated rank.
A dash means the input is not consumed. Optional clusters were disabled in the
selected configurations. The original RankNet and LambdaMART training runners are
not bundled; their configuration grids are provided in `config.json`.

| Attribute | Ridge | RankNet | LambdaMART |
| --- | --- | --- | --- |
| Retrieval/NLI score | — | Numeric | Numeric |
| Failure history | Numeric | Numeric | Numeric |
| Foundational flag | — | Binary | Binary |
| Flash and Pro estimates | Numeric | Numeric | Numeric |
| LLM mean | Numeric | — | Numeric |
| LLM disagreement | Numeric | — | Numeric |
| Structural centrality | Numeric | — | Numeric |
| Impact tier | Ordinal | One-hot | Ordinal |
| Functional domain | Target mean | One-hot | Target mean |
| Impact scope | Ordinal | One-hot | Ordinal |
| Optional text cluster | One-hot | One-hot | One-hot |

Training-domain target means exclude held-out labels but include each training
row's own target; unseen levels receive the training mean. Original derived LLM
mean and disagreement used whole-suite standardization. Revised Ridge removes the
mean, constructs disagreement using training-partition standardization, and uses
one-hot domains. Unseen one-hot levels map to the all-zero vector.

## 4. Ranking losses and selection

Ridge scores `x` using `w.T @ x + intercept`, with weighted squared loss on `-R`.
For training ranks, importance weight is

```text
importance(R) = 1 + 2 * (R_max + 1 - R) / R_max
R_max = max rank in the training partition
```

RankNet uses a shared ReLU network with dropout, scalar score `s`, and ordered-pair
loss `log(1 + exp(-(s_i - s_j)))`. Pair weights are the maximum importance of the
two cases. Original pair construction excludes same-impact-tier pairs and requires
a minimum rank gap of one. Training uses AdamW, one shuffled pass over all pairs
per epoch, and gradient clipping at norm one. Exact hidden-layer, dropout,
learning-rate, decay, epoch, and optional cluster settings are in `config.json`.

LambdaMART uses LightGBM 4.6.0 `lambdarank`, a single suite query, truncation 128,
and linear relevance/gains. Gradients scale by the NDCG change of a pair swap.
Its search spans leaves, boosting rounds, learning rate, minimum leaf size, and
L2 regularization; predictions average eight seeds. Exact grids are in the config.

Feature groups remain intact in outer and inner GroupKFold partitions. All fitted
transforms and revised derived disagreement use the current training partition.
Nested tuning selects each outer-fold model using inner held-out Spearman scores.
Non-nested selection is confined to choosing a configuration for full-data refitting,
separate from the primary outer-fold generalization estimate. The complete revised
variant-to-feature map is visible in `revision_metrics.py`.

## 5. Evaluation and baselines

Spearman and Kendall compare scores with negated reference rank. Top-10 overlap is
`|reference_top10 ∩ predicted_top10| / 10`, with cutoff reduced to the number of
available rows if fewer than ten are evaluated. Exact score ties use randomized
ordering for head-of-list metrics.

Revised NDCG uses global linear relevance `71 - R` for the paper's ranking protocol:

```text
DCG@10 = sum_{i=1..10} relevance(predicted_i) / log2(i+1)
NDCG@10 = DCG@10 / maximum attainable DCG@10
```

The archived within-fold implementation instead inverts rank using the held-out
fold's maximum rank. Archived and revised NDCG values therefore use different
conventions. Spearman and Kendall are unaffected by this choice.

Primary metrics are fold-size-weighted outer-fold averages. For secondary pooled
OOF rankings, a held-out score is mapped to its empirical percentile in its model's
training-score distribution. Midpoint treatment of training-score ties is implemented
in `calibrate_scores_by_train_ecdf`; calibration uses no held-out labels.

Paired inference resamples feature groups, keeping their scenarios together. Within
fixed outer folds, correlations are recomputed and aggregated by sampled fold size;
percentile intervals summarize the paired correlation difference. Pair bootstrap
for the target and feature-group bootstrap for model comparisons have distinct
resampling units and should not be interpreted as a model-performance ceiling.

Transparent heuristic score combines standardized Pro impact, inferred BIA tier,
and failure rate, plus the foundational flag. Equal weights are `(1,1,1,1)`;
the hand weights are `(3,1.5,1,0.5)`. Revised scaling fits training folds and its
foundational rule excludes rationale. Archived whole-suite sensitivity samples
non-negative weight ratios uniformly from a Dirichlet(1,1,1,1) simplex. The two
protocols are distinguished in the scripts and README.

The no-pointwise-LLM variant retains model-generated taxonomy and retrieval/NLI;
the strict no-LLM variant retains only history, structural counts, and a
foundational rule over feature/scenario text. The raw-text baseline embeds only
feature and scenario with a cached MiniLM model, omitting shared impact rationales,
then fits nested Ridge. Embedding loads are local-only.

## 6. Expert-pair audit

Uniformly sample existing pairs without replacement and randomize presentation
sides. Review sheets present feature/scenario text while omitting seed scores,
LLM verdicts, and pre-existing impact rationales. Ask for A, B, TIE, or UNKNOWN,
with role, experience, prior involvement/exposure, reasons, assumptions, confidence,
and deployment context. Separate follow-up checks include order swaps and
same-order repeats; counts and sampling seed are in `config.json`.

Primary agreement and binary Cohen's kappa use decisive A/B judgments only.
Consistency maps presentation sides back to case identities. Follow-up checks
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
order-invariant failure-event measures, rather than identified-distinct-fault APFD
or calibrated severity-weighted effectiveness. Recorded feature timestamps require
interpretation before a prospective evaluation; partial jobs provide only observed
cases, and cross-test dependencies are outside this replay model.

Cached ranking timing covers saved preprocessing, linear scoring, and descending
sorting. Warm runs exclude process startup and model deserialization; cold runs
include loading a saved bundle and its first rank. The repetition counts and thread
setting are in `config.json`. Feature-store retrieval, complete CI integration,
provider generation cost, and energy are outside the timing boundary. Reordering
a mandatory full suite leaves total work unchanged when execution costs are fixed.

## Repository scope

This supplement exposes methodological choices while preserving industrial
confidentiality. It contains no empirical case records or fitted outputs. Exact
provider request logs and unrecorded historical metadata cannot be reconstructed
from code or templates. The companion is prepared locally; the paper's release
URL will be supplied when its separate repository is published.
