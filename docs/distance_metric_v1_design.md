# Distance metric v1: working design

**Status:** working design doc, not the preregistered spec. Sections 1–6 define
the metric. Section 7 fixes pair selection, thresholds, budget, estimator and
failure criterion before any outcome data exists. Section 9 lists what is still
blocked or needs review. The only code written for this pass is baseline 3
(`gitm/research/distance_metric/baselines.py`).

## 0. What is being claimed

**Target: rank preservation.** For two operating points *s* (source) and *t*
(target), and the set *C* of candidates runnable at both, measure each
candidate's throughput delta at both points. The pair's outcome is the rank
agreement of the two delta vectors:

\[ \tau_{st} = \tau_b(\Delta_s(C),\ \Delta_t(C)) \]

The metric *d(s, t)* is computed from pre-experiment information only. Its
claim is ordinal: **smaller d ⇒ higher τ.** A near source's ranking can be
reused at the target; a far one cannot.

v1 is judged on two separate verdicts (section 7.5):

- **Validity:** d orders pairs by τ.
- **Utility:** on near pairs, the source's measured ranking predicts the target
  better than baseline 3 does. Baseline 3 is the target's own `predict_delta`
  ranking, with no measurement anywhere.

## 1. Inputs, and what is never an input

d may read only the following. Every item exists before any candidate is run.

| Input | Where it comes from |
|---|---|
| Model graph | `gitm/planner/registry.py` at the point's bridged `BatchConfig` (section 2) |
| Hardware | `PlannerContext.sku`, `num_gpus`, `HardwarePeak` |
| Operating point | `Regime` (`gitm/traffic/regime.py`) plus the bridged `BatchConfig` |
| Baseline config | knob values read off the baseline engine (`vllm_knobs.get_knob`) |
| Candidate specs | `load_library(workload)`, expanded per point (section 3) |
| Baseline capture | the point's baseline `Trace`, scheduler stats, and `Residuals` from a run with **no candidate applied** |

**Never an input**, even indirectly:

- any measured candidate delta;
- `RunRecord`s or `prior_runs` (`use_history` must stay off; see section 8);
- `verification.json`, A/B results, or kept/rolled-back outcomes;
- anything produced after Phase 4 starts.

A baseline capture counts as pre-experiment. It is a measurement of the
untouched deployment, and the loop takes it before choosing candidates. If a
quantity is only knowable from a candidate run, it is an outcome.

## 2. Operating point: representation and bridge (REVIEW)

There are two descriptions of "where the system is running":

- **Traffic side:** `Regime`, i.e. the input/output length percentiles,
  `io_ratio`, `burstiness`, `concurrency`, `rate_rps` and `source_kind`. The
  playbook matches on these.
- **Graph side:** `BatchConfig(batch, kv_cache_len, ...)`. The graph, the
  predicted per-op times, the stand-in coverage and the roofline residuals are
  all priced at this point.

**Decision.** The graph side is canonical for d. The operating-point
component (O) and every graph-derived component (G, B, S) are evaluated at the
bridged `BatchConfig`. The traffic side is kept as its own component (T), on
match.py's axes. That way shape differences invisible at the graph point
(p95 tails, burstiness) still count.

**Bridge, from `Regime` + baseline capture + baseline config to `BatchConfig`:**

1. **`batch`:**
   - If the baseline capture has scheduler stats, use `round(mean_running)`.
     This is exactly what the loop does (`_batch_config_from_stats`).
   - Otherwise, for a closed-loop regime (`concurrency` set), use
     `min(concurrency, max_num_seqs)`, with `max_num_seqs` read off the
     baseline engine.
   - Otherwise (open-loop, nothing measured) the point is **not bridgeable**
     and therefore not eligible. Deriving a batch from rate × latency would
     mean guessing a latency.
2. **`kv_cache_len`:** `input_p50 + output_p50 // 2`, the KV length at
   mid-decode. This is the convention the Kimi/MI355X search-space stand-in
   already used: 4096/512 gives 4352 (`scripts/search_space/feasible_kimi_mi355x.py`
   on `search-space/kimi-mi355x`).
3. **TP / `num_gpus`:** from the baseline config. These are gated (section 4),
   not compared.

**Two things the bridge must not be confused with:**

- **The loop's own graph point.** `_batch_config_from_stats` deliberately leaves
  `kv_cache_len` at the 128 default (`loop.py:404`). A `predicted_graph.json`
  from a loop run is priced at kv = 128 and is **not** reused. d builds its own
  graph at the bridged point, with the same planner version for both points of
  a pair.
- **The gate's `kv_cache_len`.** `build_planner_context` fills
  `GateContext.kv_cache_len` from the engine's `max_model_len`
  (`context.py:_engine_kv_len`). That is a config ceiling, not a point.
  Applicability (section 3) uses the loop's gate context unchanged, so the
  candidate set is the one the loop would actually see. `max_model_len`
  differences are covered by the baseline config component (K).

**Open for review:**

- Is mid-decode the right single KV summary? The alternative is the
  token-weighted mean over the decode, which comes to the same value for fixed
  output length but differs for distributions.
- Should prefill (`prefill_tokens`) be bridged too? v1 prices the decode step
  only, matching what the graphs model by default.

## 3. Candidate identity and the common candidate set (REVIEW)

**Source pool.** Catalog levers only: `load_library(workload)`. Autoresearch
proposals are excluded, because whether they exist, and what they are called,
depends on the point's bottleneck class. A pair would then differ in candidate
identity by construction.

**Identity.** "Same candidate" means **the same expanded spec name**. Each point
expands relative levers against its *own* baseline engine
(`expand_relative_candidates`), so `max_num_seqs_x2` at *s* and at *t* is the
same candidate, even if *s* resolves it to 512 and *t* to 1024. Reasons:

- Name is what the loop runs, records, and keys history on
  (`(lever, gpu_sku, fingerprint)`).
- A transferred ranking is only actionable at the target if it is expressed in
  the target's own resolution.

The resolved values are recorded per point, and any difference in baseline
values feeds component K.

**Sensitivity analysis (exploratory).** Recompute τ on the subset of *C* whose
resolved `knob_values` are equal at both points. If the conclusions change,
relative-lever resolution is driving the result, and that gets reported.

**Expansion collapse.** `expand_relative_candidates` drops grid points that
collapse onto an already-queued value (`vllm_knobs.py:317`), for example when
the current value is 0 or when clamping hits `value_max`. A name can therefore
exist at one point and not the other. That is handled by the intersection
below, not patched.

**Common set, fixed before measurement:**

C_pre(s, t) = names that are:

- present in both points' expansions, **and**
- `applicable(spec, gate)` at both points (each point's own loop gate context),
  **and**
- not vetoed at either point. The vetoes are a structural knob with no restart
  hook (`knob_kind`), and `unmet_prerequisite(engine, knob)`.

The `Policy` is fixed for the whole experiment, so policy rejections don't vary
by point: `Policy(require_qualification_commit=False, skip_high_risk=True,
use_history=False)`.

**Measured set.** C_valid ⊆ C_pre is the candidates with a valid measurement at
both points. Failures are counted, never silently dropped (section 7.6).

## 4. The distance

Shared helpers, each with range [0, 1] and value 0 at equality:

- **rel(a, b)** = `1 − min(a, b) / max(a, b)` for a, b > 0. It is 0 if both are
  equal (including both 0) and 1 if exactly one is ≤ 0. It equals
  `1 − 2^(−log2_ratio(a, b))`, a monotone map of match.py's per-axis metric,
  so the two systems agree on order.
- **TV(p, q)** = `½ Σ |p_i − q_i|`, the total-variation distance between two
  share vectors over the union of keys.

### Gates (d = ∞, pair ineligible)

d is ∞, and the pair is ineligible, if the two points differ in any of:

- model and revision;
- `gpu_sku`;
- `num_gpus` and `tensor_parallel_size`;
- engine and version;
- weight dtype (`GateContext.dtype`);
- workload.

This is the same family as match.py's exact gates. A cross-SKU or cross-model
transfer is a different question from the one v1 asks.

### Components

| | Component | Definition | Basis |
|---|---|---|---|
| **O** | operating point | `max(rel(batch_s, batch_t), rel(kv_s, kv_t))` on bridged coordinates | bridge |
| **T** | traffic shape | `max` over match.py `DEFAULT_AXES` of `1 − 2^(−AXIS_METRICS[a](s, t))`. Same axes and metrics as the playbook; `rate_rps` excluded per `RATE_AXIS_DECISION` | `Regime` |
| **K** | baseline config | `max` over the knob set Kset of: `rel` for numerics, 1 for any bool/str/None mismatch | baseline engine |
| **G** | graph profile | TV of predicted per-op time shares (`predicted_per_op`, normalised) | graph at bridged point |
| **B** | bound mix | TV of predicted time shares by `normalize_bound` class | graph at bridged point |
| **S** | candidate scope | mean over c ∈ C_pre of \|cov_s(c) − cov_t(c)\| | trace (see basis rule) |
| **R** | residual profile | `max` of the three terms listed below | captured baseline traces only |

The three terms in R are:

1. \|Δ `serialized_concurrency_fraction`\|;
2. \|Δ fraction of observed device time in roofline-memory-bound kernels\|;
3. TV of observed per-op device-time shares by `observed_op`, with
   `<unmodeled>` as its own bucket.

**Kset** = the union of knob names in C_pre's `knob_values`, plus `max_num_seqs`,
`max_num_batched_tokens`, `max_model_len`, `enable_chunked_prefill`,
`enforce_eager`, `kv_cache_dtype`, `enable_prefix_caching`,
`gpu_memory_utilization` and `block_size`. If a knob is unreadable on both
points it is skipped; if it is unreadable on one point it scores 1.

**Combination: d = max(O, T, K, G, B, S, R)**, i.e. L-inf, as in match.py:

- One large difference is enough to break transfer, and it should not be
  averaged away by six small ones.
- The limiting component is reported with every d, so a far pair says *why*
  it is far.

v1 does not weight components. Weights would have to be fitted on outcomes,
which section 1 forbids for v1.

### Coverage without reimplementing `_applies`

cov(c) is the fraction of device time in kernels where c applies. It is computed
as `predict_delta(trace, c, delta_mean=1.0)`. `predict_delta` returns
`coverage * delta_mean` (`gitm/optimizer/replay.py:51-53`), so with `delta_mean=1.0` the
result is exactly the coverage. It uses the private `_applies` as-is,
including:

- `whole_step` → 1.0;
- empty `applies_to_kernels` → 0;
- the `observed_op` match, then the substring fallback;
- 0.0 when device time ≤ 0.

**No reimplementation exists, so there is nothing to keep in sync.** If
`predict_delta` ever stops being `coverage × delta_mean`, S changes with it,
and that is the intended coupling: S measures the scope the ranking actually
uses.

### Basis rule

- S and R are computed on the **captured baseline trace at both points**.
- If either point lacks a capture:
  - S falls back to the predicted-graph stand-in at both points (one synthetic
    kernel per node, as in the search-space doc);
  - R is undefined, and the pair is labelled `basis=stand-in`.
- Bases are never mixed within a pair.
- **Confirmatory analysis uses `basis=captured` pairs only.** Stand-in pairs
  are exploratory.

### Recorded but not in d for v1

These are recorded with each pair for exploratory analysis only:

- **`classify_bottleneck` class flips:** a step function of R's terms, so it
  would double-count them.
- **Largest-residual-op flips:** a single-op summary of R's third term.
- **Per-op `r_kt` means:** zero by construction on the stand-in, and
  interval-based on captures.
- **dr / Granger hypotheses:** these need a monitor window, which baseline
  captures do not have.
- **Lever priors (`expected_delta_*`):** identical at s and t for a given
  name, so they cannot separate pairs. They only enter through baseline 3.

## 5. Outcome: what "measured delta" means here

There are two delta definitions in the codebase. They are **not** pooled.

| | Loop (`EngineApplicator.measure`) | This experiment |
|---|---|---|
| Metric | decode tok/s, `speedup − 1` | output tok/s, `median(cand) / median(base) − 1` (`row_from_runs` semantics) |
| Reps | `GITM_AB_REPS`, default 1, **mean** | ≥ 3, **median** |
| Ordering | baseline measured once and reused; candidate reps back-to-back | baseline blocks interleaved with candidate blocks (section 7.3) |
| Returned value | `delta − noise_band` (a keep gate, not an effect size) | raw effect; the noise is carried in the interval, not subtracted |

**Decision.** Confirmatory outcomes use the experiment's definition only.
Loop-produced deltas (raw `speedup − 1` from `EngineABResult`, **not** the
returned `delta − noise_band`) may be analysed in a separate exploratory table.
They are never merged with the confirmatory ones. Both are fractions, so
`throughput_pct / 100 ≡ speedup − 1` and the units line up. Latency is recorded
but not ranked in v1.

## 6. Rank agreement and ties

**Per-pair statistic: Kendall τ_b** over C_valid.

- τ_b corrects for ties in both vectors.
- It counts pairwise ordering decisions, which is what "the source's ranking
  would have picked X before Y" means.
- It behaves sensibly at n ≈ 6–12.
- Spearman was not chosen: with ties and small n its value is harder to read
  as a decision count.

**Ties.** Exact float ties in measured deltas are rare. The real problem is
*noise-level* ties: two candidates within each other's noise. Two treatments,
fixed now:

- **Primary:** τ_b on the medians, with a rep-resampling interval (section 7.4)
  that carries the noise.
- **Secondary (reported):** concordance among *resolved* candidate pairs only.
  A pair is resolved when the interval of the difference of the two
  candidates' deltas excludes 0 at both points. If few pairs are resolved,
  the pair is flagged rather than reported as a weak τ.

Predicted rankings (baseline 3) produce exact ties as a matter of course:
equal coverage × prior, or `delta = 0` for every rejected candidate. τ_b
handles them, and the tie count is reported next to each baseline τ.

**Top-k (secondary).** The loop spends `top_n_interventions = 5` slots, so two
secondary measures are reported:

- does the source's top-1 land in the target's top-3?
- the overlap of the two top-5 lists.

## 7. Pair selection, thresholds, budget, estimator, failure criterion

Everything in this section is fixed before any candidate is measured.

### 7.1 Points and eligible pairs

**Points:** a preregistered grid of operating points on one deployment:

- Kimi K2.5 on MI355X;
- TP = 8;
- one engine version;
- fixed-length synthetic requests (rag-style), so the `Regime` axes are
  controlled.

The only axes varied are concurrency and input length.

**Proposed grid (7 points, 21 unordered pairs):**

- c ∈ {32, 40, 48, 64, 96, 128} at 4096/512;
- plus c = 64 at 8192/512.

On O alone, this grid gives:

- near-ish pairs: 32–40 (0.20), 40–48 (0.17), 48–64 (0.25), 96–128 (0.25);
- far pairs: 32–64 (0.50), 48–96 (0.50), 64–128 (0.50), and others;
- the 8192 point sits at kv rel ≈ 0.48 from every 4096 point.

The other components can only raise d (L-inf), so the real bins come from the
pre-measurement check in 7.2, not from this arithmetic.

**A pair (s, t) is eligible iff all of these hold:**

1. s ≠ t, unordered.
2. All gates pass (section 4).
3. Both points are bridgeable (section 2).
4. Both points have a captured baseline trace, with the same planner version
   for both graphs.
5. \|C_pre(s, t)\| ≥ 6.

Eligibility is decided from pre-experiment information only.

### 7.2 Thresholds

| Bin | Rule | Reading |
|---|---|---|
| near | d ≤ 0.25 | no component differs by more than 1.33× (or 25% of the time mix) |
| intermediate | 0.25 < d < 0.5 | measured and reported, not in the near/far test |
| far | d ≥ 0.5 | some component differs by ≥ 2× (or ≥ 50% of the time mix) |

The thresholds bracket the factor-2 that match.py's `log2_ratio` treats as
distance 1.

**Pre-measurement bin check.** After the baseline captures and before any
candidate runs:

1. Compute d for every eligible pair.
2. Require at least 6 near pairs and at least 6 far pairs.
3. If that fails, adjust the grid (add or move points) and repeat.

Each adjustment is recorded with its d table. It uses no outcome, so it doesn't
compromise preregistration.

### 7.3 Sample size and budget

- **Candidates per point:** C_pre is capped at 8 by a fixed rule:
  1. take baseline 3's ranking at the point;
  2. go down the list, keeping candidates in the pair-wise common set;
  3. stop at 8.

  The cap is chosen before measurement, from predicted quantities only. It
  biases C toward plausible levers, which is the population the loop draws
  from.
- **Reps:** 3 per arm.
- **Arm order:** blocked `B C1 B C2 B … C8 B`, with 3 reps per block. Fully
  interleaved A/B/A/B per rep (the search-space doc's protocol) would cost two
  restarts per rep for every structural candidate (`knob_kind == "structural"`).
  Blocking with a baseline between every candidate is the compromise, and the
  B blocks give the drift check (7.6). Hot-swap (scheduling) candidates use the
  same block order for comparability, even though they could interleave
  cheaply.
- **Self-pair (noise ceiling):** one designated point (c = 64, 4096/512) is
  measured twice, in two independent blocked passes. τ_self is the τ_b between
  the two passes.
- **Cost model (upper bound, all candidates structural):**
  \[ \text{wall} \approx (P + 1)\,(2|C| + 1)\,(t_{reload} + 3\,t_{window}) \]
  With P = 7, |C| = 8, t_window ≈ 150 s and t_reload of 10–60 min, that is
  roughly **40 h (10-min reload) to 155 h (60-min reload)**. Restarts dominate.
  Each hot-swap candidate in C removes two reloads per point. The real figure
  depends on C's structural/scheduling mix, which is known once C_pre is fixed.
  Confirming t_reload on the target deployment is a prerequisite for the spec
  (section 9).

### 7.4 Estimator and intervals

- **Per pair:** τ_st = τ_b over C_valid on the median deltas. Its 95% interval
  comes from 4000 redraws: resample reps with replacement within each (arm,
  point), recompute both medians and τ_b.
- **Validity statistic:** τ_validity = Kendall τ_b between d and τ_st across
  conclusive eligible pairs. Kendall again, because d is only claimed to be
  ordinal. **Expected sign: negative.**
- **Near–far contrast:** Δτ = mean τ_st(near) − mean τ_st(far).
- **Interval for both:** a cluster bootstrap over operating points. Pairs
  share points, so they are not independent. Each of 4000 draws:
  1. resample the P points with replacement;
  2. keep every eligible pair among distinct drawn points;
  3. recompute τ_validity and Δτ.

  Report the 95% percentile intervals.

Why this estimator: the claim is ordinal at both levels (d ordinal, τ ordinal),
and the dependence structure is points, not pairs. A regression of τ on d would
assume a functional form the metric doesn't claim.

### 7.5 Failure criterion

These are classified like the P1 doc: supported, rejected, or inconclusive.
They are evaluated in order, and the first that applies wins.

1. **Inconclusive:** any run-level gate in 7.6 fails.
2. **Rejected:**
   - the Δτ 95% interval upper bound is < 0.2 (near is confidently not
     materially better than far), **or**
   - the τ_validity 95% interval lower bound is > 0 (d orders pairs the wrong
     way).
3. **Supported:** the Δτ interval lower bound is > 0 **and** the τ_validity
   interval upper bound is < 0.
4. **Inconclusive:** otherwise.

**Utility (separate verdict, only meaningful if validity is supported).** On
conclusive near pairs, compute the paired difference
`τ_b(Δ_s, Δ_t) − τ_b(baseline3_t, Δ_t)`. Here baseline3_t is the full
ranking from `catalog_predict_delta_ranking` on t's baseline trace, restricted
to C_valid. Its interval uses the same cluster bootstrap.

- Interval lower bound > 0: **v1 beats baseline 3.**
- Interval upper bound < 0: **baseline 3 beats v1.**
- Otherwise: **no difference shown.**

The comparisons against the exact-match and bottleneck/affinity baselines are
added once those baselines exist (section 9).

### 7.6 Inconclusive pairs and runs

No pair is dropped. Every eligible pair gets exactly one status, and the status
table is part of the result.

| Pair status | Rule | Remedy |
|---|---|---|
| `conclusive` | none of the below | |
| `small_set` | \|C_valid\| < 6 | widen the cap, or accept the pair as uninformative |
| `measurement_failure` | > 25% of C_pre failed at either point | fix the failing lever or harness, then re-measure |
| `tie_dominated` | > 50% of candidate pairs tied (difference interval contains 0) in either measured ranking | more reps, or a larger-effect candidate set |
| `noisy` | median rep-to-rep CV of output tok/s > 5% at either point | longer window, or a quieter node |
| `point_drift` | B blocks at a point differ by > 2% (first vs last median) | re-measure the point; all pairs on it inherit this status |

**The run is inconclusive (no validity verdict) if any of these hold:**

- fewer than 6 conclusive pairs in the near bin, or fewer than 6 in the far
  bin;
- τ_self < 0.6. The measurements can't reproduce their own ranking, so no
  cross-point τ is interpretable;
- fewer than 50% of eligible pairs are conclusive.

Each failure names the gate and the remedy. Conclusive-pair counts by bin are
always reported. The near and far means are never computed over non-conclusive
pairs.

## 8. Baseline 3 (implemented)

**Function:**
`gitm.research.distance_metric.baselines.catalog_predict_delta_ranking(trace,
library, policy, *, ctx=None, top_n=None)`.

It is a pure wrapper over `select_interventions` as called on `main`:

- no `current_values`, no `baseline_noop`;
- the sort key is unchanged: rejected, then delta ≤ 0, then demoted, then
  −delta, then name.

It deliberately differs from the loop's call site in two ways:

- **No history, `gpu_sku` or `fingerprint`.** History carries measured
  outcomes, which section 1 excludes. The loop's default is history off
  anyway: `LoopConfig.use_history=None`, so `prior_runs` is None.
- **`top_n` defaults to the full library**, not the loop's 5, because τ needs a
  complete ranking. Passing `top_n=5` reproduces the loop's cut.

## 9. Blocked, deferred, and needing review

- **Needs review:**
  - the operating-point bridge (section 2);
  - candidate identity by expanded name (section 3);
  - the near/far thresholds and grid (section 7.2);
  - the |C| = 8 cap rule (section 7.3).
- **Deferred:** the exact-match-only and bottleneck-class / keyword-affinity
  baselines. These are blocked on confirming baseline fidelity, so they are
  not built and their comparison rows in 7.5 are left open.
- **Deferred:** the preregistered spec in `docs/experiments/` format. This doc
  is its input.
- **Prerequisite for the spec:** t_reload and measurement-window length on the
  target MI355X deployment (cluster access), which fix the budget in 7.3.
