# Attribution methods survey: adopt/reject decisions for deviation attribution

> **DRAFT, not reviewed.** Two inputs this document depends on are still moving:
> Andrew's estimator status vocabulary (v0 not yet pushed) and his expanded
> mechanism list. Everything that depends on either is marked **[PENDING]** and
> isolated in §0 and §2 so it can be updated without restructuring. Every
> judgment call that should wait for those inputs is listed in §12.

**Pinned to** `main` at `28b7890`. The code cited here (`gitm/optimizer/dr.py`,
`attribution.py`, `separation.py`, `mechanism_fixtures.py`, `monitor.py`,
`gitm/tracer/schema.py`) and `docs/mechanism_model.md` are unchanged from
`origin/main` `543e3a8`.

**Empirical source.** Every claim about what dr.py *does* is cited from
`gitm/research/dr_probe.py`, which is seeded throughout and prints SHA-256
digests of its sweeps. **It is committed at `28b7890` on `main`.** Re-run from
the repo root with `.venv/bin/python -m gitm.research.dr_probe`. Its output depends on the
statsmodels version (0.15.0 here). The CPU-only worked example in §10 is a test
on `main`: `tests/test_attribution_survey_example.py`.

**Background source.** `docs/attribution_gap_audit.md` from PR #126
(`f60bf31`, unmerged). Its findings are cited as background evidence with their
gap IDs (G-nn). Its numerical checks publish no seeds or commands (the PR's own
review says so), so its numbers are quoted as reported, not re-verified here,
except where the probe reproduces the same behaviour independently.

---

## 0. Two vocabularies, kept separate

| | PR #126 audit, "separable?" column | Estimator status (Andrew) |
|---|---|---|
| Values | Partly, No, Condition only | **[PENDING: Andrew's v0 status names]** |
| Describes | What the **current code** can tell apart (diagnostic, backward-looking) | What an **estimator outputs** about one estimand (forward-looking) |
| Used here for | Background evidence only | Every adopt/reject decision |
| Mapped onto the other? | **No.** | **No.** |

Confirmed with Andrew: the audit's values are not to be merged into or mapped
onto the status scheme. This document never translates one into the other.

### 0.1 Status placeholders

Decisions below use two placeholder tokens. They are descriptions, not names.

| Placeholder | Meaning in this draft |
|---|---|
| `[PENDING-STATUS: estimable]` | The data determine the estimand (a point or a bound), and the estimator reports it with an uncertainty that has been checked. |
| `[PENDING-STATUS: not-identifiable]` | The data cannot determine the estimand. The estimator says so, with reasons and the remedy that would change the answer. |

Andrew's scheme has a third term, *absent*. This draft does not know whether it
means "the mechanism is estimated to be absent" or "the signal the estimand
needs is absent from the data", so **it is not used anywhere below**. See §12,
item 1.

### 0.2 Mechanisms

No decision below is keyed to a mechanism ID. Sections refer to "a mechanism
`M`" and to the *structure* of `docs/mechanism_model.md`: pairs of mechanisms,
each confounded at named observation layers and separated by something specific.
Where that document is quoted, the quote is as of the pinned commit. The
mechanism list is being expanded, so any mechanism-specific statement is an
example, not a dependency.

---

## 1. Evidence base: what dr.py does today

All rows E1-E6 are probe output at `28b7890`, run against dr.py
SHA-256 `5084791b4f77b82b`. Rows E7 and E8 are read from code.

| id | finding | probe scenario | numbers |
|---|---|---|---|
| E1 | **Direction-blind.** On dr.py's own test fixture (`tests/test_runtime_on_trace.py::test_attribute_dr_ranks_pairs`, seed 3), where the true direction is A→B, B→A ranks first. The test checks only that the notes string contains "doubly-robust ATE". | 4 | B→A ATE +1.006 (se 0.011); A→B ATE +0.780 (se 0.017). Both p underflow to 0.0, so the order is decided by \|z\|. |
| E2 | **Truncation hides a real treatment.** Every op's series is cut to the shortest op's length (`dr.py:122`). `lm_head` has one row per step (8), so attention's 64 treated rows (steps 2-3) are cut to step 0, layers 0-7, where none are treated. | 3a | 64/256 attention rows treated in the data; 0/8 after truncation; 0 hypotheses. |
| E3 | **Three different situations return the same empty result.** No anomaly, every row treated (no controls), and E2's truncation all return `RankedHypotheses(hypotheses=[])`. | 1, 2, 3a | 0 hypotheses each. Scenario 2: 256/256 treated in the data, 8/8 after truncation. |
| E4 | **Independent-null rejection rate, seed-range dependent.** A spikes at 8 random rows, B is independent noise, 40 rows, A→B is the only pair. | 5 | Seeds 0-199: 15/200 = 7.5% (Wilson 95% 4.6%-12.0%). Seeds 100-299: 26/200 = 13.0% (9.0%-18.4%). Pooled over the 300 distinct seeds 0-299: 34/300 = 11.3% (8.2%-15.4%). Nominal 5%. |
| E5 | **Autocorrelated-null rejection rate.** Two independent AR(1) series, φ = 0.9. | 6 | 200 rows, `doubly_robust_ate` directly: seeds 0-199 56/200 = 28.0% (22.2%-34.6%); seeds 500-699 61/200 = 30.5% (24.5%-37.2%). 40 rows through `attribute_dr`, seeds 0-199: 53/200 = 26.5% (20.9%-33.0%). |
| E6 | **A near-zero standard error passes the guard.** With a constant outcome, se is rounding error (~1e-17), not exactly 0, so `z = ate / se if se not in (0.0, inf)` (`dr.py:136`) divides noise by noise. An exactly-zero se was reached only with a constructed outcome of exact 0.0. | 7a, 7b | 7a, four pairs: se 3.6e-18 to 3.5e-17; p 4.9e-6, 0.007, 0.063, 0.199, all labelled "+ slower". 7b: ATE 0.0, se 0.0, z 0.0, p 1.0, still listed, labelled "- faster". |
| E7 | **The graph argument is ignored.** `attribute_dr(residuals, graph, ...)` (`dr.py:108`) never reads `graph`; every ordered op pair is a candidate. The audit reports the same for Granger's `attribute` (G-28). | code | `dr.py:108-154` |
| E8 | **The DR output decides nothing.** The loop computes it (`gitm/scheduler/loop.py:979`) and serializes the top 5 into `residuals.json` (`loop.py:1006-1010`); claims cite Granger or a live A/B. The audit reports the same (G-14). | code | `loop.py:979`, `1006-1010` |

How to cite E4: the inflation is **not established on seeds 0-199 alone** (the
interval contains 5%). It is supported by the pooled 300 seeds, whose interval
excludes 5%. The pooled figure is derived from the union of the two hit lists the
probe prints; the probe does not print it directly. E5 is the stronger finding:
every variant's interval excludes 5% by a wide margin.

Background from the audit that the probe did not test: series paired by launch
index rather than step (G-02, the root cause of E2), the only covariate being
position (G-13), 380 uncorrected Granger tests at 20 ops (G-12), non-stationary
series entering unfiltered (G-29), the two-sided treatment lumping fast and slow
anomalies (G-30), and the normal-approximation p-value at group size 3 (G-37).

---

## 2. Shared result type (proposed, not implemented)

Every category below proposes an estimator interface that returns the same
record. It is modelled on `separation.Decision` (`gitm/optimizer/separation.py:217-234`),
the one place in the repo that already returns an outcome together with reasons
and remedies. **Only `status` depends on Andrew's vocabulary**, so landing his v0
changes one type, not seven sections.

```python
@dataclass
class AttributionResult:                  # proposed
    status: "Status"                      # [PENDING: Andrew's v0 status type]
    estimand: str                         # e.g. "ATE of cause-op anomaly on effect-op r_kt"
    identified: str                       # what the data determine: the estimand, or a function/bound of it
    estimate: float | None
    interval: tuple[float, float] | None
    method: str                           # which section's estimator produced this
    reasons: list[str]                    # why this status, as separation.Decision.reasons
    remedies: list[str]                   # what would change it, as separation.REMEDY
    diagnostics: dict[str, Any]           # n_treated, n_control, rows used/dropped, seeds, ...
```

`identified` is separate from `estimand` because §8's partial identification
reports a bound or a function of the estimand, not the estimand itself.

---

## 3. Identification via causal graphs

**Decision.** **REJECT** dr.py's implicit graph: every ordered pair of ops is a
candidate cause→effect and the predicted graph is never read (E7). **ADOPT**
`docs/mechanism_model.md`'s pair-and-layer reasoning as the identification
framework. It is the closest thing in the repo to graph-based identification:
each pair of mechanisms is stated as *confounded* at named observation layers
and *separated by* a specific observation, telemetry or intervention
(`mechanism_model.md:17-24`), against four defined layers (`mechanism_model.md:60-73`).

| | |
|---|---|
| **Observed variables** | The four observation layers: residuals, documented residuals, raw trace, untraced serving (`mechanism_model.md:62-67`). |
| **Latent variables** | Mechanism parameters: slowdown factor, additive cost, efficiency, fixed cost, traffic excess (`mechanism_model.md:43-52`). |
| **Assumptions** | A0-A7 (`mechanism_model.md:298-307`), each listed with what it is used by and when it fails. |
| **Identifiable quantity** | Per pair: whether an available layer distinguishes the two mechanisms, and if not, what would. For example, P1's sweep identifies the slope and intercept of `d = a·t(x) + b` from two operating points and tests the line with three (`mechanism_model.md:166-180`). |
| **Failure modes** | (1) A shared driver outside the model (multi-rank waits, scheduler limits and noise are explicitly not covered, `mechanism_model.md:309-315`). (2) A pair missing from the list, which grows as the mechanism model expands. (3) Direction: with contemporaneous rows and symmetric dependence, cause and effect are not distinguishable at all. E1 shows dr.py ranking the wrong direction first on its own fixture. |
| **Telemetry or intervention** | The "separated by" column itself. Of the pairs listed today, two need an intervention (an operating-point sweep; varying z out of time order), one is cheap (the `off` arm), two need telemetry or code, one needs only code. |
| **Proposed interface** | `identifiability(pair, layers_available) -> AttributionResult`. Status `[PENDING-STATUS: not-identifiable]` when no available layer separates the pair; `reasons` names the layers checked; `remedies` quotes the "separated by" entry. The pair table is read from data, not hardcoded. |

**[PENDING] judgment call:** this assumes Andrew's expanded mechanism list keeps
the pairwise "confounded at layer / separated by" structure. If it moves to a
different structure (for example, many-way confusion sets), this interface
changes shape.

---

## 4. Adjustment

**Decision.** **REJECT** dr.py's current adjustment. **ADOPT** in principle, if
covariates for shared drivers become available, as the audit proposes (G-13:
clock, batch size, kv_len).

dr.py adjusts for one covariate: row position, `pos = np.arange(n)`
(`dr.py:123`, described as "a simple confounder proxy" at `dr.py:14-15`). Row
position is not time and not step. Each op's series is in trace order and then
truncated to the shortest (`dr.py:115-122`). Events carry no step, batch or
kv_len (`mechanism_model.md:88-89`). So the same position can mean different
steps for different ops (G-02), and E2 shows the consequence.

**What exists off the trace today, versus what needs new telemetry:**

| covariate | status | source |
|---|---|---|
| `start_ns`, `end_ns` | in every event | `gitm/tracer/schema.py:15-23` |
| `stream_id`, `device_id`, `pid` | in every event | `schema.py:15-23` |
| grid and block dims, shared memory, registers | in every kernel event | `schema.py:26-36` |
| op and layer from NVTX (`range_op`, `range_layer`) | only when the capture had ranges | `schema.py:39-44` |
| batch | **proxy only**: on a serving run the decode kernel's grid tracks the decode batch | `separation.py:172-174` (`window_stat` docstring) |
| step index | **needs telemetry** | `mechanism_model.md:88-89`; P3 (`:249-253`); A6 (`:306`) |
| batch size, kv_len per step | **needs telemetry** | `mechanism_model.md:88-89`, `:249-253` |
| clock, power state | **needs telemetry** | G-13 ("what would close it") |

**A code gap comes before the telemetry gap.** None of the trace-recoverable
covariates reaches dr.py. `attribute_dr` receives `Residuals`, and each
`KernelResidual` carries only op, layer, `r_kt`, `r_mt`, `t_obs_s`, `t_pred_s`,
`bound` and `n_classes` (`gitm/optimizer/monitor.py:23-52`). Timestamps,
streams and grids are dropped at that boundary.

| | |
|---|---|
| **Observed variables** | Treatment (cause-op residual out of band), outcome (effect-op residual), covariates from the table above. |
| **Latent variables** | Shared drivers: clock throttling, batch change, kv_len growth. |
| **Assumptions** | No unmeasured confounding given the covariates. Positivity: dr.py requires 3 treated and 3 control rows (`_MIN_GROUP`, `dr.py:42`), and it clips the propensity to [0.05, 0.95] (`dr.py:86`), which hides near-violations rather than reporting them. Independent rows: the standard error is `std / sqrt(n)` (`dr.py:104`). |
| **Identifiable quantity** | The ATE of a cause-op anomaly on an effect-op residual, conditional on the covariates, **if** rows are aligned by step and exchangeability holds. |
| **Failure modes** | A shared driver without a covariate (G-11, G-13). Misaligned rows (G-02; E2). Autocorrelation breaking the independent-rows standard error (E5: 26.5%-30.5% rejections under an independent null). |
| **Telemetry or intervention** | Step index per kernel (A6), batch and kv_len per step, clock. As a code change first: carry `start_ns`, stream and grid through `KernelResidual`. |
| **Proposed interface** | `adjusted_effect(rows, cause, effect, covariates) -> AttributionResult`, where `rows` is keyed by (step, layer), not by position. When a needed covariate is missing, status `[PENDING-STATUS: not-identifiable]` with a remedy naming the telemetry. |

---

## 5. Negative controls

**Decision.** **REJECT** the current state: neither dr.py nor attribution.py uses
a negative control. The word "control" appears in dr.py only as "control units",
meaning untreated rows (`dr.py:23`, `:40`, `:67`). **ADOPT** two precedents.

1. **`separation.py`'s control op.** `CONTROL_OP = "rms_norm"`
   (`separation.py:87-89`), justified in the code as "depends on batch and hidden
   size only, so a KV dtype change must not move it. A difference is node or
   clock drift, not the mechanism." `decide` gates on it: if the control op moves
   by more than `CONTROL_MAX = 0.02` (`:63-67`), or no control measurement exists,
   the decision abstains (`:350-354`) with `REMEDY["control"]` (`:101`).
2. **Ops outside a lever's `applies_to_kernels`.** For a claim motivated by a
   catalog lever, ops the lever does not name (`gitm/kernels/spec.py:63`) should
   not move. Levers with `whole_step=True` (`spec.py:64-70`) have no such ops and
   get no negative control from this rule.

**Caveat on the graph.** `rms_norm` rows exist only if the prediction graph has
an `rms_norm` node. Today only the GLM graph emits one
(`gitm/planner/glm_graph.py:613-631`). The dense default graph has six ops and
no norm, and `tests/test_separation.py:197` already works around this with
`monkeypatch.setattr(sep, "CONTROL_OP", "qkv_proj")  # fixtures have no norm kernel`.
The two code paths differ:

- `separation.window_stat` classifies **raw kernels** by name
  (`separation.py:180`, via `deviation.py:151`), so it finds norm kernels in a
  trace even when the graph has no norm node.
- dr.py sees only **residuals**, and `monitor.residuals` drops every kernel whose
  op has no graph node (`monitor.py:157-162`). On a dense graph, norm kernels
  never reach dr.py.

| | |
|---|---|
| **Observed variables** | The residual or raw duration of a control op, at the same rows as the treatment. |
| **Latent variables** | The shared driver the control is meant to detect. |
| **Assumptions** | The control shares the confounders but not the mechanism. This is the claim `separation.py:87-88` makes for `rms_norm` under a KV dtype change, and it must be argued per mechanism and per lever. |
| **Identifiable quantity** | Detection, not removal: if the control op shows an "effect" comparable to the effect op's, the attribution is refuted. |
| **Failure modes** | The mechanism also touches the control (a step-wide lever). The control op is absent from the graph (above). The control's own noise trips the gate: `tests/test_separation.py:437` (`test_noisy_control_op_does_not_void_the_run`) covers this for separation. |
| **Telemetry or intervention** | None, if the control op's kernels are captured. Code: either a graph node for the control or a raw-kernel path into the estimator. |
| **Proposed interface** | `negative_control(rows, cause, effect, control_ops) -> AttributionResult`. `control_ops` comes from data (the complement of the lever's `applies_to_kernels`, or configuration), never a hardcoded op name. When no control op is present, status `[PENDING-STATUS: not-identifiable]` for the "not confounded" claim, with a remedy naming the missing op. |

---

## 6. Placebo and permutation tests

**Decision.** **REJECT** the current state: no permutation or placebo inference
exists in the codebase. Every `shuffle` call in `gitm/` outside `gitm/research/`
samples data (`workloads.py`, `runtime_driver.py`, benchmarks,
`replay_validation.py`); none builds a null distribution. dr.py's p-value is a
two-sided normal approximation from the z-score (`dr.py:148`). **ADOPT
candidate**: a permutation null, using the probe's scenario 5 as the template
and its scenario 6 as the hard case.

Proposal, for one pair (cause, effect):

1. Compute the observed ATE with `doubly_robust_ate` (`dr.py:55`), as today.
2. Reassign the treatment B times under a named seed, recompute the ATE each
   time, and take
   `p_perm = (1 + #{|ATE_b| ≥ |ATE_obs|}) / (B + 1)`.
3. Calibrate on the probe:
   - **Scenario 5** (independent null): the rejection rate should fall to about 5%.
   - **Scenario 6** (AR(1)): an i.i.d. shuffle of the treatment destroys its
     autocorrelation and so builds too narrow a null; this draft expects it to
     stay anticonservative. A circular shift of the treatment relative to the
     outcome preserves both series' autocorrelation. **This is a hypothesis to
     test on scenario 6, not a result.**
4. For multiple pairs, permute once and take the maximum |statistic| over all
   pairs per draw, which addresses the uncorrected-tests problem (G-12).

A permutation p-value no longer divides by the standard error, so E6's
rounding-noise p-values (4.9e-6 on a constant outcome) cannot arise from it. A
zero-variance outcome still needs an explicit abstention (§9).

| | |
|---|---|
| **Observed variables** | Only what dr.py already has. No new telemetry. |
| **Latent variables** | None added. |
| **Assumptions** | Exchangeable treatment labels under the null (i.i.d. shuffle), or stationarity (circular shift). |
| **Identifiable quantity** | A calibrated test of the sharp null of no effect. Not an effect size. |
| **Failure modes** | Autocorrelation under an i.i.d. shuffle (above). Misalignment is untouched: permuting E2's truncated rows still tests the wrong rows. A shared driver still rejects the null, because permutation tests association, not causation; pair it with §5. |
| **Telemetry or intervention** | None. The cost is B extra `doubly_robust_ate` calls per pair, each fitting a statsmodels Logit; to be measured. |
| **Proposed interface** | `permutation_test(rows, cause, effect, *, seed: int, draws: int, scheme: Literal["iid", "circular"]) -> AttributionResult`. `seed` is required with no default, as in the probe and in task 2's `random_ranking`. |

---

## 7. Sensitivity analysis

**Decision.** **REJECT** for dr.py: it reports a point estimate and a p-value
with no statement of how strong an unmeasured confounder would have to be to
change the conclusion. No sensitivity analysis exists in `gitm/optimizer/` (the
only "sensitiv" matches are "case-insensitive" comments in `deviation.py`).
**ADOPT** the precedent in `separation.py`, although it does not apply to dr.py.

**`EPS_MAX_S`** (`separation.py:52-56`) is an assumed upper bound on how much the
tracer inflates one kernel's duration, "an assumption, stated in the report next
to every `m`". `decide` propagates it into a bound on how much tracer inflation
could leak into the fitted intercept, `tracer_leak_s = |n_c − k·n_b|·ε`, and
widens the additive margin to at least twice that leak
(`separation.py:377-380`). That is a sensitivity analysis in structure: an
unverifiable quantity stated as a number and carried into the decision, rather
than assumed to be zero.

| | |
|---|---|
| **Observed variables** | None new. |
| **Latent variables** | One stated sensitivity parameter: a bound on an unmeasured confounder's association with treatment and outcome. **[PENDING] judgment call** on which parameterisation (see §12). |
| **Assumptions** | The bound itself. It is unverifiable by construction and must be reported next to every result, as `EPS_MAX_S` is. |
| **Identifiable quantity** | An interval, or the confounder strength at which the conclusion would flip, rather than a point. |
| **Failure modes** | A loose bound makes everything abstain; a tight one gives false confidence. |
| **Telemetry or intervention** | None to compute it. The telemetry of §4 (clock, batch) shrinks the bound needed. |
| **Proposed interface** | `sensitivity(result, *, confounder_bound: float) -> AttributionResult`, returning the widened interval, with status `[PENDING-STATUS: not-identifiable]` when the interval crosses zero. |

---

## 8. Partial identification

**Decision.** **ADOPT** the pattern, which already exists in the repo in spirit
though not by name. dr.py has none: it reports a point and a p-value.

`docs/mechanism_model.md`, on P1's sweep: "This separates the multiplicative and
additive parts of the deviation, not `α` from `η`" (`mechanism_model.md:177-179`).
The slope `a = (1+α)/η` is identified; `α` and `η` separately are not. Comparing
with a reference sweep identifies `1+α` for a *change* (`a/a0 = b/b0 = 1+α`,
`:179-180`). That is partial identification: the data determine a function of
the parameters, and the document says which.

`separation.decide` acts on it with intervals and margins rather than points.
Effects within `KAPPA` (multiplicative) or the additive margin count as none.
When an interval straddles its margin, the outcome is `inconclusive`, with the
reason "the data cannot place the effect" (`separation.py:407-412`).

| | |
|---|---|
| **Observed variables** | Per pair; for P1, durations at three or more operating points. |
| **Latent variables** | The components that are not separately identified (for P1, `α` and `η`). |
| **Assumptions** | For P1: A4 (affine cost across the sweep) and A5 (fixed mechanism parameters across points), `mechanism_model.md:304-305`. |
| **Identifiable quantity** | A function of the parameters (P1's slope `a`), or an interval for one. |
| **Failure modes** | A broken assumption that looks like an effect: efficiency changing with operating point gives "a spurious intercept of either sign" (`mechanism_model.md:187-190`). |
| **Telemetry or intervention** | A reference sweep for attributing a change. `mechanism_model.md` does not say what would separate `α` from `η`; see §12. |
| **Proposed interface** | The shared result's `identified` field states which function is identified (for example "(1+α)/η"), with `interval` for it. Status `[PENDING-STATUS: estimable]` applies to that function only, never to its components. |

---

## 9. Explicit abstention

**Decision.** **REJECT** dr.py's silent abstention. **ADOPT** `separation.py`'s
outcome + reasons + remedy pattern, with status names **[PENDING: Andrew's v0
status names]**.

dr.py abstains in three places and reports none of them:

- fewer than two ops with four or more rows: returns an empty result (`dr.py:119-121`);
- a cause with fewer than 3 treated or 3 control rows: `continue`, no record (`dr.py:129-130`);
- `doubly_robust_ate` returns `(0.0, inf)` on degenerate input (`dr.py:72-73`). Its comment says "Refuse rather than emit a meaningless number", but `attribute_dr` never reaches it with such input, because it skips those causes first.

The return type cannot carry a reason: `RankedHypotheses` has one field,
`hypotheses` (`attribution.py:30-35`). So "no anomaly", "no controls" and "the
treatment was truncated away" return the same value (E3), and §10's third test
asserts it.

**It also fails to abstain when it should.** A constant outcome produces a
near-zero standard error that passes the guard (E6), so dr.py emits p-values
from rounding noise rather than abstaining.

**The precedent.** `separation.decide` returns a `Decision` whose `outcome` comes
from a closed tuple that includes `"inconclusive"` (`separation.py:91`). Each
failed gate appends a reason (`:232`, and throughout `:315-412`). `_finish` maps
failed gates to remedies through `REMEDY` (`:93-104`, `:416-418`).

**Reasons dr.py should report.** These are descriptions, not status names:

| situation | example | remedy |
|---|---|---|
| a treatment exists in the data but not in the rows kept after truncation | E2: 64 treated in the data, 0 kept | align rows by step (A6), or exclude or handle the op that sets the length |
| no control rows | E3, scenario 2 | more steps, or steps without the anomaly |
| no treated rows | E3, scenario 1 | none needed: this is the one empty result that means "nothing to explain" |
| zero or near-zero outcome variance | E6 | report the outcome as constant; no test is meaningful |
| propensity clipping active on many rows | `dr.py:86` | report the share clipped; positivity is in doubt |

**This is what should trigger a `[PENDING-STATUS: not-identifiable]`-style status**
(a "not identifiable from this data" status, in whatever form Andrew's v0 takes)
in all but the "no treated rows" case.

| | |
|---|---|
| **Observed variables** | Counts dr.py already computes and discards: rows per op, rows kept after truncation, treated and control counts, outcome variance. |
| **Latent variables** | None. |
| **Assumptions** | None beyond the estimator's own. |
| **Identifiable quantity** | Not an estimate: a statement of whether one exists, and why not. |
| **Failure modes** | Too many reasons and too few remedies make the output unusable. `separation.py` limits remedies to one per failed gate. |
| **Telemetry or intervention** | None for the reporting itself. The remedies name what each case needs. |
| **Proposed interface** | `attribute_dr` returns one `AttributionResult` per *candidate* pair, including those it could not estimate, instead of a list of the ones it could. |

---

## 10. Worked example (CPU-only): one identification, one silent abstention

`tests/test_attribution_survey_example.py`, on `main`. It uses the CPU-only
generator `gitm/optimizer/mechanism_fixtures.py` with a `RegionSlowdown` of
factor 2 (`alpha=1.0`) on `attn_score_value` at decode steps 2 and 3 only, batch
8, kv 2048, 8 steps, `noise_cv=0.0`, seed 0. It runs through the real
`monitor.residuals` and `dr.attribute_dr`, and is the probe's scenario 3.

Run: `.venv/bin/python -m pytest tests/test_attribution_survey_example.py -q`

**Identification (scenario 3b, `lm_head` removed):**
`test_without_lm_head_the_null_effect_is_identified_but_its_p_values_are_noise`.

- Ground truth: the slowdown touches attention only, so every other op's
  residuals equal the healthy run's exactly. The true effect of attention on
  each is 0.
- dr.py tests all four attention→X pairs with 64 treated and 192 control rows,
  and each ATE is within 1e-12 of 0. **The point estimate is identified.**
- **The inference is not.** Each outcome is constant, so se is rounding error
  (0 < se < 1e-12). It passes the guard, and each p-value is
  `erfc(|ATE/se|/√2)` of rounding noise. The test asserts p < 1 for every pair.
  The probe's printed values are 4.9e-6, 0.007, 0.063 and 0.199. This is a
  successful identification of a null effect with unusable p-values, **not a
  clean null**.

**Silent abstention (scenario 3a, `lm_head` included):**
`test_lm_head_truncation_abstains_silently_although_a_treatment_exists`.

- Attention has 64 treated rows of 256.
- `lm_head` has 8 rows (one per step), so every series is cut to 8. The rows
  kept are attention's step 0, layers 0-7, none treated.
- `attribute_dr` returns no hypotheses. The same data without `lm_head` returns
  four, so the empty result is a failure to estimate, not an absence of anything
  estimable.
- **This is what should produce a `[PENDING-STATUS: not-identifiable]`-style
  status, with a reason like "treatment present in the data but absent from the
  rows kept after truncation". Today it fails silently.**

**The empty result is ambiguous:**
`test_no_anomaly_no_controls_and_truncation_return_the_same_empty_result`. No
anomaly, every row treated, and truncation all return
`RankedHypotheses(hypotheses=[])`, whose only field is `hypotheses`.

These tests pin current behaviour. They are meant to fail when dr.py reports why
it abstains (the third test), stops trusting a near-zero se (the first), or stops
truncating to the shortest series (the second), and then to be rewritten against
the new contract.

---

## 11. Summary of decisions

| category | dr.py today | decision | precedent in repo | needs |
|---|---|---|---|---|
| Causal graphs | every ordered pair; graph ignored (E7); direction-blind (E1) | reject current; adopt pair-and-layer reasoning | `mechanism_model.md` P1-P6 | the pair table as data **[PENDING: expanded mechanism list]** |
| Adjustment | row position only (`dr.py:123`) | reject current; adopt in principle | none | code: carry trace fields through `KernelResidual`; telemetry: step, batch, kv_len, clock |
| Negative controls | none | reject current; adopt | `separation.CONTROL_OP`, control gate | a control op present in the graph, or a raw-kernel path |
| Placebo / permutation | normal approximation (`dr.py:148`); inflated under autocorrelation (E5) | reject current; adopt candidate | none | calibration on probe scenarios 5 and 6 |
| Sensitivity | none | reject for dr.py | `separation.EPS_MAX_S` | a chosen sensitivity parameter |
| Partial identification | point estimate only | adopt pattern | P1's identified slope; `decide`'s straddle → inconclusive | the `identified` field |
| Explicit abstention | silent (E3); does not abstain on near-zero se (E6) | reject current; adopt | `separation.Decision` outcome + reasons + `REMEDY` | **[PENDING: Andrew's v0 status names]** |

---

## 12. Judgment calls to revisit when Andrew's code lands

1. **The third status term.** *Absent* is not used because this draft does not
   know whether it means "mechanism estimated absent" or "required signal absent
   from the data". The "no treated rows" row in §9 is the case most likely to
   need it.
2. **Two placeholders, not three.** §0.1 defines `estimable` and
   `not-identifiable` placeholders only. If Andrew's scheme splits either, §2's
   `status` type and the tables in §§3-9 that name a placeholder need updating;
   nothing else does.
3. **Pairwise mechanism structure.** §3 assumes the expanded mechanism list keeps
   `mechanism_model.md`'s "confounded at layer / separated by" form.
4. **The worked example's mechanism label.** §10 uses the fixture class
   `RegionSlowdown` (code). It corresponds to a mechanism-model ID today; that
   label may change with the expansion, and the example does not depend on it.
5. **Sensitivity parameterisation.** §7 leaves open which parameter to bound (a
   confounder's association with treatment and outcome, a Rosenbaum-style odds
   ratio, or a bound on a named driver such as clock).
6. **Circular-shift permutation.** §6's expectation that an i.i.d. shuffle stays
   anticonservative under AR(1) and a circular shift does not is untested.
7. **Batch from grid.** §4 treats the decode kernel's grid as a batch proxy on
   the strength of `separation.py:172-174`. Whether that holds outside
   attention, or under CUDA graphs, is unverified.
8. **What separates α from η.** `mechanism_model.md` states that P1 does not; it
   does not say what would. Ask Andrew before §8 proposes anything.

## 13. Not covered

Granger attribution (`attribution.py`) except where it shares dr.py's series
construction; scheduler and collective causes (audit Q3.3-Q3.4); multi-rank
mechanisms; any GPU measurement. Nothing in this document has run on a GPU.
