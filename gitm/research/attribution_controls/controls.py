"""Control cases with a planted cause, and the default suite.

A :class:`ControlCase` keeps two things apart:

* ``build()`` returns the :class:`EstimatorInput`: residuals, graph and trace,
  exactly what a capture gives. It is the only thing an estimator sees.
* ``truth`` is a frozen :class:`Truth`: what was planted, and the *group* of
  causes the capture cannot tell from it. The judge reads it; no estimator does.

The group is not taken on trust from ``docs/mechanism_model.md``. Each member
has a *witness*: a scenario that plants that member instead. A test regenerates
every witness and checks its trace is identical to the case's, so a member is
in the group only if the fixtures show it is indistinguishable. Three confounds
qualify (all at one operating point, with no ``off`` arm in the input):

* P1: ``M1(R, α)`` and ``M3k(R, δ = α·d)`` on the same op.
* P3: ``M4`` gating an inner mechanism, and that mechanism on the gated steps.
* P4: ``M3k`` on every kernel and ``M5e`` tracer overhead of the same size.

So even a "positive" case's truth is a group over mechanism classes: the op is
determined, the class is not. P2 (``M2`` vs ``M3g``) and P6 (``M5m`` vs ``M1``)
are not used as groups: their raw traces differ (exposed side-stream time, and
the per-step kernel count), and the trace is part of the input.

Fixture cases are noise-free (``noise_cv=0``) except the nulls, because P1's
equivalence is exact only without noise: with noise, ``M1`` scales it and
``M3k`` does not. The series cases reproduce ``gitm/research/dr_probe.py``'s
generators exactly (a test checks this) without importing it, since the probe
imports dr.py.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

import numpy as np

from gitm.optimizer import mechanism_fixtures as mf
from gitm.optimizer.monitor import KernelResidual, Residuals, residuals
from gitm.planner.graph import Graph
from gitm.planner.roofline import BatchConfig
from gitm.research.attribution_controls.verdict import Verdict
from gitm.research.attribution_controls.vocabulary import M1, M3K, M4, M5E
from gitm.tracer.schema import Trace

POSITIVE, NULL, AMBIGUOUS, HOSTILE = "positive", "null", "ambiguous", "hostile"
KINDS = (POSITIVE, NULL, AMBIGUOUS, HOSTILE)

#: The probe's operating point and step count, so its scenarios reproduce here.
POINT = BatchConfig(batch=8, kv_cache_len=2048)
N_STEPS = 8
#: Layers a region case touches. lm_head runs once per step, so dr.py truncates
#: every series to N_STEPS rows: step 0, layers 0-7. Layers 0-3 leave both
#: treated and control rows there; a case meant to defeat that uses other steps.
LAYERS = frozenset(range(4))
NULL_NOISE_CV = 0.02
NULL_SEEDS = (0, 1, 2)
#: dr_probe scenario 4 (dr.py's own test fixture).
DIRECTION_SEED = 3
#: dr_probe scenario 5. Under attribute_dr: seed 45 gives p = 2.8e-5, seed 0 p = 0.57.
SPIKE_SEEDS = (0, 45)
#: dr_probe scenario 6b (40 rows). Under attribute_dr: seed 2 gives A->B p = 6e-4,
#: seed 10 B->A p = 0.0, seed 0 A->B p = 0.27.
AR1_SEEDS = (0, 2, 10)
SERIES_ROWS = 40

ATTN, QKV, MLP_DOWN = "attn_score_value", "qkv_proj", "mlp_down"


@dataclass(frozen=True, slots=True)
class EstimatorInput:
    """What a capture gives. Nothing about what was planted; ``slots`` stops one being attached."""

    residuals: Residuals
    #: ``None`` for the series cases, which have no model graph.
    graph: Graph | None
    #: ``None`` for the series cases, which are residual rows only.
    trace: Trace | None


class Estimator(Protocol):
    def __call__(self, inp: EstimatorInput, /) -> Verdict: ...


@dataclass(frozen=True, slots=True)
class Cause:
    """A member of a truth group. ``op`` is ``None`` for a cause on no particular op;
    ``mechanism`` is ``None`` for a planted cause outside the mechanism model."""

    op: str | None
    mechanism: str | None

    def describe(self) -> str:
        return f"{self.op or '*'}/{self.mechanism or '-'}"


@dataclass(frozen=True, slots=True)
class Planted:
    mechanism: str | None
    ops: tuple[str, ...] | None
    layers: tuple[int, ...] | None
    steps: tuple[int, ...] | None
    params: tuple[tuple[str, float], ...] = ()

    def describe(self) -> str:
        parts = [self.mechanism or "planted link"]
        for name, v in (("ops", self.ops), ("layers", self.layers), ("steps", self.steps)):
            if v is not None:
                parts.append(f"{name}={_span(v)}")
        parts += [f"{k}={v:g}" for k, v in self.params]
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class Truth:
    #: Empty means nothing was planted.
    planted: tuple[Planted, ...]
    #: Every cause the capture cannot tell from what was planted. Empty iff nothing planted.
    group: frozenset[Cause]

    def __post_init__(self) -> None:
        if bool(self.planted) != bool(self.group):
            raise ValueError("a truth group must be empty exactly when nothing is planted")

    @property
    def nothing_planted(self) -> bool:
        return not self.planted

    def describe(self) -> str:
        if self.nothing_planted:
            return "nothing planted"
        group = sorted(c.describe() for c in self.group)
        return group[0] if len(group) == 1 else "{" + ", ".join(group) + "}"


@dataclass(frozen=True)
class ControlCase:
    name: str
    kind: str
    seed: int
    #: Where the scenario comes from (a doc proposition or a probe scenario).
    source: str
    build: Callable[[], EstimatorInput] = field(repr=False, compare=False)
    truth: Truth = field(compare=False)
    #: One trace builder per group member, each planting that member instead.
    witnesses: tuple[tuple[Cause, Callable[[], Trace]], ...] = field(
        default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} is not one of {KINDS}")


@dataclass(frozen=True)
class Suite:
    cases: tuple[ControlCase, ...]

    def __post_init__(self) -> None:
        names = [c.name for c in self.cases]
        if len(set(names)) != len(names):
            raise ValueError("control case names must be unique")

    def of_kind(self, kind: str) -> tuple[ControlCase, ...]:
        return tuple(c for c in self.cases if c.kind == kind)

    def seeds(self) -> tuple[tuple[str, int], ...]:
        return tuple((c.name, c.seed) for c in self.cases)


def _span(v: tuple[int | str, ...]) -> str:
    if all(isinstance(x, int) for x in v) and len(v) > 2 and list(v) == list(range(v[0], v[-1] + 1)):
        return f"{v[0]}..{v[-1]}"
    return ",".join(str(x) for x in v)


# ── fixture cases ───────────────────────────────────────────────────────────


def _scenario(mechanisms=(), observation=(), *, step_z=None, noise_cv=0.0, seed=0) -> mf.Scenario:
    return mf.Scenario(points=(POINT,), mechanisms=tuple(mechanisms),
                       observation=tuple(observation), n_steps=N_STEPS, step_z=step_z,
                       noise_cv=noise_cv, seed=seed)


def _fixture_input(scn: mf.Scenario) -> EstimatorInput:
    fx = mf.generate(scn)[0]
    return EstimatorInput(residuals=residuals(fx.trace, fx.graph), graph=fx.graph, trace=fx.trace)


def _fixture_trace(scn: mf.Scenario) -> Trace:
    return mf.generate(scn)[0].trace


def op_ns(op: str) -> float:
    """True baseline duration of ``op`` at :data:`POINT` (every layer has the same shape)."""
    scn = _scenario()
    graph = mf.generate(scn)[0].graph
    node = next(n for n in graph.nodes if n.op == op)
    return scn.truth.duration_ns(node, scn.hw, POINT)


def _gate_z(steps: frozenset[int] | None) -> tuple[float, ...]:
    return tuple(1.0 if steps is None or s in steps else 0.0 for s in range(N_STEPS))


def region_witnesses(op: str, alpha: float, layers: frozenset[int] | None,
                     steps: frozenset[int] | None) -> dict[Cause, mf.Scenario]:
    """P1 and P3: a slowdown, the additive cost of the same size, and a gate on exactly those steps."""
    delta = alpha * op_ns(op)
    return {
        Cause(op, M1): _scenario([mf.RegionSlowdown(alpha, ops={op}, layers=layers, steps=steps)]),
        Cause(op, M3K): _scenario([mf.AdditiveCost(delta, ops={op}, layers=layers, steps=steps)]),
        Cause(op, M4): _scenario([mf.RegimeGate(mf.RegionSlowdown(alpha, ops={op}, layers=layers),
                                                0.5)], step_z=_gate_z(steps)),
    }


def _fixture_case(name: str, kind: str, source: str, planted: Planted, scn: mf.Scenario,
                  witnesses: Mapping[Cause, mf.Scenario]) -> ControlCase:
    return ControlCase(
        name=name, kind=kind, seed=scn.seed, source=source,
        build=partial(_fixture_input, scn),
        truth=Truth((planted,), frozenset(witnesses)),
        witnesses=tuple((c, partial(_fixture_trace, s)) for c, s in witnesses.items()),
    )


def _tuple(v) -> tuple | None:
    return None if v is None else tuple(sorted(v))


def _region_case(name: str, kind: str, source: str, op: str, alpha: float,
                 layers: frozenset[int] | None, steps: frozenset[int] | None,
                 mechanism: str = M1) -> ControlCase:
    """Plant ``mechanism`` (M1 or M3k) on ``op``; its group is :func:`region_witnesses`."""
    w = region_witnesses(op, alpha, layers, steps)
    if mechanism == M1:
        params = (("alpha", alpha),)
    else:
        params = (("delta_ns", alpha * op_ns(op)), ("equivalent_alpha", alpha))
    planted = Planted(mechanism, (op,), _tuple(layers), _tuple(steps), params)
    return _fixture_case(name, kind, source, planted, w[Cause(op, mechanism)], w)


def _gate_vs_onset_case() -> ControlCase:
    """P3b: a gate on a condition that only rises is an onset in time."""
    z = tuple(float(s) for s in range(N_STEPS))
    threshold = 3.5
    gated = frozenset(s for s, v in enumerate(z) if v > threshold)
    gate = _scenario([mf.RegimeGate(mf.RegionSlowdown(1.0, ops={ATTN}, layers=LAYERS), threshold)],
                     step_z=z)
    w = region_witnesses(ATTN, 1.0, LAYERS, gated)
    w[Cause(ATTN, M4)] = gate
    planted = Planted(M4, (ATTN,), _tuple(LAYERS), _tuple(gated),
                      (("alpha", 1.0), ("z_threshold", threshold)))
    return _fixture_case("amb_gate_vs_onset", AMBIGUOUS,
                         "mechanism_model.md P3b (monotone gate = onset), with P1", planted, gate, w)


def _cost_vs_tracer_case() -> ControlCase:
    """P4: a real per-kernel cost and tracer overhead of the same size give the same traced run."""
    eps = 2_000.0
    w = {
        Cause(None, M5E): _scenario(observation=[mf.TracerOverhead(eps)]),
        Cause(None, M3K): _scenario([mf.AdditiveCost(eps)]),
        Cause(None, M4): _scenario([mf.RegimeGate(mf.AdditiveCost(eps), 0.5)],
                                   step_z=_gate_z(None)),
    }
    planted = Planted(M5E, None, None, None, (("eps_ns", eps),))
    return _fixture_case("amb_cost_vs_tracer", AMBIGUOUS,
                         "mechanism_model.md P4 (no off arm in the input), with P3a",
                         planted, w[Cause(None, M5E)], w)


def _null_case(seed: int) -> ControlCase:
    scn = _scenario(noise_cv=NULL_NOISE_CV, seed=seed)
    return ControlCase(f"null_healthy_seed{seed}", NULL, seed,
                       f"healthy run, eta 0.8, noise_cv {NULL_NOISE_CV}",
                       build=partial(_fixture_input, scn), truth=Truth((), frozenset()))


# ── series cases (dr_probe scenarios 4, 5 and 6b) ───────────────────────────


def series_residuals(series: Mapping[str, np.ndarray]) -> Residuals:
    """Interleave rows step by step, as a trace would emit them (dr_probe ``_residuals``)."""
    res = Residuals()
    n = max(len(v) for v in series.values())
    for i in range(n):
        for op, v in series.items():
            if i < len(v):
                res.per_kernel.append(KernelResidual(op=op, layer=None, r_kt=float(v[i]), r_mt=None))
    return res


def direction_series(seed: int) -> dict[str, np.ndarray]:
    """A spikes every 5th row; B = 0.8 A + noise. A causes B."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.05, SERIES_ROWS)
    a[::5] = 1.0
    b = 0.8 * a + rng.normal(0, 0.05, SERIES_ROWS)
    return {"A": a, "B": b}


def spike_series(seed: int) -> dict[str, np.ndarray]:
    """A spikes at 8 random rows; B is independent noise."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.05, SERIES_ROWS)
    a[rng.choice(SERIES_ROWS, 8, replace=False)] = 1.0
    b = rng.normal(0, 0.05, SERIES_ROWS)
    return {"A": a, "B": b}


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.9, sd: float = 0.3) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def ar1_series(seed: int) -> dict[str, np.ndarray]:
    """Two independent AR(1) series, phi 0.9 (A drawn first)."""
    rng = np.random.default_rng(seed)
    a = _ar1(rng, SERIES_ROWS)
    b = _ar1(rng, SERIES_ROWS)
    return {"A": a, "B": b}


def _series_input(make: Callable[[int], dict[str, np.ndarray]], seed: int) -> EstimatorInput:
    return EstimatorInput(residuals=series_residuals(make(seed)), graph=None, trace=None)


def _series_case(name: str, kind: str, source: str, make, seed: int, truth: Truth) -> ControlCase:
    return ControlCase(name, kind, seed, source, build=partial(_series_input, make, seed),
                       truth=truth)


# ── the suite ───────────────────────────────────────────────────────────────


def default_suite() -> Suite:
    every = None
    nothing = Truth((), frozenset())
    cases = [
        _region_case("pos_attn_strong", POSITIVE, "M1 on attention, layers 0-3, every step",
                     ATTN, 1.0, LAYERS, every),
        _region_case("pos_attn_weak", POSITIVE, "M1 on attention just above the r_kt band",
                     ATTN, 0.15, LAYERS, every),
        _region_case("pos_mlp_down_steps0to3", POSITIVE, "M1 on mlp_down, layers 0-3, steps 0-3",
                     MLP_DOWN, 1.0, LAYERS, frozenset(range(4))),
        _region_case("pos_qkv_additive", POSITIVE, "M3k on qkv_proj, layers 0-3, every step",
                     QKV, 1.0, LAYERS, every, mechanism=M3K),
        _series_case(f"pos_series_direction_seed{DIRECTION_SEED}", POSITIVE,
                     "dr_probe scenario 4 (dr.py's own test fixture): A causes B",
                     direction_series, DIRECTION_SEED,
                     Truth((Planted(None, ("A",), None, None, (("b_on_a", 0.8),)),),
                           frozenset({Cause("A", None)}))),
        *(_null_case(s) for s in NULL_SEEDS),
        _gate_vs_onset_case(),
        _cost_vs_tracer_case(),
        _region_case("hostile_lm_head_truncation", HOSTILE,
                     "dr_probe scenario 3a: M1 on attention at steps 2-3 only",
                     ATTN, 1.0, None, frozenset({2, 3})),
        _region_case("hostile_every_row_treated", HOSTILE,
                     "dr_probe scenario 2: M1 on every attention row",
                     ATTN, 0.5, None, every),
        *(_series_case(f"hostile_independent_spikes_seed{s}", HOSTILE,
                       "dr_probe scenario 5: A spikes, B independent", spike_series, s, nothing)
          for s in SPIKE_SEEDS),
        *(_series_case(f"hostile_independent_ar1_seed{s}", HOSTILE,
                       "dr_probe scenario 6b: independent AR(1), 40 rows", ar1_series, s, nothing)
          for s in AR1_SEEDS),
    ]
    return Suite(tuple(cases))
