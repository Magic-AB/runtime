"""Pre-experiment features of one operating point (design doc §1-§4).

:func:`extract_features` is a pure function of a :class:`PointDescription`. The
description has a fixed set of slots, each one an input the design doc allows
(§1). None of them is a measured candidate outcome, a history record or a
speedup, and because the class is frozen and slotted, one cannot be added:
an unknown keyword is a ``TypeError`` at construction and an unknown attribute
is an ``AttributeError`` afterwards.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.bound_classes import MEMORY_BOUND, normalize_bound
from gitm.optimizer.deviation import observed_op, predicted_per_op
from gitm.optimizer.monitor import residuals
from gitm.optimizer.replay import predict_delta
from gitm.optimizer.vllm_knobs import (
    expand_relative_candidates,
    get_knob,
    knob_kind,
    unmet_prerequisite,
)
from gitm.planner.context import PlannerContext
from gitm.planner.graph import Graph
from gitm.planner.roofline import BatchConfig
from gitm.playbook.match import AXIS_METRICS
from gitm.research.distance_metric.baselines import catalog_predict_delta_ranking
from gitm.research.distance_metric.config import DistanceConfig
from gitm.scheduler.loop import _batch_config_from_stats
from gitm.tracer.schema import Trace
from gitm.traffic.regime import Regime

#: Bucket for kernels that classify to no op, as in ``deviation.py``.
UNMODELED = "<unmodeled>"
#: Bucket for predicted nodes whose roofline bound maps to no bottleneck class.
NO_BOUND = "<no_bound>"

BASIS_CAPTURED = "captured"
BASIS_NONE = "none"

#: Config fields that change what :func:`extract_features` computes. Features
#: record them, and :func:`check_extracted_with` refuses a config that differs.
EXTRACTION_FIELDS = ("skip_high_risk", "require_qualification_commit", "kset_base_knobs")


def extraction_key(config: DistanceConfig) -> tuple[tuple[str, Hashable], ...]:
    return tuple((f, getattr(config, f)) for f in EXTRACTION_FIELDS)


def check_extracted_with(config: DistanceConfig, *features: PointFeatures) -> None:
    key = extraction_key(config)
    for f in features:
        if f.extracted_with != key:
            raise ValueError(
                f"features of {f.name!r} were extracted under {dict(f.extracted_with)}, "
                f"but this config has {dict(key)}; re-extract them with this config"
            )


@dataclass(frozen=True, slots=True)
class PointDescription:
    """What is known about one operating point before any candidate runs.

    ``engine`` is the untouched baseline engine. Only named knobs are read off
    it (through :func:`~gitm.optimizer.vllm_knobs.get_knob`), plus whether it
    carries the loop's ``gitm_restart_fn`` hook. ``baseline_trace`` and
    ``scheduler_stats`` come from a run with no candidate applied, which the
    design doc counts as pre-experiment (§1). ``graph_at`` builds the predicted
    graph at a bridged :class:`BatchConfig`; ``planner_version`` names it so a
    pair can require the same planner on both sides (§7.1).
    """

    name: str
    model: str
    model_revision: str
    engine_name: str
    engine_version: str
    planner: PlannerContext
    regime: Regime
    engine: Any
    library: tuple[InterventionSpec, ...]
    graph_at: Callable[[BatchConfig], Graph]
    planner_version: str
    baseline_trace: Trace | None = None
    scheduler_stats: Any | None = None
    #: Free text naming where ``baseline_trace`` came from, e.g. a synthetic
    #: fixture rather than a GPU capture. Reported, never compared.
    capture_source: str | None = None


@dataclass(frozen=True)
class KnobRead:
    readable: bool
    value: Hashable = None


@dataclass(frozen=True)
class CandidateFeature:
    """One expanded candidate at one point. ``coverage`` is
    ``predict_delta(trace, spec, delta_mean=1.0)``: the fraction of device time
    in kernels the candidate applies to. ``rank`` is its position among the
    point's runnable candidates in baseline 3's ranking."""

    name: str
    knob_values: tuple[tuple[str, Hashable], ...]
    coverage: float | None
    runnable: bool
    reason: str | None = None
    rank: int | None = None


@dataclass(frozen=True)
class ResidualProfile:
    serialized_concurrency_fraction: float
    memory_bound_fraction: float
    observed_op_shares: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PointFeatures:
    name: str
    # Gated (§4): equal or the pair is ineligible.
    model: str
    model_revision: str
    sku: str | None
    num_gpus: int
    tensor_parallel_size: Hashable
    engine_name: str
    engine_version: str
    dtype: str | None
    workload: str
    planner_version: str
    # Bridge (§2).
    batch: int | None
    kv_cache_len: int | None
    batch_source: str | None
    not_bridgeable_reason: str | None
    # Components.
    traffic: tuple[tuple[str, float], ...]
    knobs: tuple[tuple[str, KnobRead], ...]
    op_shares: tuple[tuple[str, float], ...] | None
    bound_shares: tuple[tuple[str, float], ...] | None
    candidates: tuple[CandidateFeature, ...]
    #: Runnable candidates in baseline 3's order. The cap is applied later, by
    #: :meth:`capped`, so features do not depend on it.
    ranked: tuple[str, ...]
    basis: str
    residual: ResidualProfile | None
    #: The config fields these features were extracted under (see
    #: :data:`EXTRACTION_FIELDS`).
    extracted_with: tuple[tuple[str, Hashable], ...] = ()
    capture_source: str | None = None

    @property
    def bridgeable(self) -> bool:
        return self.not_bridgeable_reason is None

    def capped(self, config: DistanceConfig) -> tuple[str, ...]:
        """This point's fixed measurement set (§7.3): the first
        ``candidate_cap`` runnable candidates of its own baseline-3 ranking."""
        return self.ranked[: config.candidate_cap]

    def candidate_map(self) -> dict[str, CandidateFeature]:
        return {c.name: c for c in self.candidates}

    def knob_map(self) -> dict[str, KnobRead]:
        return dict(self.knobs)


def _freeze(value: Any) -> Hashable:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, list | tuple | set | frozenset):
        items = [_freeze(v) for v in value]
        return tuple(sorted(items, key=repr) if isinstance(value, set | frozenset) else items)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _shares(totals: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    total = sum(v for v in totals.values() if v > 0)
    if total <= 0:
        return ()
    return tuple(sorted((k, max(v, 0.0) / total) for k, v in totals.items()))


def _read_knob(engine: Any, knob: str) -> KnobRead:
    try:
        return KnobRead(True, _freeze(get_knob(engine, knob)))
    except AttributeError:
        return KnobRead(False)


def bridge(point: PointDescription) -> tuple[int | None, int | None, str | None, str | None]:
    """``(batch, kv_cache_len, batch_source, not_bridgeable_reason)`` per §2."""
    regime = point.regime
    kv = regime.input_p50 + regime.output_p50 // 2
    measured = _batch_config_from_stats(point.scheduler_stats)
    if measured is not None:
        return measured.batch, kv, "mean_running", None
    if regime.concurrency is None:
        return None, None, None, "open_loop_unmeasured"
    seqs = _read_knob(point.engine, "max_num_seqs")
    if not seqs.readable or not isinstance(seqs.value, int) or isinstance(seqs.value, bool):
        return None, None, None, "max_num_seqs_unreadable"
    return min(regime.concurrency, seqs.value), kv, "min(concurrency, max_num_seqs)", None


def _expand(point: PointDescription) -> list[InterventionSpec]:
    expanded = [c for spec in point.library for c in expand_relative_candidates(spec, point.engine)]
    names = [c.name for c in expanded]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"point {point.name!r}: expanded candidate names collide: {dupes}")
    return expanded


def _veto(point: PointDescription, spec: InterventionSpec) -> str | None:
    """The two vetoes of §3: a structural knob with no restart hook, and an
    unmet prerequisite."""
    has_hook = callable(getattr(point.engine, "gitm_restart_fn", None))
    for knob in spec.knob_values:
        if knob_kind(knob) == "structural" and not has_hook:
            return f"structural knob {knob!r} and no restart hook"
        why = unmet_prerequisite(point.engine, knob)
        if why is not None:
            return why
    return None


def _candidates(
    point: PointDescription, expanded: list[InterventionSpec], config: DistanceConfig
) -> tuple[tuple[CandidateFeature, ...], tuple[str, ...]]:
    trace = point.baseline_trace
    if trace is None:
        feats = tuple(
            CandidateFeature(c.name, _freeze(c.knob_values), None, False, "no_baseline_capture")
            for c in expanded
        )
        return tuple(sorted(feats, key=lambda f: f.name)), ()

    ranked = catalog_predict_delta_ranking(trace, expanded, config.policy(), ctx=point.planner.gate)
    feats: list[CandidateFeature] = []
    runnable_order: list[str] = []
    for rc in ranked:
        spec = rc.spec
        reason = rc.rejected_reason or _veto(point, spec)
        rank = None
        if reason is None:
            rank = len(runnable_order)
            runnable_order.append(spec.name)
        feats.append(CandidateFeature(
            name=spec.name,
            knob_values=_freeze(spec.knob_values),
            coverage=predict_delta(trace, spec, delta_mean=1.0),
            runnable=reason is None,
            reason=reason,
            rank=rank,
        ))
    return tuple(sorted(feats, key=lambda f: f.name)), tuple(runnable_order)


def _residual_profile(trace: Trace, graph: Graph) -> ResidualProfile:
    res = residuals(trace, graph)
    kernels = list(trace.kernels())
    device_s = sum(max(0, k.end_ns - k.start_ns) for k in kernels) / 1e9
    by_op: dict[str, float] = {}
    for k in kernels:
        op = observed_op(k.name, k.range_op) or UNMODELED
        by_op[op] = by_op.get(op, 0.0) + max(0, k.end_ns - k.start_ns)
    memory_s = sum(
        k.t_obs_s or 0.0 for k in res.per_kernel if normalize_bound(k.bound) == MEMORY_BOUND
    )
    return ResidualProfile(
        serialized_concurrency_fraction=float(res.serialized_concurrency_fraction),
        # Per-kernel seconds against summed nanoseconds can round past 1.
        memory_bound_fraction=min(memory_s / device_s, 1.0) if device_s > 0 else 0.0,
        observed_op_shares=_shares(by_op),
    )


def _graph_profile(graph: Graph) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    by_bound: dict[str, float] = {}
    for node in graph.nodes:
        cls = normalize_bound(node.prediction.bound) or NO_BOUND
        by_bound[cls] = by_bound.get(cls, 0.0) + node.prediction.t_pred_s
    return _shares(predicted_per_op(graph)), _shares(by_bound)


def _traffic(regime: Regime) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((axis, float(getattr(regime, axis))) for axis in AXIS_METRICS))


def _knob_names(config: DistanceConfig, expanded: Iterable[InterventionSpec]) -> list[str]:
    names = set(config.kset_base_knobs)
    for c in expanded:
        names.update(c.knob_values)
    return sorted(names)


def extract_features(point: PointDescription, config: DistanceConfig) -> PointFeatures:
    """The point's features, from pre-experiment information only."""
    batch, kv, batch_source, not_bridgeable = bridge(point)
    expanded = _expand(point)
    candidates, ranked = _candidates(point, expanded, config)

    op_shares = bound_shares = residual = None
    if not_bridgeable is None:
        graph = point.graph_at(BatchConfig(batch=batch, kv_cache_len=kv))
        op_shares, bound_shares = _graph_profile(graph)
        if point.baseline_trace is not None:
            residual = _residual_profile(point.baseline_trace, graph)

    tp = _read_knob(point.engine, "tensor_parallel_size")
    return PointFeatures(
        name=point.name,
        model=point.model,
        model_revision=point.model_revision,
        sku=point.planner.sku,
        num_gpus=point.planner.num_gpus,
        tensor_parallel_size=tp.value if tp.readable else None,
        engine_name=point.engine_name,
        engine_version=point.engine_version,
        dtype=point.planner.gate.dtype,
        workload=point.planner.gate.workload,
        planner_version=point.planner_version,
        batch=batch,
        kv_cache_len=kv,
        batch_source=batch_source,
        not_bridgeable_reason=not_bridgeable,
        traffic=_traffic(point.regime),
        knobs=tuple((k, _read_knob(point.engine, k)) for k in _knob_names(config, expanded)),
        op_shares=op_shares,
        bound_shares=bound_shares,
        candidates=candidates,
        ranked=ranked,
        basis=BASIS_CAPTURED if point.baseline_trace is not None else BASIS_NONE,
        residual=residual,
        extracted_with=extraction_key(config),
        capture_source=point.capture_source,
    )
