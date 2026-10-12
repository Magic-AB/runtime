"""d(s, t): gates, then the L-inf of seven components (design doc §4).

Every component is in [0, 1] and is 0 at equality. ``d`` is never returned as a
bare float: :class:`DistanceResult` carries the value, the component that set
it, every component's value, and why the pair is ineligible when it is.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from gitm.playbook.match import AXIS_METRICS
from gitm.research.distance_metric.candidates import common_candidates
from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.features import (
    BASIS_CAPTURED,
    KnobRead,
    PointFeatures,
    check_extracted_with,
)

#: Component names in the design doc's order. Ties for the maximum go to the
#: first, so the setting component does not depend on argument order.
COMPONENTS = ("O", "T", "K", "G", "B", "S", "R")

OK = "ok"
INELIGIBLE = "ineligible"
INCONCLUSIVE = "inconclusive"

#: (reason, attribute) for each §4 gate, in the doc's order.
GATES: tuple[tuple[str, str], ...] = (
    ("cross_model", "model"),
    ("cross_model_revision", "model_revision"),
    ("cross_sku", "sku"),
    ("cross_num_gpus", "num_gpus"),
    ("cross_tensor_parallel", "tensor_parallel_size"),
    ("cross_engine", "engine_name"),
    ("cross_engine_version", "engine_version"),
    ("cross_dtype", "dtype"),
    ("cross_workload", "workload"),
)


@dataclass(frozen=True)
class DistanceResult:
    a: str
    b: str
    status: str
    #: ``math.inf`` when ineligible; ``None`` when inconclusive.
    distance: float | None
    setting_component: str | None
    components: tuple[tuple[str, float | None], ...]
    #: For each component, which axis, knob or term produced its value.
    details: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...] = ()

    @property
    def reason(self) -> str | None:
        return self.reasons[0] if self.reasons else None

    def component(self, name: str) -> float | None:
        return dict(self.components)[name]


def rel(a: float, b: float) -> float:
    """``1 - min/max``. 0 when equal (including both 0); 1 when either side is
    <= 0 and they differ."""
    if a == b:
        return 0.0
    if a <= 0 or b <= 0:
        return 1.0
    return 1.0 - min(a, b) / max(a, b)


def tv(p: Iterable[tuple[str, float]], q: Iterable[tuple[str, float]]) -> float:
    """Total-variation distance between two share vectors over the union of keys."""
    pd, qd = dict(p), dict(q)
    return 0.5 * sum(abs(pd.get(k, 0.0) - qd.get(k, 0.0)) for k in pd.keys() | qd.keys())


def _argmax(terms: list[tuple[str, float]]) -> tuple[float, str]:
    best_name, best = "", 0.0
    for name, value in terms:
        if value > best:
            best_name, best = name, value
    return best, best_name


def _is_number(v: Hashable) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _knob_term(ra: KnobRead | None, rb: KnobRead | None) -> float | None:
    a_ok = ra is not None and ra.readable
    b_ok = rb is not None and rb.readable
    if not a_ok and not b_ok:
        return None
    if a_ok != b_ok:
        return 1.0
    if ra.value == rb.value:
        return 0.0
    if _is_number(ra.value) and _is_number(rb.value):
        return rel(float(ra.value), float(rb.value))
    return 1.0


def _component_o(a: PointFeatures, b: PointFeatures) -> tuple[float, str]:
    return _argmax([("batch", rel(a.batch, b.batch)),
                    ("kv_cache_len", rel(a.kv_cache_len, b.kv_cache_len))])


def _component_t(a: PointFeatures, b: PointFeatures, config: DistanceConfig) -> tuple[float, str]:
    ta, tb = dict(a.traffic), dict(b.traffic)
    terms = []
    for axis in config.traffic_axes:
        m = AXIS_METRICS[axis](ta[axis], tb[axis])
        terms.append((axis, 1.0 if math.isinf(m) else 1.0 - 2.0 ** (-m)))
    return _argmax(terms)


def kset(a: PointFeatures, b: PointFeatures, common: Iterable[str], config: DistanceConfig) -> tuple[str, ...]:
    names = set(config.kset_base_knobs)
    ca, cb = a.candidate_map(), b.candidate_map()
    for n in common:
        names.update(k for k, _ in ca[n].knob_values)
        names.update(k for k, _ in cb[n].knob_values)
    return tuple(sorted(names))


def _component_k(a: PointFeatures, b: PointFeatures, knobs: Iterable[str]) -> tuple[float, str]:
    ka, kb = a.knob_map(), b.knob_map()
    terms = []
    for k in knobs:
        v = _knob_term(ka.get(k), kb.get(k))
        if v is not None:
            terms.append((k, v))
    return _argmax(terms)


def _component_s(a: PointFeatures, b: PointFeatures, common: tuple[str, ...]) -> tuple[float, str]:
    ca, cb = a.candidate_map(), b.candidate_map()
    diffs = [(n, abs(ca[n].coverage - cb[n].coverage)) for n in common]
    _, worst = _argmax(diffs)
    return sum(d for _, d in diffs) / len(diffs), worst


def _component_r(a: PointFeatures, b: PointFeatures) -> tuple[float, str]:
    ra, rb = a.residual, b.residual
    return _argmax([
        ("serialized_concurrency_fraction",
         abs(ra.serialized_concurrency_fraction - rb.serialized_concurrency_fraction)),
        ("memory_bound_fraction", abs(ra.memory_bound_fraction - rb.memory_bound_fraction)),
        ("observed_op_shares", tv(ra.observed_op_shares, rb.observed_op_shares)),
    ])


def eligibility_reasons(a: PointFeatures, b: PointFeatures) -> tuple[str, ...]:
    """Every reason the pair is ineligible, gates first. Empty when eligible."""
    reasons = [reason for reason, attr in GATES if getattr(a, attr) != getattr(b, attr)]
    for p in (a, b):
        if not p.bridgeable:
            reasons.append(f"not_bridgeable:{p.name}:{p.not_bridgeable_reason}")
    for p in (a, b):
        if p.basis != BASIS_CAPTURED:
            reasons.append(f"no_baseline_capture:{p.name}")
    if a.planner_version != b.planner_version:
        reasons.append("cross_planner_version")
    return tuple(reasons)


def distance(a: PointFeatures, b: PointFeatures, config: DistanceConfig) -> DistanceResult:
    """d(a, b) per §4. Every component is symmetric, so d(a, b) == d(b, a)."""
    check_extracted_with(config, a, b)
    reasons = eligibility_reasons(a, b)
    if reasons:
        return DistanceResult(a.name, b.name, INELIGIBLE, math.inf, None, (), (), reasons)

    common = common_candidates(a, b, config).common
    values: dict[str, tuple[float, str] | None] = {
        "O": _component_o(a, b),
        "T": _component_t(a, b, config),
        "K": _component_k(a, b, kset(a, b, common, config)),
        "G": (tv(a.op_shares, b.op_shares), ""),
        "B": (tv(a.bound_shares, b.bound_shares), ""),
        "S": _component_s(a, b, common) if common else None,
        "R": _component_r(a, b),
    }
    components = tuple((c, None if values[c] is None else values[c][0]) for c in COMPONENTS)
    details = tuple((c, values[c][1]) for c in COMPONENTS if values[c] is not None and values[c][1])
    if values["S"] is None:
        return DistanceResult(a.name, b.name, INCONCLUSIVE, None, None, components, details,
                              ("no_common_candidates",))
    d, setting = _argmax([(c, v) for c, v in components])
    return DistanceResult(a.name, b.name, OK, d, setting or None, components, details)
