"""The ranked list the fast loop picks from (ENG-97).

``select_interventions`` already ranks the library for a trace. This module turns that
ranking into the list the loop consumes: it attaches each lever's ``class`` and
``restart_cost`` from ``library.yaml``, drops what the loop must not try, and records
why each lever stands where it does.

Fields read from the yaml, per lever (ENG-79 adds them; until it lands nothing has them):

* ``lever_class``: 1 accuracy-neutral (in the MVP), 2 changes numerics, 3 post-MVP. ENG-79's
  owner gave this name on Oct 23, still to be confirmed with the reviewers; a bare ``class`` is
  accepted too, so either spelling of the field reads;
* ``restart_cost``: ``none`` (live engine API), ``restart`` (new engine, weights reused),
  ``rebuild`` (weights re-sharded);
* ``expected_delta_mean``, ``expected_delta_lo``, ``expected_delta_hi``.

They are read straight from the yaml and not through ``InterventionSpec``, which rejects
unknown keys and is being changed by ENG-79. Once the spec carries the fields this can
read them from there.

Strict by default: a lever with no class, or no restart cost, is left out and listed with
that reason, because ENG-79's rule is that a class is never guessed. ``strict=False``
keeps such levers in, marked unknown, so the list can be inspected before ENG-79 lands.

Two things this does not do:

* It does not read live telemetry. The evidence on each entry is what ``select_interventions``
  already carries (the cause that motivated the lever, the ops it is aimed at and the seconds
  above their floor, whether the delta is a prior or measured). Counters from the collector
  (ENG-83) will reach it through that function's ``motivated`` and ``recoverable`` inputs.
* ``cost_weighted`` ordering (expected delta per second of trying) is an option, off by
  default. The tier seconds are planning units from ENG-79's wording (per second, per
  minute, per ten minutes), and whether the loop should rank by them is a policy decision
  for review, not something this code settles.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gitm.agents.policy import RankedCandidate

__all__ = [
    "LeverMeta", "RankedEntry", "RankedList", "RESTART_SECONDS", "SCHEMA_VERSION",
    "rank_for_loop", "read_lever_meta",
]

SCHEMA_VERSION = 1
CLASSES = (1, 2, 3)
#: Planning units for how long a try takes, from ENG-79: live API, new engine, re-shard.
RESTART_SECONDS = {"none": 1.0, "restart": 60.0, "rebuild": 600.0}


@dataclass(frozen=True)
class LeverMeta:
    name: str
    lever_class: int | None = None
    restart_cost: str | None = None
    expected_delta_mean: float | None = None
    expected_delta_lo: float | None = None
    expected_delta_hi: float | None = None
    #: What is wrong with this lever's fields, empty when they are fine.
    problems: tuple[str, ...] = ()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def read_lever_meta(path: str | Path) -> dict[str, LeverMeta]:
    """Each lever's class, restart cost and expected delta, keyed by name.

    Never raises on a bad field: the problem is recorded on the lever and the lever is left
    out of a strict ranking. Raises ``ValueError`` only when the file is not a library at
    all, or names a lever twice.
    """
    raw = yaml.safe_load(Path(path).read_text())
    items = raw.get("interventions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"{path}: no 'interventions' list")
    out: dict[str, LeverMeta] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"{path}: interventions[{i}] has no name")
        name = item["name"]
        if name in out:
            raise ValueError(f"{path}: lever {name!r} appears twice")
        problems: list[str] = []
        cls = item.get("lever_class", item.get("class"))
        if cls is None:
            problems.append("class_missing")
        elif isinstance(cls, bool) or cls not in CLASSES:
            problems.append(f"class_invalid:{cls!r}")
            cls = None
        rc = item.get("restart_cost")
        if rc is None:
            problems.append("restart_cost_missing")
        elif rc not in RESTART_SECONDS:
            problems.append(f"restart_cost_invalid:{rc!r}")
            rc = None
        out[name] = LeverMeta(
            name=name, lever_class=cls, restart_cost=rc,
            expected_delta_mean=_number(item.get("expected_delta_mean")),
            expected_delta_lo=_number(item.get("expected_delta_lo")),
            expected_delta_hi=_number(item.get("expected_delta_hi")),
            problems=tuple(problems))
    return out


@dataclass(frozen=True)
class RankedEntry:
    rank: int
    name: str
    predicted_delta: float
    delta_source: str
    lever_class: int | None
    restart_cost: str | None
    try_cost_s: float | None
    expected_delta_range: tuple[float, float] | None
    motivated_by: str | None
    targets: tuple[tuple[str, float], ...]
    demoted: bool
    #: One line per reason the lever stands here, built only from what the ranking carries.
    why: tuple[str, ...]


@dataclass
class RankedList:
    entries: list[RankedEntry]
    #: ``(name, reason)`` for each lever left out, in the ranking's order.
    excluded: list[tuple[str, str]]
    strict: bool
    cost_weighted: bool
    mvp_classes: tuple[int, ...]
    assumptions: tuple[str, ...] = field(default=(
        "evidence is what select_interventions carried; no live counters are read here",
        "restart_cost tier seconds are planning units, not measurements",
    ))

    @property
    def top(self) -> list[str]:
        return [e.name for e in self.entries]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def _why(c: RankedCandidate, m: LeverMeta | None, causes: Sequence[Any] = ()) -> tuple[str, ...]:
    lines = [f"predicted delta {c.predicted_delta:+.3f} from {c.delta_source}"]
    if c.motivated_by:
        lines.append(f"this run showed {c.motivated_by}, which argues for it")
        note = next((x.note for x in causes if x.signal == c.motivated_by), None)
        if note:
            lines.append(f"counters: {note}")
    if c.targets:
        ops = ", ".join(f"{op} ({s:.3f}s above floor)" for op, s in c.targets[:3])
        lines.append(f"aimed at ops losing time in this run: {ops}")
    if c.demoted:
        lines.append("demoted: its own record both won and lost")
    if m is not None and m.expected_delta_lo is not None and m.expected_delta_hi is not None:
        lines.append(f"library range {m.expected_delta_lo:+.2f} to {m.expected_delta_hi:+.2f}")
    return tuple(lines)


def rank_for_loop(
    ranked: Sequence[RankedCandidate],
    meta: Mapping[str, LeverMeta],
    *,
    top_n: int | None = None,
    mvp_classes: Iterable[int] = (1,),
    strict: bool = True,
    cost_weighted: bool = False,
    causes: Sequence[Any] = (),
) -> RankedList:
    """The list the loop consumes, from ``select_interventions``'s output and the yaml fields.

    ``ranked`` keeps its order unless ``cost_weighted``. A lever is left out, with the reason,
    when the ranking rejected it, when it has nothing left to change (``baseline_noop``), when
    its class is outside ``mvp_classes``, or, under ``strict``, when its class or restart cost
    is missing or invalid. A lever the yaml does not mention is left out as ``not_in_meta``.
    ``causes`` (objects with ``signal`` and ``note``, such as the scheduler causes) adds each
    motivated lever's cause note, which carries the counters, to its ``why``.
    """
    classes = tuple(mvp_classes)
    kept: list[tuple[RankedCandidate, LeverMeta | None]] = []
    excluded: list[tuple[str, str]] = []
    for c in ranked:
        name = c.spec.name
        m = meta.get(name)
        if c.rejected_reason is not None:
            excluded.append((name, f"rejected:{c.rejected_reason}"))
        elif c.baseline_noop is not None:
            excluded.append((name, f"baseline_noop:{c.baseline_noop}"))
        elif m is None:
            excluded.append((name, "not_in_meta"))
        elif strict and m.problems:
            excluded.append((name, ",".join(m.problems)))
        elif m.lever_class is not None and m.lever_class not in classes:
            excluded.append((name, f"class_{m.lever_class}_outside_mvp"))
        else:
            kept.append((c, m))

    def cost(m: LeverMeta | None) -> float | None:
        return RESTART_SECONDS.get(m.restart_cost) if m and m.restart_cost else None

    if cost_weighted:
        unknown = max(RESTART_SECONDS.values())  # an unknown cost is priced as the worst tier
        kept.sort(key=lambda cm: (
            cm[0].demoted, -(max(cm[0].predicted_delta, 0.0) / (cost(cm[1]) or unknown))))

    entries = []
    for rank, (c, m) in enumerate(kept[:top_n] if top_n else kept, start=1):
        rng = (m.expected_delta_lo, m.expected_delta_hi) if (
            m and m.expected_delta_lo is not None and m.expected_delta_hi is not None) else None
        entries.append(RankedEntry(
            rank=rank, name=c.spec.name, predicted_delta=c.predicted_delta,
            delta_source=c.delta_source, lever_class=m.lever_class if m else None,
            restart_cost=m.restart_cost if m else None, try_cost_s=cost(m),
            expected_delta_range=rng, motivated_by=c.motivated_by, targets=c.targets,
            demoted=c.demoted, why=_why(c, m, causes)))
    return RankedList(entries, excluded, strict, cost_weighted, classes)
