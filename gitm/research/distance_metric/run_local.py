"""Compute and bin every pair of a synthetic grid, with no outcome anywhere.

    python -m gitm.research.distance_metric.run_local [--seed N]

Builds the CPU-only synthetic grid (:mod:`.synthetic`), extracts each point's
pre-experiment features, computes d for every pair, bins the pairs and runs the
pre-measurement bin check. The output is deterministic for a given seed and
says plainly that it is not a validation result.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys

from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.distance import COMPONENTS
from gitm.research.distance_metric.features import extract_features
from gitm.research.distance_metric.pairs import (
    FAR,
    INCONCLUSIVE,
    INELIGIBLE,
    INTERMEDIATE,
    NEAR,
    PairSelectionReport,
    select_pairs,
)
from gitm.research.distance_metric.synthetic import SYNTHETIC_CAPTURE, synthetic_grid

DEFAULT_SEED = 0

BANNER = (
    "NOT A VALIDATION RESULT. No candidate was run and no outcome (A/B delta,\n"
    "history record, speedup) was read or used. Baseline captures are synthetic\n"
    "fixture traces, not GPU measurements. This only shows how the metric bins\n"
    "a grid before any measurement."
)


def _fmt(v: float | None) -> str:
    if v is None:
        return "  -  "
    return "  inf" if math.isinf(v) else f"{v:.3f}"


def render(report: PairSelectionReport, features, seed: int) -> str:
    cfg = report.config
    out = ["distance metric v1: local synthetic run", "", BANNER, "",
           f"seed: {seed}  (fixtures use noise_cv=0, so the seed draws nothing)",
           f"capture source: {SYNTHETIC_CAPTURE}", "", "config:"]
    out += [f"  {f.name} = {getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg)]

    out += ["", "points:",
            f"  {'name':22s} {'batch':>5s} {'kv':>6s}  {'batch source':32s} {'runnable':>8s} {'capped':>6s}"]
    for f in sorted(features, key=lambda f: f.name):
        runnable = sum(c.runnable for c in f.candidates)
        out.append(f"  {f.name:22s} {f.batch!s:>5s} {f.kv_cache_len!s:>6s}  {f.batch_source!s:32s} "
                   f"{runnable:>8d} {len(f.capped(cfg)):>6d}")

    header = "  ".join(f"{c:>5s}" for c in COMPONENTS)
    out += ["", "pairs:",
            f"  {'a':22s} {'b':22s} {'d':>5s}  {'set by':16s}  {header}  {'common':>6s} {'capped':>6s} "
            f"{'same':>4s}  status / reasons"]
    for p in report.pairs:
        d = p.distance
        comps = dict(d.components)
        details = dict(d.details)
        setting = f"{d.setting_component}:{details.get(d.setting_component, '')}" if d.setting_component else "-"
        sizes = p.candidates.sizes if p.candidates else {}
        cells = "  ".join(_fmt(comps.get(c)) for c in COMPONENTS)
        why = f"{p.status}" + (f" ({', '.join(p.reasons)})" if p.reasons else "")
        out.append(f"  {p.a:22s} {p.b:22s} {_fmt(d.distance)}  {setting:16s}  {cells}  "
                   f"{sizes.get('common', 0):>6d} {sizes.get('capped', 0):>6d} {sizes.get('same_values', 0):>4d}  {why}")

    counts = report.counts
    out += ["", "bins:",
            f"  near (d <= {cfg.near_threshold}): {counts[NEAR]}",
            f"  intermediate (unused): {counts[INTERMEDIATE]}",
            f"  far (d >= {cfg.far_threshold}): {counts[FAR]}",
            f"  inconclusive: {counts[INCONCLUSIVE]}",
            f"  ineligible: {counts[INELIGIBLE]}",
            "", "pre-measurement bin check:"]
    for c in report.bin_checks:
        out.append(f"  {c.bin}: {c.n} / {c.required} required -> {c.status}"
                   + (f" ({c.reason})" if c.reason else ""))
    out.append(f"  selection status: {report.status}")

    out += ["", "chosen pairs:"]
    out += [f"  {p.status:4s} {p.a} / {p.b}  d={_fmt(p.distance.distance)}" for p in report.chosen] or ["  none"]
    out += ["", "excluded pairs:"]
    out += [f"  {p.a} / {p.b}: {p.status} ({', '.join(p.reasons)})" for p in report.excluded] or ["  none"]
    out += ["", BANNER]
    return "\n".join(out) + "\n"


def run(seed: int = DEFAULT_SEED, config: DistanceConfig | None = None) -> str:
    config = config or DistanceConfig()
    features = [extract_features(p, config) for p in synthetic_grid(config, seed=seed)]
    return render(select_pairs(features, config), features, seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"seed passed to the fixture generator (default {DEFAULT_SEED})")
    args = ap.parse_args(argv)
    sys.stdout.write(run(seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
