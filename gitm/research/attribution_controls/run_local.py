"""Run the known-cause control suite on the test doubles and on dr.py.

    python -m gitm.research.attribution_controls.run_local [--threshold 0.95] [--random-seed 0]

Deterministic: every case is seeded (the seeds are printed), and nothing reads
a clock or an unseeded generator.
"""

from __future__ import annotations

import argparse

from gitm.research.attribution_controls.adapters import dr_estimator
from gitm.research.attribution_controls.controls import default_suite
from gitm.research.attribution_controls.doubles import (
    always_abstain,
    always_claim,
    oracle,
    random_claim,
)
from gitm.research.attribution_controls.gate import validate
from gitm.research.attribution_controls.judge import OUTCOMES, JudgeConfig

BANNER = ("CPU-ONLY SYNTHETIC SUITE. Planted causes in mechanism_fixtures traces and "
          "dr_probe series.\nIt says nothing about real GPU traces.")
#: For this demonstration run only. Choosing the threshold is the team's decision.
DEMO_THRESHOLD = 0.95
DEMO_RANDOM_SEED = 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gitm.research.attribution_controls.run_local")
    ap.add_argument("--threshold", type=float, default=DEMO_THRESHOLD)
    ap.add_argument("--random-seed", type=int, default=DEMO_RANDOM_SEED)
    args = ap.parse_args(argv)

    suite = default_suite()
    config = JudgeConfig(confidence_threshold=args.threshold)
    print(BANNER)
    print(f"\nconfidence threshold {config.confidence_threshold:g} "
          "(chosen for this run, not a team decision)")
    print(f"random_claim seed {args.random_seed}")
    print("case seeds: " + ", ".join(f"{n}={s}" for n, s in suite.seeds()))
    print("\ncases:")
    for c in suite.cases:
        planted = "; ".join(p.describe() for p in c.truth.planted) or "nothing planted"
        print(f"  {c.name:<36} {c.kind:<9} planted: {planted}")
        print(f"  {'':<36} {'':<9} truth group: {c.truth.describe()}   [{c.source}]")

    estimators = [oracle(suite), always_abstain(), always_claim("attn_score_value"),
                  random_claim(args.random_seed), dr_estimator]
    results = []
    for est in estimators:
        v = validate(est, suite, config)
        results.append(v)
        print(f"\n{'=' * 78}\n{v.report.estimator}\n{'=' * 78}")
        print(v.report.table())
        print("counts: " + ", ".join(f"{o} {v.report.count(o)}" for o in OUTCOMES))
        print(v.explain())

    print(f"\n{'=' * 78}\nvalidity and usefulness, side by side\n{'=' * 78}")
    for v in results:
        print(v.summary())
    print(f"\n{BANNER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
