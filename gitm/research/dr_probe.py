"""Reproducible probe of ``gitm/optimizer/dr.py``'s current behaviour.

Read-only: every estimate comes from dr.py's public entry points,
``attribute_dr(residuals, graph, *, band)`` and ``doubly_robust_ate(y, t, X)``.
Nothing in dr.py is patched or reimplemented. ``_view`` repeats attribute_dr's
series bookkeeping (group by op in trace order, keep ops with >= 4 rows,
truncate to the shortest, treat ``|r_kt| > band``) only to print the treated
and control counts it does not return, and cross-checks them against the
``n_treated`` that attribute_dr writes into each hypothesis's notes.

Every random draw goes through ``np.random.default_rng(seed)`` with a seed
named below and printed in the output. There is no global or unseeded
randomness. The mechanism fixtures are deterministic (``noise_cv=0.0``); their
seed is set explicitly anyway.

Run from the repo root:

    .venv/bin/python -m gitm.research.dr_probe              # summary per sweep
    .venv/bin/python -m gitm.research.dr_probe --per-seed   # every seed's row

Results depend on the statsmodels version (the propensity model is a
statsmodels Logit), so the header records it along with dr.py's SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
from math import erfc, sqrt
from pathlib import Path

import numpy as np

from gitm.optimizer import mechanism_fixtures as mf
from gitm.optimizer.dr import attribute_dr, doubly_robust_ate
from gitm.optimizer.monitor import KernelResidual, Residuals
from gitm.planner.graph import predict_graph
from gitm.planner.roofline import BatchConfig

#: Mechanism-fixture seed (``Scenario.seed``). Draws nothing while
#: ``noise_cv == 0.0``, which every fixture here uses.
FIXTURE_SEED = 0
#: Seed of ``tests/test_runtime_on_trace.py::test_attribute_dr_ranks_pairs``,
#: the only test that calls attribute_dr; scenario 4 rebuilds its data exactly.
DR_TEST_SEED = 3
#: Scenario 5: one ``default_rng(seed)`` per run.
SPURIOUS_SEEDS = range(0, 200)
#: The seeds the earlier informal run used (``default_rng(100 + s)``, s < 200).
SPURIOUS_SEEDS_EARLIER = range(100, 300)
#: Scenario 6: one ``default_rng(seed)`` per run.
AR1_SEEDS = range(0, 200)
#: The seeds the earlier informal run used (``default_rng(500 + s)``, s < 200).
AR1_SEEDS_EARLIER = range(500, 700)
#: Scenario 7b: draws the cause series; the effect series draws nothing.
ZERO_SE_SEED = 0

BAND = 0.4  # dr.py's default: the kernel_time invariant's band_width
POINT = BatchConfig(batch=8, kv_cache_len=2048)
N_STEPS = 8
ATTN = "attn_score_value"
ALPHA = 0.05

ROOT = Path(__file__).resolve().parents[2]


# ── reporting helpers ──────────────────────────────────────────────────────


def _header() -> None:
    import statsmodels

    dr_src = (ROOT / "gitm/optimizer/dr.py").read_bytes()
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    print("dr.py probe")
    print(f"  python {platform.python_version()}, numpy {np.__version__}, "
          f"statsmodels {statsmodels.__version__}")
    print(f"  git HEAD {head}; gitm/optimizer/dr.py sha256 {hashlib.sha256(dr_src).hexdigest()[:16]}")
    print(f"  seeds: FIXTURE_SEED={FIXTURE_SEED} DR_TEST_SEED={DR_TEST_SEED} "
          f"SPURIOUS_SEEDS={_span(SPURIOUS_SEEDS)} AR1_SEEDS={_span(AR1_SEEDS)} "
          f"ZERO_SE_SEED={ZERO_SE_SEED}")
    print(f"  earlier-run seeds (reconciliation): SPURIOUS={_span(SPURIOUS_SEEDS_EARLIER)} "
          f"AR1={_span(AR1_SEEDS_EARLIER)}")


def _span(r: range) -> str:
    return f"{r.start}..{r.stop - 1}"


def _title(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def _residuals(series: dict[str, np.ndarray]) -> Residuals:
    """Interleave rows step by step, as a trace would emit them."""
    res = Residuals()
    n = max(len(v) for v in series.values())
    for i in range(n):
        for op, v in series.items():
            if i < len(v):
                res.per_kernel.append(KernelResidual(op=op, layer=None, r_kt=float(v[i]), r_mt=None))
    return res


def _view(res: Residuals) -> dict:
    """What attribute_dr will see. Reporting only: estimates come from dr.py."""
    series: dict[str, list[float]] = {}
    for kr in res.per_kernel:
        series.setdefault(kr.op, []).append(kr.r_kt)
    ops = [op for op, v in series.items() if len(v) >= 4]
    n = min((len(series[op]) for op in ops), default=0)
    per_op = {}
    for op, v in series.items():
        full_t = int(sum(abs(x) > BAND for x in v))
        trunc_t = int(sum(abs(x) > BAND for x in v[:n])) if op in ops else None
        eligible = op in ops and trunc_t >= 3 and (n - trunc_t) >= 3
        per_op[op] = dict(rows=len(v), treated_full=full_t, treated_trunc=trunc_t, eligible=eligible)
    causes = [op for op in ops if per_op[op]["eligible"]]
    pairs = [(c, e) for c in causes for e in ops if e != c]
    return dict(series=series, ops=ops, n=n, per_op=per_op, pairs=pairs)


def _print_view(v: dict) -> None:
    print(f"  ops with >= 4 rows: {len(v['ops'])}; every series truncated to n = {v['n']} rows")
    print(f"  {'op':<18}{'rows':>6}{'treated (full)':>16}{'treated/control (truncated)':>30}  cause?")
    for op, d in v["per_op"].items():
        tc = "-" if d["treated_trunc"] is None else f"{d['treated_trunc']}/{v['n'] - d['treated_trunc']}"
        print(f"  {op:<18}{d['rows']:>6}{d['treated_full']:>16}{tc:>30}  {'yes' if d['eligible'] else 'no'}")
    print(f"  pairs tested: {len(v['pairs'])}" + (f" -> {v['pairs']}" if v["pairs"] else ""))


def _print_hyps(ranked, v: dict | None = None) -> None:
    hs = ranked.hypotheses
    print(f"  attribute_dr returned {len(hs)} hypotheses")
    for i, h in enumerate(hs, 1):
        print(f"    #{i} {h.cause_op} -> {h.effect_op}  p={h.p_value!r}  {h.direction}  [{h.notes}]")
    if v is not None:
        expected = {(c, e) for c, e in v["pairs"]}
        got = {(h.cause_op, h.effect_op) for h in hs}
        n_t_ok = all(f"n_treated={v['per_op'][h.cause_op]['treated_trunc']})" in h.notes for h in hs)
        print(f"  cross-check: hypotheses == pairs tested: {got == expected}; "
              f"n_treated matches truncated count: {n_t_ok}")


def _fixture(mechanisms=()) -> mf.Fixture:
    scn = mf.Scenario(points=[POINT], mechanisms=tuple(mechanisms), n_steps=N_STEPS,
                      noise_cv=0.0, seed=FIXTURE_SEED)
    return mf.generate(scn)[0]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _sweep_summary(name: str, seeds: range, rows: list[tuple], per_seed: bool) -> int:
    """rows: (seed, ate, se, p, n_treated, n). Returns the count with p < ALPHA."""
    hits = [r for r in rows if r[3] < ALPHA]
    lo, hi = _wilson(len(hits), len(rows))
    digest = hashlib.sha256(repr([(r[0], r[3]) for r in rows]).encode()).hexdigest()[:16]
    print(f"  [{name}] seeds {_span(seeds)}: p < {ALPHA} in {len(hits)}/{len(rows)} "
          f"= {len(hits) / len(rows):.1%} (Wilson 95% {lo:.1%}-{hi:.1%}; nominal {ALPHA:.0%})")
    print(f"  [{name}] sha256 of (seed, p) rows: {digest}")
    shown = rows if per_seed else hits
    label = "every seed" if per_seed else "seeds with p < 0.05"
    print(f"  [{name}] {label}: seed, ATE, se, z, p, n_treated/n")
    for s, ate, se, p, n_t, n in shown:
        z = ate / se if se not in (0.0, float("inf")) else 0.0
        print(f"      {s:>4}  ATE={ate:+.4f}  se={se:.4f}  z={z:+.3f}  p={p:.4g}  {n_t}/{n}")
    return len(hits)


# ── scenarios ──────────────────────────────────────────────────────────────


def scenario_1() -> None:
    _title("1. Healthy run: TruthModel eta = 0.8, no mechanism")
    fx = _fixture()
    print(f"  fixture: {POINT}, n_steps={N_STEPS}, eta_m={fx.scenario.truth.eta_m}, "
          f"seed={FIXTURE_SEED}, noise_cv=0.0")
    res = mf.observe(fx).residuals
    r = sorted({round(k.r_kt, 9) for k in res.per_kernel})
    print(f"  distinct r_kt values: {r} (band {BAND})")
    v = _view(res)
    _print_view(v)
    _print_hyps(attribute_dr(res, fx.graph), v)


def scenario_2() -> None:
    _title("2. M1 on attention at every step: RegionSlowdown(alpha=0.5, ops={attn_score_value})")
    fx = _fixture([mf.RegionSlowdown(0.5, ops={ATTN})])
    res = mf.observe(fx).residuals
    v = _view(res)
    _print_view(v)
    d = v["per_op"][ATTN]
    print(f"  {ATTN}: {d['treated_full']}/{d['rows']} rows treated in the full series; "
          f"{d['treated_trunc']}/{v['n']} after truncation, leaving {v['n'] - d['treated_trunc']} controls")
    _print_hyps(attribute_dr(res, fx.graph), v)


def scenario_3() -> tuple[Residuals, object]:
    _title("3. M1 on attention at steps 2-3 only: RegionSlowdown(alpha=1.0, steps={2, 3})")
    fx = _fixture([mf.RegionSlowdown(1.0, ops={ATTN}, steps={2, 3})])
    res = mf.observe(fx).residuals
    attn = [k for k in res.per_kernel if k.op == ATTN]
    n_layers = fx.scenario.model.n_layers
    assert [k.layer for k in attn] == [i % n_layers for i in range(len(attn))]
    treated_steps = sorted({i // n_layers for i, k in enumerate(attn) if abs(k.r_kt) > BAND})
    print(f"  {ATTN} rows cycle layers 0..{n_layers - 1} once per step (checked), so row i is step "
          f"i // {n_layers}; treated rows fall in steps {treated_steps}")

    print("\n  3a. full trace (lm_head included)")
    v = _view(res)
    _print_view(v)
    kept = attn[:v["n"]]
    print(f"  the {v['n']} {ATTN} rows that survive truncation are steps "
          f"{sorted({i // n_layers for i in range(v['n'])})}, layers {[k.layer for k in kept]}, "
          f"r_kt {[round(k.r_kt, 4) for k in kept]}")
    _print_hyps(attribute_dr(res, fx.graph), v)

    print("\n  3b. same trace with every lm_head row removed")
    no_lm = Residuals(per_kernel=[k for k in res.per_kernel if k.op != "lm_head"],
                      serialized_concurrency_fraction=res.serialized_concurrency_fraction)
    v = _view(no_lm)
    _print_view(v)
    _print_hyps(attribute_dr(no_lm, fx.graph), v)
    return no_lm, v


def scenario_4() -> None:
    _title(f"4. dr.py's own test fixture (test_attribute_dr_ranks_pairs), seed {DR_TEST_SEED}")
    print("  data: a ~ N(0, 0.05), a[::5] = 1.0; b = 0.8 a + N(0, 0.05); 40 rows each. "
          "True direction: A -> B")
    rng = np.random.default_rng(DR_TEST_SEED)
    a = rng.normal(0, 0.05, 40)
    a[::5] = 1.0
    b = 0.8 * a + rng.normal(0, 0.05, 40)
    res = _residuals({"A": a, "B": b})
    v = _view(res)
    _print_view(v)
    ranked = attribute_dr(res, predict_graph())
    _print_hyps(ranked, v)
    top = ranked.top(1)[0]
    print(f"  the test asserts only '\"doubly-robust ATE\" in top.notes'. "
          f"Top pair here: {top.cause_op} -> {top.effect_op}")


def _spurious_run(seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.05, 40)
    a[rng.choice(40, 8, replace=False)] = 1.0
    b = rng.normal(0, 0.05, 40)
    res = _residuals({"A": a, "B": b})
    hs = attribute_dr(res, predict_graph()).hypotheses
    assert [(h.cause_op, h.effect_op) for h in hs] == [("A", "B")], hs
    t = (np.abs(a) > BAND).astype(float)
    ate, se = doubly_robust_ate(b, t, np.arange(40, dtype=float))
    p = hs[0].p_value
    assert p == erfc(abs(ate / se) / sqrt(2)), (seed, p)
    return seed, ate, se, p, int(t.sum()), 40


def scenario_5(per_seed: bool) -> None:
    _title("5. Spurious: A spikes at 8 random rows, B independent N(0, 0.05); 40 rows")
    print("  per seed: rng = default_rng(seed); a = rng.normal(0, .05, 40); "
          "a[rng.choice(40, 8, replace=False)] = 1.0; b = rng.normal(0, .05, 40)")
    print("  B never crosses the band, so A -> B is the only pair tested (asserted every seed).")
    print("  p is attribute_dr's; ATE and se are doubly_robust_ate on the same rows, and "
          "p == erfc(|ATE/se|/sqrt 2) is asserted every seed.")
    _sweep_summary("primary", SPURIOUS_SEEDS, [_spurious_run(s) for s in SPURIOUS_SEEDS], per_seed)
    _sweep_summary("earlier seeds", SPURIOUS_SEEDS_EARLIER,
                   [_spurious_run(s) for s in SPURIOUS_SEEDS_EARLIER], per_seed)


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.9, sd: float = 0.3) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _ar1_direct(seed: int, n: int = 200) -> tuple:
    rng = np.random.default_rng(seed)
    a, b = _ar1(rng, n), _ar1(rng, n)
    t = (np.abs(a) > BAND).astype(float)
    ate, se = doubly_robust_ate(b, t, np.arange(n, dtype=float))
    z = ate / se if se not in (0.0, float("inf")) else 0.0
    return seed, ate, se, erfc(abs(z) / sqrt(2)), int(t.sum()), n


def _ar1_via_attribute_dr(seed: int, n: int = 40) -> tuple:
    rng = np.random.default_rng(seed)
    a, b = _ar1(rng, n), _ar1(rng, n)
    hs = {(h.cause_op, h.effect_op): h for h in
          attribute_dr(_residuals({"A": a, "B": b}), predict_graph()).hypotheses}
    t = (np.abs(a) > BAND).astype(float)
    ate, se = doubly_robust_ate(b, t, np.arange(n, dtype=float))
    h = hs.get(("A", "B"))
    p = h.p_value if h is not None else 1.0  # A -> B not tested: A had < 3 treated or controls
    return seed, ate, se, p, int(t.sum()), n


def scenario_6(per_seed: bool) -> None:
    _title("6. Independent AR(1) pair: x[i] = 0.9 x[i-1] + N(0, 0.3), x[0] = 0")
    print("  per seed: rng = default_rng(seed); a = AR1(rng, n); b = AR1(rng, n) (a drawn first)")
    print("  treatment t = |a| > 0.4; outcome b; covariate = position")
    print("\n  6a. earlier protocol: n = 200 rows, doubly_robust_ate(b, t, arange(200)) directly;"
          " p = erfc(|z|/sqrt 2), z = 0 when se is 0 or inf (attribute_dr's rule)")
    rows = [_ar1_direct(s) for s in AR1_SEEDS]
    _sweep_summary("primary", AR1_SEEDS, rows, per_seed)
    z196 = sum(1 for _, ate, se, *_ in rows if se not in (0.0, float("inf")) and abs(ate / se) > 1.96)
    print(f"  [primary] |z| > 1.96 (the earlier run's criterion): {z196}/{len(rows)}")
    rows = [_ar1_direct(s) for s in AR1_SEEDS_EARLIER]
    _sweep_summary("earlier seeds", AR1_SEEDS_EARLIER, rows, per_seed)
    z196 = sum(1 for _, ate, se, *_ in rows if se not in (0.0, float("inf")) and abs(ate / se) > 1.96)
    print(f"  [earlier seeds] |z| > 1.96 (the earlier run's criterion): {z196}/{len(rows)}")
    print("\n  6b. scenario 5's harness: n = 40 rows through attribute_dr; A -> B pair's p "
          "(1.0 when A is not eligible as a cause)")
    _sweep_summary("primary, 40 rows", AR1_SEEDS, [_ar1_via_attribute_dr(s) for s in AR1_SEEDS], per_seed)


def _zero_se_rows(cause: str, ranked, series: dict, n: int) -> None:
    t = (np.abs(np.asarray(series[cause][:n])) > BAND).astype(float)
    pos = np.arange(n, dtype=float)
    by_pair = {(h.cause_op, h.effect_op): h for h in ranked.hypotheses}
    for effect in [op for op in series if op != cause]:
        y = np.asarray(series[effect][:n], dtype=float)
        ate, se = doubly_robust_ate(y, t, pos)
        z = ate / se if se not in (0.0, float("inf")) else 0.0
        h = by_pair.get((cause, effect))
        print(f"    {cause} -> {effect}: distinct y = {sorted(set(y.tolist()))}")
        print(f"      doubly_robust_ate: ATE={ate!r}, se={se!r}, se == 0.0: {se == 0.0}")
        print(f"      z by attribute_dr's rule = {z!r}, p = erfc(|z|/sqrt 2) = {erfc(abs(z) / sqrt(2))!r}")
        print(f"      listed by attribute_dr: {h is not None}"
              + (f" -> p={h.p_value!r}, direction={h.direction!r}, notes=[{h.notes}]" if h else ""))


def scenario_7(no_lm: Residuals, v: dict) -> None:
    _title("7. Zero standard error")
    print("  7a. scenario 3b's input: every effect series is one constant value")
    graph = _fixture([mf.RegionSlowdown(1.0, ops={ATTN}, steps={2, 3})]).graph
    series = {op: v["series"][op] for op in v["ops"]}
    _zero_se_rows(ATTN, attribute_dr(no_lm, graph), series, v["n"])

    print(f"\n  7b. constructed: A as in scenario 5 with seed {ZERO_SE_SEED}; B is exactly 0.0 on all 40 rows")
    rng = np.random.default_rng(ZERO_SE_SEED)
    a = rng.normal(0, 0.05, 40)
    a[rng.choice(40, 8, replace=False)] = 1.0
    b = np.zeros(40)
    res = _residuals({"A": a, "B": b})
    vb = _view(res)
    _print_view(vb)
    ranked = attribute_dr(res, predict_graph())
    _print_hyps(ranked, vb)
    _zero_se_rows("A", ranked, {"A": a, "B": b}, 40)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-seed", action="store_true",
                    help="print every seed's row in scenarios 5 and 6, not only those with p < 0.05")
    args = ap.parse_args()
    _header()
    scenario_1()
    scenario_2()
    no_lm, v = scenario_3()
    scenario_4()
    scenario_5(args.per_seed)
    scenario_6(args.per_seed)
    scenario_7(no_lm, v)
    print("\ndone.")


if __name__ == "__main__":
    main()
