"""A CPU-only synthetic grid of operating points for exercising the metric.

Each point's baseline "capture" is a :mod:`gitm.optimizer.mechanism_fixtures`
trace: the dense fixture model on its default hardware, healthy (no mechanism),
at the point's bridged batch and KV length. Nothing here ran on a GPU and no
candidate was ever applied, so these traces stand in for baseline captures only.
They are labelled ``capture_source=SYNTHETIC_CAPTURE`` and must never be read as
measurements.

The grid's (concurrency, input, output) triples come from
:attr:`DistanceConfig.grid`, so only concurrency and input length vary, as in
the design doc's §7.1. The model is not Kimi K2.5 and the hardware is not
MI355X: the fixture generator only builds dense graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from gitm.kernels.library import load_library
from gitm.optimizer import mechanism_fixtures as mf
from gitm.optimizer.preconditions import GateContext
from gitm.planner.context import PlannerContext, peak_for_sku
from gitm.planner.graph import predict_graph
from gitm.planner.roofline import BatchConfig, HardwareSpec, ModelSpec
from gitm.research.distance_metric.config import DistanceConfig
from gitm.research.distance_metric.features import PointDescription
from gitm.traffic.regime import Regime, SourceKind

SYNTHETIC_CAPTURE = "mechanism_fixtures (synthetic, healthy, noise_cv=0)"
WORKLOAD = "vllm-decode"
#: Decode steps per synthetic capture. Coverage and shares are per-step
#: fractions, so the count does not change any feature; 2 keeps it fast.
N_STEPS = 2


@dataclass(frozen=True)
class SyntheticEngine:
    """A baseline engine as flat attributes, the duck type ``get_knob`` falls
    back to. It never runs; ``gitm_restart_fn`` exists so structural candidates
    are not vetoed, as on a deployment that provides the restart hook."""

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_model_len: int = 16384
    max_seq_len_to_capture: int = 16384
    enable_chunked_prefill: bool = True
    enforce_eager: bool = False
    kv_cache_dtype: str = "auto"
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    swap_space: int = 8
    tensor_parallel_size: int = 1

    def gitm_restart_fn(self, old_engine: Any, knob_values: dict[str, Any]) -> Any:
        raise RuntimeError("synthetic engine: never restarted, only read")


def synthetic_regime(concurrency: int, input_len: int, output_len: int) -> Regime:
    """Fixed-length closed-loop traffic: p50 equals p95. Burstiness is the same
    at every point, so the traffic component sees only the lengths."""
    return Regime(
        source_kind=SourceKind.SYNTHETIC,
        trace="distance-metric-v1-synthetic",
        requests=1024,
        rate_rps=1.0,
        io_ratio=input_len / output_len,
        input_p50=input_len,
        input_p95=input_len,
        output_p50=output_len,
        output_p95=output_len,
        burstiness=1.0,
        concurrency=concurrency,
    )


def synthetic_point(
    concurrency: int,
    input_len: int,
    output_len: int,
    *,
    seed: int,
    engine: SyntheticEngine | None = None,
    model: ModelSpec | None = None,
    hw: HardwareSpec | None = None,
    library: tuple | None = None,
    name: str | None = None,
) -> PointDescription:
    engine = engine or SyntheticEngine()
    model = model or ModelSpec()
    hw = hw or HardwareSpec()
    regime = synthetic_regime(concurrency, input_len, output_len)
    batch = min(concurrency, engine.max_num_seqs)
    kv = input_len + output_len // 2
    fx = mf.generate(mf.Scenario(model=model, hw=hw, points=(BatchConfig(batch=batch, kv_cache_len=kv),),
                                 n_steps=N_STEPS, noise_cv=0.0, seed=seed))[0]
    gate = GateContext(workload=WORKLOAD, dtype="bf16", hardware=hw.name,
                       kv_cache_len=engine.max_model_len, num_gpus=1)
    return PointDescription(
        name=name or f"c{concurrency}_in{input_len}_out{output_len}",
        model=model.name,
        model_revision="fixture",
        engine_name="vllm",
        engine_version="fixture",
        planner=PlannerContext(gate=gate, peak=peak_for_sku(hw.name), sku=hw.name, num_gpus=1),
        regime=regime,
        engine=engine,
        library=tuple(load_library(workload=WORKLOAD)) if library is None else library,
        graph_at=partial(predict_graph, model, hw),
        planner_version="predict_graph (dense, mechanism_fixtures)",
        baseline_trace=fx.trace,
        scheduler_stats=None,
        capture_source=SYNTHETIC_CAPTURE,
    )


def synthetic_grid(config: DistanceConfig, *, seed: int) -> list[PointDescription]:
    library = tuple(load_library(workload=WORKLOAD))
    return [synthetic_point(c, i, o, seed=seed, library=library) for c, i, o in config.grid]
