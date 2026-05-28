"""Phase 1 (ctx_max) and phase 2 (full metrics) orchestration."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

from moe_optimizator.bench.models import BenchRecord, BenchRunError
from moe_optimizator.bench.runner import run_llama_bench
from moe_optimizator.optimizer.config import OptimizationConfig, SearchSpace


def _comma(values: list) -> str:
    return ",".join(str(v) for v in values)


def _search_space_to_bench_args(space: SearchSpace) -> list[str]:
    args: list[str] = []
    if space.n_gpu_layers:
        args.extend(["-ngl", _comma(space.n_gpu_layers)])
    if space.n_threads:
        args.extend(["-t", _comma(space.n_threads)])
    if space.n_batch:
        args.extend(["-b", _comma(space.n_batch)])
    if space.n_ubatch:
        args.extend(["-ub", _comma(space.n_ubatch)])
    if space.flash_attn:
        args.extend(["-fa", _comma(space.flash_attn)])
    if space.cache_type_k:
        args.extend(["-ctk", _comma(space.cache_type_k)])
    if space.cache_type_v:
        args.extend(["-ctv", _comma(space.cache_type_v)])
    if space.no_kv_offload:
        args.extend(["-nkvo", _comma(space.no_kv_offload)])
    if space.use_mmap:
        args.extend(["-mmp", _comma(space.use_mmap)])
    if space.n_cpu_moe:
        args.extend(["-ncmoe", _comma(space.n_cpu_moe)])
    return args


def phase2_test_matrix(config: OptimizationConfig, ctx_max: int) -> list[list[str]]:
    """Bench argument sets for multi-metric phase 2."""
    p, n, d = config.prompt_tokens, config.gen_tokens, ctx_max
    return [
        # decode @ full context (primary)
        ["-n", str(n), "-p", "0", "-d", str(d)],
        # prefill @ full context
        ["-p", str(p), "-n", "0", "-d", str(d)],
        # combined @ full context
        ["-pg", f"{p},{n}", "-d", str(d)],
        # cold baselines
        ["-n", str(n), "-p", "0", "-d", "0"],
        ["-p", str(p), "-n", "0", "-d", "0"],
    ]


def find_ctx_max(config: OptimizationConfig) -> int:
    """
    Phase 1: binary search largest ``-d`` where smoke ``tg`` succeeds.

    Uses minimal gen length and baseline inference flags.
    """
    lo = config.ctx_min
    hi = config.ctx_max
    best = 0
    smoke = ["-n", "16", "-p", "0", "-r", "1"]

    while lo <= hi:
        mid = ((lo + hi) // 2 // config.ctx_step) * config.ctx_step
        mid = max(mid, config.ctx_min)

        try:
            run_llama_bench(
                config.llama_bench,
                model=config.model,
                extra_args=[*smoke, "-d", str(mid)],
            )
            best = mid
            lo = mid + config.ctx_step
        except BenchRunError:
            hi = mid - config.ctx_step

    return best if best > 0 else config.ctx_min


def run_phase2_benchmarks(
    config: OptimizationConfig,
    ctx_max: int,
) -> list[BenchRecord]:
    """Phase 2: search space × metric matrix at ``-d ctx_max`` (and cold tests)."""
    base = _search_space_to_bench_args(config.search)
    base.extend(["-r", str(config.bench_repetitions)])

    all_records: list[BenchRecord] = []
    for test_args in phase2_test_matrix(config, ctx_max):
        records = run_llama_bench(
            config.llama_bench,
            model=config.model,
            extra_args=[*base, *test_args],
        )
        all_records.extend(records)
    return all_records


def _config_key(rec: BenchRecord) -> tuple:
    """Group records by inference flags (exclude test shape)."""
    return (
        rec.n_gpu_layers,
        rec.n_threads,
        rec.n_batch,
        rec.n_ubatch,
        rec.type_k,
        rec.type_v,
        rec.flash_attn,
        rec.no_kv_offload,
        rec.use_mmap,
        rec.split_mode,
        rec.main_gpu,
        rec.poll,
    )


def _pick_metric(
    records: list[BenchRecord],
    *,
    kind: str,
    depth: int,
) -> float | None:
    check = {"pp": lambda r: r.is_pp, "tg": lambda r: r.is_tg, "pg": lambda r: r.is_pg}[kind]
    for r in records:
        if r.n_depth == depth and check(r):
            return r.avg_ts
    return None


def rank_configurations(
    records: list[BenchRecord],
    ctx_max: int,
) -> list[tuple[tuple, dict[str, float | None], float]]:
    """
    Rank config groups: primary ``tg@ctx``, tie-breakers ``pg@ctx``, ``pp@ctx``, stability.

    Returns list of (config_key, metrics_dict, sort_score) descending by score.
    """
    groups: dict[tuple, list[BenchRecord]] = {}
    for rec in records:
        groups.setdefault(_config_key(rec), []).append(rec)

    ranked: list[tuple[tuple, dict[str, float | None], float]] = []
    for key, group in groups.items():
        metrics = {
            "tg@ctx": _pick_metric(group, kind="tg", depth=ctx_max),
            "pp@ctx": _pick_metric(group, kind="pp", depth=ctx_max),
            "pg@ctx": _pick_metric(group, kind="pg", depth=ctx_max),
            "tg@cold": _pick_metric(group, kind="tg", depth=0),
            "pp@cold": _pick_metric(group, kind="pp", depth=0),
        }
        tg = metrics["tg@ctx"] or 0.0
        pg = metrics["pg@ctx"] or 0.0
        pp = metrics["pp@ctx"] or 0.0
        stability = sum(r.stddev_ts for r in group if r.n_depth == ctx_max) or 1.0
        score = tg * 1_000_000 + pg * 1_000 + pp - stability
        ranked.append((key, metrics, score))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def iter_search_combinations(space: SearchSpace) -> Iterator[dict[str, int | str]]:
    """Enumerate search space (for future per-combo runs without bench cartesian)."""
    keys = [
        ("n_gpu_layers", space.n_gpu_layers),
        ("n_threads", space.n_threads),
        ("n_batch", space.n_batch),
        ("n_ubatch", space.n_ubatch),
        ("flash_attn", space.flash_attn),
        ("cache_type_k", space.cache_type_k),
        ("cache_type_v", space.cache_type_v),
        ("no_kv_offload", space.no_kv_offload),
        ("use_mmap", space.use_mmap),
        ("n_cpu_moe", space.n_cpu_moe),
    ]
    names = [k for k, _ in keys]
    values = [v for _, v in keys]
    for combo in itertools.product(*values):
        yield dict(zip(names, combo, strict=True))
