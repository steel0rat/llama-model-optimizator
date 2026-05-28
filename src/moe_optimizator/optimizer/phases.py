"""Phase 1 (inference tuning) and phase 2 (ctx_max for best config)."""

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


def default_config_key(space: SearchSpace) -> tuple:
    """Single configuration from the first value of each search-space axis."""
    return (
        space.n_gpu_layers[0],
        space.n_threads[0],
        space.n_batch[0],
        space.n_ubatch[0],
        space.cache_type_k[0],
        space.cache_type_v[0],
        bool(space.flash_attn[0]),
        bool(space.no_kv_offload[0]),
        bool(space.use_mmap[0]),
        "layer",
        0,
        50,
    )


def config_key_to_bench_args(key: tuple) -> list[str]:
    """Fixed inference flags for one configuration (phase 2 ctx search)."""
    (
        ngl,
        threads,
        batch,
        ubatch,
        type_k,
        type_v,
        flash_attn,
        no_kv_offload,
        use_mmap,
        *_rest,
    ) = key
    return [
        "-ngl",
        str(ngl),
        "-t",
        str(threads),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-fa",
        "1" if flash_attn else "0",
        "-ctk",
        str(type_k),
        "-ctv",
        str(type_v),
        "-nkvo",
        "1" if no_kv_offload else "0",
        "-mmp",
        "1" if use_mmap else "0",
    ]


# Phase 1 always runs without KV context (fast grid).
PHASE1_DEPTH = 0


def inference_tuning_matrix(config: OptimizationConfig) -> list[list[str]]:
    """Phase 1: tg / pp / pg at ``-d 0`` (no filled KV cache)."""
    p, n = config.prompt_tokens, config.gen_tokens
    return [
        ["-n", str(n), "-p", "0", "-d", "0"],
        ["-p", str(p), "-n", "0", "-d", "0"],
        ["-pg", f"{p},{n}", "-d", "0"],
    ]


def inference_test_matrix(config: OptimizationConfig, depth: int) -> list[list[str]]:
    """Full metric matrix at a given ``-d`` (used after ctx_max is known)."""
    p, n = config.prompt_tokens, config.gen_tokens
    return [
        ["-n", str(n), "-p", "0", "-d", str(depth)],
        ["-p", str(p), "-n", "0", "-d", str(depth)],
        ["-pg", f"{p},{n}", "-d", str(depth)],
        ["-n", str(n), "-p", "0", "-d", "0"],
        ["-p", str(p), "-n", "0", "-d", "0"],
    ]


def tuning_depth_for(_config: OptimizationConfig) -> int:
    """Phase 1 comparison depth (always zero — no context)."""
    return PHASE1_DEPTH


def run_inference_tuning(config: OptimizationConfig) -> list[BenchRecord]:
    """Phase 1: grid over search space at ``-d 0``."""
    base = _search_space_to_bench_args(config.search)
    base.extend(["-r", str(config.bench_repetitions)])
    all_records: list[BenchRecord] = []
    for test_args in inference_tuning_matrix(config):
        records = run_llama_bench(
            config.llama_bench,
            model=config.model,
            extra_args=[*base, *test_args],
        )
        all_records.extend(records)
    return all_records


def find_ctx_max_for_config(
    config: OptimizationConfig,
    config_key: tuple,
) -> int:
    """
    Phase 2: binary search largest ``-d`` for a fixed inference configuration.

    Uses a short smoke ``tg`` test.
    """
    lo = config.ctx_min
    hi = config.ctx_max
    best = 0
    base = config_key_to_bench_args(config_key)
    smoke = ["-n", "16", "-p", "0", "-r", "1"]

    while lo <= hi:
        mid = ((lo + hi) // 2 // config.ctx_step) * config.ctx_step
        mid = max(mid, config.ctx_min)

        try:
            run_llama_bench(
                config.llama_bench,
                model=config.model,
                extra_args=[*base, *smoke, "-d", str(mid)],
            )
            best = mid
            lo = mid + config.ctx_step
        except BenchRunError:
            hi = mid - config.ctx_step

    return best if best > 0 else config.ctx_min


def run_confirmation_at_ctx(
    config: OptimizationConfig,
    config_key: tuple,
    ctx_max: int,
) -> list[BenchRecord]:
    """Final ``tg@ctx`` at discovered ctx_max for the winning configuration."""
    base = config_key_to_bench_args(config_key)
    base.extend(["-r", str(config.bench_repetitions)])
    n = config.gen_tokens
    return run_llama_bench(
        config.llama_bench,
        model=config.model,
        extra_args=[*base, "-n", str(n), "-p", "0", "-d", str(ctx_max)],
    )


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
    ctx_depth: int,
) -> list[tuple[tuple, dict[str, float | None], float]]:
    """
    Rank config groups by throughput at ``ctx_depth``.

    Phase 1 (``ctx_depth == 0``): primary ``tg@cold``, then ``pg`` / ``pp`` at ``d=0``.
    Phase 2+ at full context: primary ``tg@ctx``, tie-breakers ``pg@ctx``, ``pp@ctx``.
    """
    groups: dict[tuple, list[BenchRecord]] = {}
    for rec in records:
        groups.setdefault(_config_key(rec), []).append(rec)

    ranked: list[tuple[tuple, dict[str, float | None], float]] = []
    for key, group in groups.items():
        if ctx_depth == 0:
            metrics = {
                "tg@ctx": None,
                "pp@ctx": None,
                "pg@ctx": None,
                "tg@cold": _pick_metric(group, kind="tg", depth=0),
                "pp@cold": _pick_metric(group, kind="pp", depth=0),
                "pg@cold": _pick_metric(group, kind="pg", depth=0),
            }
            tg = metrics["tg@cold"] or 0.0
            pg = metrics["pg@cold"] or 0.0
            pp = metrics["pp@cold"] or 0.0
        else:
            metrics = {
                "tg@ctx": _pick_metric(group, kind="tg", depth=ctx_depth),
                "pp@ctx": _pick_metric(group, kind="pp", depth=ctx_depth),
                "pg@ctx": _pick_metric(group, kind="pg", depth=ctx_depth),
                "tg@cold": _pick_metric(group, kind="tg", depth=0),
                "pp@cold": _pick_metric(group, kind="pp", depth=0),
                "pg@cold": None,
            }
            tg = metrics["tg@ctx"] or 0.0
            pg = metrics["pg@ctx"] or 0.0
            pp = metrics["pp@ctx"] or 0.0
        stability = sum(r.stddev_ts for r in group if r.n_depth == ctx_depth) or 1.0
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


def find_ctx_max(
    config: OptimizationConfig,
    config_key: tuple | None = None,
) -> int:
    """Backward-compatible entry: ctx search for one configuration."""
    key = config_key or default_config_key(config.search)
    return find_ctx_max_for_config(config, key)


# Backward-compatible aliases
phase2_test_matrix = inference_test_matrix


def run_phase2_benchmarks(
    config: OptimizationConfig,
    ctx_max: int,
) -> list[BenchRecord]:
    """Deprecated name: full matrix at ``ctx_max`` (not phase 1)."""
    base = _search_space_to_bench_args(config.search)
    base.extend(["-r", str(config.bench_repetitions)])
    all_records: list[BenchRecord] = []
    for test_args in inference_test_matrix(config, ctx_max):
        records = run_llama_bench(
            config.llama_bench,
            model=config.model,
            extra_args=[*base, *test_args],
        )
        all_records.extend(records)
    return all_records
