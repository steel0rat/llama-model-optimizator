"""Full two-phase optimization with progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moe_optimizator.bench.models import BenchRecord, BenchRunError
from moe_optimizator.bench.runner import run_llama_bench
from moe_optimizator.optimizer.callbacks import NullProgress, ProgressCallback, ProgressEvent
from moe_optimizator.optimizer.cancel import CancelToken, OptimizationCancelled
from moe_optimizator.optimizer.config import OptimizationConfig
from moe_optimizator.optimizer.phases import (
    _search_space_to_bench_args,
    config_key_to_bench_args,
    default_config_key,
    inference_tuning_matrix,
    rank_configurations,
    run_confirmation_at_ctx,
    tuning_depth_for,
)
from moe_optimizator.optimizer.report import build_report


@dataclass
class OptimizationResult:
    ctx_max: int
    tuning_depth: int
    best_config: tuple
    records: list[BenchRecord]
    ranked: list[tuple]
    report_path: Path


def run_inference_tuning_with_progress(
    config: OptimizationConfig,
    progress: ProgressCallback,
    cancel: CancelToken | None = None,
) -> list[BenchRecord]:
    """Phase 1: search-space grid without KV context (``-d 0``)."""
    base = _search_space_to_bench_args(config.search)
    base.extend(["-r", str(config.bench_repetitions)])
    matrix = inference_tuning_matrix(config)
    all_records: list[BenchRecord] = []

    for idx, test_args in enumerate(matrix):
        if cancel is not None:
            cancel.check()
        label = " ".join(test_args)
        pct = (idx / len(matrix)) * 100.0
        progress.on_progress(
            ProgressEvent(
                phase="tuning",
                message=f"Прогон {idx + 1}/{len(matrix)}: {label} (без контекста, d=0)",
                percent=pct,
            )
        )
        progress.on_log(f"[фаза 1] llama-bench {label}")
        try:
            records = run_llama_bench(
                config.llama_bench,
                model=config.model,
                extra_args=[*base, *test_args],
                cancel=cancel,
            )
        except OptimizationCancelled:
            raise
        all_records.extend(records)
        for rec in records[:3]:
            progress.on_log(
                f"  → {rec.test_label}: {rec.avg_ts:.2f} t/s "
                f"(ngl={rec.n_gpu_layers}, t={rec.n_threads}, fa={rec.flash_attn})"
            )
        if len(records) > 3:
            progress.on_log(f"  … ещё {len(records) - 3} записей")

    progress.on_progress(
        ProgressEvent(phase="tuning", message="Подбор inference завершён", percent=100.0)
    )
    return all_records


def find_ctx_max_with_progress(
    config: OptimizationConfig,
    config_key: tuple,
    progress: ProgressCallback,
    cancel: CancelToken | None = None,
) -> int:
    """Phase 2: binary search ``-d`` for the chosen configuration."""
    lo = config.ctx_min
    hi = config.ctx_max
    best = 0
    base = config_key_to_bench_args(config_key)
    smoke = ["-n", "16", "-p", "0", "-r", "1"]
    step = 0
    flags = " ".join(base)

    while lo <= hi:
        if cancel is not None:
            cancel.check()
        mid = ((lo + hi) // 2 // config.ctx_step) * config.ctx_step
        mid = max(mid, config.ctx_min)
        step += 1
        pct = min(95.0, step * 12.0)
        progress.on_progress(
            ProgressEvent(
                phase="ctx_search",
                message=f"Проверка d={mid:,} (диапазон {lo:,}–{hi:,})",
                percent=pct,
            )
        )
        progress.on_log(f"[фаза 2] llama-bench -d {mid} · {flags}")

        try:
            run_llama_bench(
                config.llama_bench,
                model=config.model,
                extra_args=[*base, *smoke, "-d", str(mid)],
                cancel=cancel,
            )
            best = mid
            lo = mid + config.ctx_step
            progress.on_log(f"[фаза 2] OK — d={mid} влезает")
        except BenchRunError as exc:
            hi = mid - config.ctx_step
            progress.on_log(f"[фаза 2] OOM/ошибка при d={mid}: {exc}")
        except OptimizationCancelled:
            raise

    result = best if best > 0 else config.ctx_min
    progress.on_progress(
        ProgressEvent(
            phase="ctx_search",
            message=f"ctx_max = {result:,}",
            percent=100.0,
        )
    )
    return result


def run_optimization(
    config: OptimizationConfig,
    *,
    skip_tuning: bool = False,
    skip_ctx_search: int | None = None,
    skip_phase1: int | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> OptimizationResult:
    """
    Phase 1: inference grid at ``-d 0`` (no KV context).
    Phase 2: binary search max ``-d`` for the winning configuration.
    """
    cb = progress or NullProgress()
    out = config.output_dir or Path("optimization_out")
    tuning_depth = tuning_depth_for(config)

    if cancel is not None:
        cancel.check()

    # --- Phase 1: performance / inference flags ---
    if skip_tuning:
        best_config = default_config_key(config.search)
        records: list[BenchRecord] = []
        ranked: list[tuple] = []
        cb.on_log("Фаза 1 пропущена — конфигурация по умолчанию из search space")
    else:
        cb.on_progress(
            ProgressEvent(
                phase="tuning",
                message="Подбор inference (d=0, без KV)…",
                percent=0,
            )
        )
        records = run_inference_tuning_with_progress(config, cb, cancel)
        if cancel is not None:
            cancel.check()
        cb.on_progress(
            ProgressEvent(phase="ranking", message="Ранжирование конфигураций…", percent=50)
        )
        ranked = rank_configurations(records, tuning_depth)
        if not ranked:
            best_config = default_config_key(config.search)
            cb.on_log("Нет успешных прогонов — используется конфигурация по умолчанию")
        else:
            best_config = ranked[0][0]
            m = ranked[0][1]
            cb.on_log(
                f"Лучшая конфигурация: tg@cold={m.get('tg@cold')}, "
                f"pp@cold={m.get('pp@cold')}, pg@cold={m.get('pg@cold')}"
            )

    if cancel is not None:
        cancel.check()

    # --- Phase 2: ctx_max for best config ---
    ctx_skip = skip_ctx_search if skip_ctx_search is not None else skip_phase1
    if ctx_skip is not None:
        ctx_max = ctx_skip
        cb.on_log(f"Фаза 2 пропущена, ctx_max = {ctx_max:,}")
    else:
        cb.on_progress(
            ProgressEvent(phase="ctx_search", message="Поиск ctx_max…", percent=0)
        )
        ctx_max = find_ctx_max_with_progress(config, best_config, cb, cancel)

    if cancel is not None:
        cancel.check()

    if not skip_tuning and ranked and ctx_max > 0:
        cb.on_log(f"Контрольный tg@ctx при d={ctx_max:,}")
        try:
            records.extend(run_confirmation_at_ctx(config, best_config, ctx_max))
        except Exception as exc:
            cb.on_log(f"Контрольный прогон не удался: {exc}")

    cb.on_progress(ProgressEvent(phase="report", message="Сохранение отчёта…", percent=90))
    report_path = build_report(
        ctx_max=ctx_max,
        tuning_depth=tuning_depth,
        best_config=best_config,
        records=records,
        ranked=ranked,
        output_dir=out,
    )
    cb.on_progress(
        ProgressEvent(phase="done", message=f"Готово: {report_path}", percent=100.0)
    )
    return OptimizationResult(
        ctx_max=ctx_max,
        tuning_depth=tuning_depth,
        best_config=best_config,
        records=records,
        ranked=ranked,
        report_path=report_path,
    )
