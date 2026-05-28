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
    phase2_test_matrix,
    rank_configurations,
)
from moe_optimizator.optimizer.report import build_report


@dataclass
class OptimizationResult:
    ctx_max: int
    records: list[BenchRecord]
    ranked: list[tuple]
    report_path: Path


def find_ctx_max_with_progress(
    config: OptimizationConfig,
    progress: ProgressCallback,
    cancel: CancelToken | None = None,
) -> int:
    lo = config.ctx_min
    hi = config.ctx_max
    best = 0
    smoke = ["-n", "16", "-p", "0", "-r", "1"]
    step = 0

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
                message=f"Проверка глубины контекста d={mid:,} (диапазон {lo:,}–{hi:,})",
                percent=pct,
            )
        )
        progress.on_log(f"[фаза 1] llama-bench -d {mid}")

        try:
            run_llama_bench(
                config.llama_bench,
                model=config.model,
                extra_args=[*smoke, "-d", str(mid)],
                cancel=cancel,
            )
            best = mid
            lo = mid + config.ctx_step
            progress.on_log(f"[фаза 1] OK — d={mid} влезает")
        except BenchRunError as exc:
            hi = mid - config.ctx_step
            progress.on_log(f"[фаза 1] OOM/ошибка при d={mid}: {exc}")
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


def run_phase2_with_progress(
    config: OptimizationConfig,
    ctx_max: int,
    progress: ProgressCallback,
    cancel: CancelToken | None = None,
) -> list[BenchRecord]:
    base = _search_space_to_bench_args(config.search)
    base.extend(["-r", str(config.bench_repetitions)])
    matrix = phase2_test_matrix(config, ctx_max)
    all_records: list[BenchRecord] = []

    for idx, test_args in enumerate(matrix):
        if cancel is not None:
            cancel.check()
        label = " ".join(test_args)
        pct = (idx / len(matrix)) * 100.0
        progress.on_progress(
            ProgressEvent(
                phase="benchmark",
                message=f"Прогон {idx + 1}/{len(matrix)}: {label}",
                percent=pct,
            )
        )
        progress.on_log(f"[фаза 2] llama-bench {label}")
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
        ProgressEvent(phase="benchmark", message="Бенчмарки завершены", percent=100.0)
    )
    return all_records


def run_optimization(
    config: OptimizationConfig,
    *,
    skip_phase1: int | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelToken | None = None,
) -> OptimizationResult:
    cb = progress or NullProgress()
    out = config.output_dir or Path("optimization_out")

    if cancel is not None:
        cancel.check()

    if skip_phase1 is not None:
        ctx_max = skip_phase1
        cb.on_log(f"Фаза 1 пропущена, ctx_max = {ctx_max:,}")
    else:
        cb.on_progress(ProgressEvent(phase="ctx_search", message="Поиск ctx_max…", percent=0))
        ctx_max = find_ctx_max_with_progress(config, cb, cancel)

    if cancel is not None:
        cancel.check()

    cb.on_progress(ProgressEvent(phase="benchmark", message="Запуск бенчмарков…", percent=0))
    records = run_phase2_with_progress(config, ctx_max, cb, cancel)

    if cancel is not None:
        cancel.check()

    cb.on_progress(ProgressEvent(phase="ranking", message="Ранжирование конфигураций…", percent=50))
    ranked = rank_configurations(records, ctx_max)

    cb.on_progress(ProgressEvent(phase="report", message="Сохранение отчёта…", percent=90))
    report_path = build_report(
        ctx_max=ctx_max,
        records=records,
        ranked=ranked,
        output_dir=out,
    )
    cb.on_progress(
        ProgressEvent(phase="done", message=f"Готово: {report_path}", percent=100.0)
    )
    return OptimizationResult(
        ctx_max=ctx_max,
        records=records,
        ranked=ranked,
        report_path=report_path,
    )
