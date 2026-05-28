"""Reports and server flag export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from moe_optimizator.bench.models import BenchRecord


def records_to_rows(records: list[BenchRecord]) -> list[dict[str, Any]]:
    """Flatten all bench records for CSV/JSON export."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row = {
            "test_label": rec.test_label,
            "metric_id": rec.metric_id,
            "avg_ts": rec.avg_ts,
            "stddev_ts": rec.stddev_ts,
            "avg_ns": rec.avg_ns,
            "stddev_ns": rec.stddev_ns,
            "n_prompt": rec.n_prompt,
            "n_gen": rec.n_gen,
            "n_depth": rec.n_depth,
            "n_gpu_layers": rec.n_gpu_layers,
            "n_threads": rec.n_threads,
            "n_batch": rec.n_batch,
            "n_ubatch": rec.n_ubatch,
            "type_k": rec.type_k,
            "type_v": rec.type_v,
            "flash_attn": rec.flash_attn,
            "no_kv_offload": rec.no_kv_offload,
            "use_mmap": rec.use_mmap,
            "backends": rec.backends,
            "cpu_info": rec.cpu_info,
            "gpu_info": rec.gpu_info,
        }
        rows.append(row)
    return rows


def server_flags_from_record(rec: BenchRecord, *, ctx_size: int, parallel: int = 1) -> list[str]:
    """Build llama-server CLI args from a bench record."""
    fa = "on" if rec.flash_attn else "off"
    nkvo = ["--no-kv-offload"] if rec.no_kv_offload else []
    return [
        "-m",
        rec.model_filename,
        "-c",
        str(ctx_size),
        "-np",
        str(parallel),
        "-ngl",
        str(rec.n_gpu_layers),
        "-t",
        str(rec.n_threads),
        "-tb",
        str(rec.n_threads),
        "-b",
        str(rec.n_batch),
        "-ub",
        str(rec.n_ubatch),
        "-fa",
        fa,
        "-ctk",
        rec.type_k,
        "-ctv",
        rec.type_v,
        *nkvo,
    ]


def build_report(
    *,
    ctx_max: int,
    tuning_depth: int,
    best_config: tuple,
    records: list[BenchRecord],
    ranked: list[tuple],
    output_dir: Path,
) -> Path:
    """Write JSON report with all metrics and best server command."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from moe_optimizator.optimizer.phases import _config_key

    best_rec = None
    if best_config:
        for rec in records:
            if _config_key(rec) == best_config and rec.is_tg and rec.n_depth == ctx_max:
                best_rec = rec
                break
        if best_rec is None:
            for rec in records:
                if _config_key(rec) == best_config and rec.is_tg:
                    best_rec = rec
                    break

    report = {
        "ctx_max": ctx_max,
        "tuning_depth": tuning_depth,
        "best_config": best_config,
        "server_parallel": 1,
        "records": records_to_rows(records),
        "ranking": [
            {"config": k, "metrics": m, "score": s} for k, m, s in ranked
        ],
        "server_command": None,
    }
    if best_rec:
        flags = server_flags_from_record(best_rec, ctx_size=ctx_max, parallel=1)
        report["server_command"] = "llama-server " + " ".join(flags)

    path = output_dir / "optimization_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
