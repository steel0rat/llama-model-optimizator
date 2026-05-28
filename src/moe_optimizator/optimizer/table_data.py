"""Tabular views of optimization results."""

from __future__ import annotations

from typing import Any

from moe_optimizator.bench.models import BenchRecord
from moe_optimizator.optimizer.phases import _config_key


def ranking_table_rows(
    ranked: list[tuple],
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (key, metrics, score) in enumerate(ranked[:limit]):
        rows.append(
            {
                "rank": i + 1,
                "ngl": key[0],
                "threads": key[1],
                "batch": key[2],
                "ubatch": key[3],
                "type_k": key[4],
                "type_v": key[5],
                "flash_attn": key[6],
                "no_kv_offload": key[7],
                "mmap": key[8],
                "tg@ctx": _fmt(metrics.get("tg@ctx")),
                "pp@ctx": _fmt(metrics.get("pp@ctx")),
                "pg@ctx": _fmt(metrics.get("pg@ctx")),
                "tg@cold": _fmt(metrics.get("tg@cold")),
                "pp@cold": _fmt(metrics.get("pp@cold")),
                "score": round(score, 2),
            }
        )
    return rows


def records_table_rows(records: list[BenchRecord], *, limit: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records[:limit]:
        rows.append(
            {
                "test": rec.test_label,
                "metric": rec.metric_id,
                "t/s": round(rec.avg_ts, 2),
                "±": round(rec.stddev_ts, 2),
                "depth": rec.n_depth,
                "ngl": rec.n_gpu_layers,
                "threads": rec.n_threads,
                "fa": rec.flash_attn,
                "ctk": rec.type_k,
                "ctv": rec.type_v,
            }
        )
    return rows


def best_record(
    records: list[BenchRecord],
    ranked: list[tuple],
    ctx_max: int,
) -> BenchRecord | None:
    if not ranked:
        return None
    key = ranked[0][0]
    for rec in records:
        if _config_key(rec) == key and rec.is_tg and rec.n_depth == ctx_max:
            return rec
    return records[0] if records else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"
