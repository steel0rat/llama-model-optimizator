"""Parse llama-bench JSON output."""

from __future__ import annotations

import json
from typing import Any

from moe_optimizator.bench.models import BenchRecord


def _test_label(n_prompt: int, n_gen: int, n_depth: int) -> str:
    if n_depth > 0:
        depth = f" @ d{n_depth}"
    else:
        depth = ""
    if n_prompt > 0 and n_gen == 0:
        return f"pp{n_prompt}{depth}"
    if n_gen > 0 and n_prompt == 0:
        return f"tg{n_gen}{depth}"
    if n_prompt > 0 and n_gen > 0:
        return f"pg{n_prompt},{n_gen}{depth}"
    return f"unknown{n_prompt},{n_gen}{depth}"


def _record_from_dict(row: dict[str, Any]) -> BenchRecord:
    n_prompt = int(row.get("n_prompt", 0))
    n_gen = int(row.get("n_gen", 0))
    n_depth = int(row.get("n_depth", 0))

    samples_ts = row.get("samples_ts") or []
    samples_ns = row.get("samples_ns") or []

    return BenchRecord(
        n_prompt=n_prompt,
        n_gen=n_gen,
        n_depth=n_depth,
        test_label=_test_label(n_prompt, n_gen, n_depth),
        avg_ts=float(row["avg_ts"]),
        stddev_ts=float(row.get("stddev_ts", 0.0)),
        avg_ns=int(row["avg_ns"]),
        stddev_ns=int(row.get("stddev_ns", 0)),
        n_batch=int(row.get("n_batch", 0)),
        n_ubatch=int(row.get("n_ubatch", 0)),
        n_threads=int(row.get("n_threads", 0)),
        n_gpu_layers=int(row.get("n_gpu_layers", 0)),
        type_k=str(row.get("type_k", "")),
        type_v=str(row.get("type_v", "")),
        split_mode=str(row.get("split_mode", "")),
        main_gpu=int(row.get("main_gpu", 0)),
        no_kv_offload=bool(int(row.get("no_kv_offload", 0))),
        flash_attn=bool(int(row.get("flash_attn", 0))),
        use_mmap=bool(int(row.get("use_mmap", 1))),
        embeddings=bool(int(row.get("embeddings", 0))),
        cpu_mask=str(row.get("cpu_mask", "")),
        cpu_strict=bool(int(row.get("cpu_strict", 0))),
        poll=int(row.get("poll", 0)),
        tensor_split=str(row.get("tensor_split", "")),
        backends=str(row.get("backends", "")),
        model_filename=str(row.get("model_filename", "")),
        model_type=str(row.get("model_type", "")),
        model_size=int(row.get("model_size", 0)),
        model_n_params=int(row.get("model_n_params", 0)),
        cpu_info=str(row.get("cpu_info", "")),
        gpu_info=str(row.get("gpu_info", "")),
        build_commit=str(row.get("build_commit", "")),
        build_number=int(row.get("build_number", 0)),
        test_time=str(row.get("test_time", "")),
        samples_ts=[float(x) for x in samples_ts],
        samples_ns=[int(x) for x in samples_ns],
        raw=dict(row),
    )


def parse_bench_json(text: str) -> list[BenchRecord]:
    """Parse stdout from ``llama-bench -o json``."""
    data = json.loads(text)
    if not isinstance(data, list):
        msg = "expected JSON array from llama-bench"
        raise ValueError(msg)
    return [_record_from_dict(row) for row in data]
