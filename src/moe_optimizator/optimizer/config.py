"""Optimization configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SearchSpace:
    """Grid values for phase 1 inference tuning (comma-separated → bench sweeps)."""

    n_gpu_layers: list[int] = field(default_factory=lambda: [99])
    n_threads: list[int] = field(default_factory=lambda: [4, 8, 16])
    n_batch: list[int] = field(default_factory=lambda: [2048])
    n_ubatch: list[int] = field(default_factory=lambda: [512])
    flash_attn: list[int] = field(default_factory=lambda: [0, 1])
    cache_type_k: list[str] = field(default_factory=lambda: ["f16", "q8_0"])
    cache_type_v: list[str] = field(default_factory=lambda: ["f16", "q8_0"])
    no_kv_offload: list[int] = field(default_factory=lambda: [0, 1])
    use_mmap: list[int] = field(default_factory=lambda: [1])
    n_cpu_moe: list[int] = field(default_factory=lambda: [0])


@dataclass
class OptimizationConfig:
    """User-facing optimization settings."""

    model: Path
    llama_bench: Path
    ctx_min: int = 4096
    ctx_max: int = 131072
    ctx_step: int = 4096
    prompt_tokens: int = 512
    gen_tokens: int = 128
    server_parallel: int = 1
    search: SearchSpace = field(default_factory=SearchSpace)
    bench_repetitions: int = 3
    output_dir: Path | None = None
