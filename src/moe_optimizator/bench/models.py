"""Data models for llama-bench JSON output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchRecord:
    """One row from llama-bench (-o json)."""

    # Test shape
    n_prompt: int
    n_gen: int
    n_depth: int
    test_label: str

    # Throughput
    avg_ts: float
    stddev_ts: float
    avg_ns: int
    stddev_ns: int

    # Config (inference)
    n_batch: int
    n_ubatch: int
    n_threads: int
    n_gpu_layers: int
    type_k: str
    type_v: str
    split_mode: str
    main_gpu: int
    no_kv_offload: bool
    flash_attn: bool
    use_mmap: bool
    embeddings: bool
    cpu_mask: str
    cpu_strict: bool
    poll: int
    tensor_split: str

    # Environment
    backends: str
    model_filename: str
    model_type: str
    model_size: int
    model_n_params: int
    cpu_info: str
    gpu_info: str
    build_commit: str
    build_number: int
    test_time: str

    samples_ts: list[float] = field(default_factory=list)
    samples_ns: list[int] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_pp(self) -> bool:
        return self.n_prompt > 0 and self.n_gen == 0

    @property
    def is_tg(self) -> bool:
        return self.n_gen > 0 and self.n_prompt == 0

    @property
    def is_pg(self) -> bool:
        return self.n_prompt > 0 and self.n_gen > 0

    @property
    def depth_tag(self) -> str:
        return "ctx" if self.n_depth > 0 else "cold"

    @property
    def metric_id(self) -> str:
        """Stable id, e.g. ``tg@ctx:n128``."""
        if self.is_pp:
            return f"pp@{self.depth_tag}:p{self.n_prompt}"
        if self.is_tg:
            return f"tg@{self.depth_tag}:n{self.n_gen}"
        if self.is_pg:
            return f"pg@{self.depth_tag}:p{self.n_prompt},n{self.n_gen}"
        return f"unknown@{self.depth_tag}"


class BenchRunError(RuntimeError):
    """llama-bench exited with failure."""

    def __init__(self, message: str, *, returncode: int, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
