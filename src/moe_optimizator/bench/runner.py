"""Run llama-bench as subprocess."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from moe_optimizator.bench.models import BenchRecord, BenchRunError
from moe_optimizator.bench.parser import parse_bench_json

if TYPE_CHECKING:
    from moe_optimizator.optimizer.cancel import CancelToken


def run_llama_bench(
    executable: Path | str,
    *,
    model: Path | str,
    extra_args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_sec: float | None = None,
    cancel: CancelToken | None = None,
) -> list[BenchRecord]:
    """
    Run llama-bench with ``-o json`` and return parsed records.

    ``extra_args`` — flags without executable/model (e.g. ``-ngl 99 -d 4096``).
    """
    if cancel is not None:
        cancel.check()

    cmd: list[str] = [
        str(executable),
        "-m",
        str(model),
        "-o",
        "json",
        *(extra_args or ()),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=None if env is None else dict(env),
    )
    deadline: float | None = None
    if timeout_sec is not None:
        deadline = time.monotonic() + timeout_sec

    try:
        while proc.poll() is None:
            if cancel is not None and cancel.is_cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                from moe_optimizator.optimizer.cancel import OptimizationCancelled

                raise OptimizationCancelled("llama-bench прерван")
            if deadline is not None and time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise BenchRunError("llama-bench timeout", returncode=-1, stderr="timeout")
            time.sleep(0.15)
        stdout, stderr = proc.communicate()
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        raise

    if proc.returncode != 0:
        raise BenchRunError(
            f"llama-bench failed (exit {proc.returncode})",
            returncode=proc.returncode or 1,
            stderr=stderr or "",
        )
    return parse_bench_json(stdout or "")
