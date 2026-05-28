"""Export launch scripts for llama-server (.bat / .sh)."""

from __future__ import annotations

from pathlib import Path

from moe_optimizator.bench.models import BenchRecord


def _quote(path: Path | str) -> str:
    s = str(path)
    return f'"{s}"' if " " in s else s


def _server_args(
    record: BenchRecord,
    *,
    model: Path,
    ctx_size: int,
    host: str,
    port: int,
    parallel: int,
) -> str:
    model_path = Path(model).resolve()
    fa = "on" if record.flash_attn else "off"
    nkvo = " --no-kv-offload" if record.no_kv_offload else ""
    return (
        f"-m {_quote(model_path)} "
        f"-c {ctx_size} "
        f"-np {parallel} "
        f"-ngl {record.n_gpu_layers} "
        f"-t {record.n_threads} "
        f"-tb {record.n_threads} "
        f"-b {record.n_batch} "
        f"-ub {record.n_ubatch} "
        f"-fa {fa} "
        f"-ctk {record.type_k} "
        f"-ctv {record.type_v}"
        f"{nkvo} "
        f"--host {host} "
        f"--port {port}"
    )


def write_server_bat(
    path: Path,
    *,
    llama_server: Path | str,
    model: Path | str,
    record: BenchRecord,
    ctx_size: int,
    host: str = "0.0.0.0",
    port: int = 11434,
    parallel: int = 1,
) -> Path:
    server = Path(llama_server).resolve()
    model_path = Path(model).resolve()
    args = _server_args(
        record, model=model_path, ctx_size=ctx_size, host=host, port=port, parallel=parallel
    )
    content = f"""@echo off
chcp 65001 >nul
title llama-server (MOE Optimizator)
cd /d "%~dp0"

echo Starting llama-server...
echo   Model:   {model_path.name}
echo   Host:    {host}
echo   Port:    {port}
echo   Context: {ctx_size}
echo.

{_quote(server)} {args}

if errorlevel 1 (
  echo.
  echo Server exited with error %%errorlevel%%
  pause
)
"""
    out = path.with_suffix(".bat")
    out.write_text(content, encoding="utf-8")
    return out


def write_server_sh(
    path: Path,
    *,
    llama_server: Path | str,
    model: Path | str,
    record: BenchRecord,
    ctx_size: int,
    host: str = "0.0.0.0",
    port: int = 11434,
    parallel: int = 1,
) -> Path:
    server = Path(llama_server).resolve()
    model_path = Path(model).resolve()
    args = _server_args(
        record, model=model_path, ctx_size=ctx_size, host=host, port=port, parallel=parallel
    )
    content = f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Starting llama-server..."
echo "  Model:   {model_path.name}"
echo "  Host:    {host}"
echo "  Port:    {port}"
echo "  Context: {ctx_size}"
echo

exec {_quote(server)} {args}
"""
    out = path.with_suffix(".sh")
    out.write_text(content, encoding="utf-8")
    out.chmod(out.stat().st_mode | 0o111)
    return out
