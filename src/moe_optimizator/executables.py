"""Resolve external tool executables (llama-bench, llama-server)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _bench_filenames() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("llama-bench.exe", "llama-bench")
    return ("llama-bench",)


def resolve_executable(spec: str) -> Path | None:
    """
  Find an executable by absolute/relative path or by name on ``PATH``.

  Returns resolved path if found, else ``None``.
  """
    text = spec.strip()
    if not text:
        return None

    expanded = Path(text).expanduser()
    if expanded.is_file():
        return expanded.resolve()

    # Relative path from current working directory
    rel = Path(text)
    if rel.is_file():
        return rel.resolve()

    found = shutil.which(text)
    if found:
        return Path(found).resolve()

    if sys.platform == "win32" and not text.lower().endswith(".exe"):
        found = shutil.which(f"{text}.exe")
        if found:
            return Path(found).resolve()

    return None


def discover_llama_bench() -> Path | None:
    """Try PATH and common llama.cpp build output locations."""
    for name in _bench_filenames():
        found = resolve_executable(name)
        if found is not None:
            return found

    home = Path.home()
    search_roots: list[Path] = [
        home / "llama.cpp" / "build" / "bin",
        home / "llama.cpp" / "build",
        home / "src" / "llama.cpp" / "build" / "bin",
        home / "src" / "llama.cpp" / "build",
        Path("/usr/local/bin"),
        Path("/opt/homebrew/bin"),
    ]
    if sys.platform == "win32":
        search_roots.extend(
            [
                home / "llama.cpp" / "build" / "bin" / "Release",
                home / "llama.cpp" / "build" / "bin" / "Debug",
            ]
        )

    for directory in search_roots:
        if not directory.is_dir():
            continue
        for name in _bench_filenames():
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()

    return None


def llama_bench_not_found_message(spec: str) -> str:
    """User-facing hint when llama-bench cannot be resolved."""
    lines = [
        f"Не найден: {spec or '(пусто)'}",
        "",
        "Укажите полный путь к llama-bench (кнопка «…») или соберите из llama.cpp:",
        "  cmake -B build && cmake --build build --target llama-bench",
        "",
        "Бинарник обычно лежит в build/bin/llama-bench",
    ]
    if shutil.which("llama-bench") is None and shutil.which("llama-bench.exe") is None:
        lines.append("Сейчас llama-bench не виден в PATH.")
    return "\n".join(lines)
