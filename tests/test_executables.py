from pathlib import Path

from moe_optimizator.executables import llama_bench_not_found_message, resolve_executable


def test_resolve_executable_by_path(tmp_path: Path) -> None:
    exe = tmp_path / "llama-bench"
    exe.write_text("", encoding="utf-8")
    assert resolve_executable(str(exe)) == exe.resolve()


def test_resolve_executable_empty() -> None:
    assert resolve_executable("") is None
    assert resolve_executable("   ") is None


def test_llama_bench_not_found_message() -> None:
    msg = llama_bench_not_found_message("llama-bench")
    assert "Не найден" in msg
    assert "llama.cpp" in msg

