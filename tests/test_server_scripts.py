from pathlib import Path

from moe_optimizator.bench.parser import parse_bench_json
from moe_optimizator.optimizer.server_scripts import write_server_sh

FIXTURE = Path(__file__).parent / "fixtures" / "bench_sample.json"


def test_write_server_sh(tmp_path: Path):
    rec = parse_bench_json(FIXTURE.read_text())[0]
    out = write_server_sh(
        tmp_path / "start",
        llama_server=tmp_path / "llama-server",
        model=tmp_path / "model.gguf",
        record=rec,
        ctx_size=32768,
        host="127.0.0.1",
        port=8080,
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "--host 127.0.0.1" in text
    assert "--port 8080" in text
    assert out.stat().st_mode & 0o111
