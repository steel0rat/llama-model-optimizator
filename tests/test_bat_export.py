from pathlib import Path

from moe_optimizator.bench.parser import parse_bench_json
from moe_optimizator.optimizer.bat_export import write_server_bat

FIXTURE = Path(__file__).parent / "fixtures" / "bench_sample.json"


def test_write_server_bat(tmp_path: Path):
    rec = parse_bench_json(FIXTURE.read_text())[0]
    out = write_server_bat(
        tmp_path / "start",
        llama_server=tmp_path / "llama-server.exe",
        model=tmp_path / "model.gguf",
        record=rec,
        ctx_size=65536,
        host="0.0.0.0",
        port=11434,
    )
    text = out.read_text(encoding="utf-8")
    assert "--host 0.0.0.0" in text
    assert "--port 11434" in text
    assert "-c 65536" in text
    assert "-np 1" in text
    assert "model.gguf" in text
