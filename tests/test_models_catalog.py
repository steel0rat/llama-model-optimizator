from pathlib import Path

from moe_optimizator.models_catalog import list_gguf_models


def test_list_gguf_models(tmp_path: Path):
    (tmp_path / "a.gguf").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.gguf").write_bytes(b"y")
    names = [p.name for p in list_gguf_models(tmp_path)]
    assert names == ["a.gguf", "b.gguf"]
