from moe_optimizator.settings import UserSettings, load_settings, save_settings


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "moe_optimizator.settings.settings_path",
        lambda: tmp_path / "settings.json",
    )
    save_settings(
        UserSettings(
            models_dir="/models",
            llama_bench="/bin/bench",
            ctx_max=65536,
        )
    )
    loaded = load_settings()
    assert loaded.models_dir == "/models"
    assert loaded.ctx_max == 65536
