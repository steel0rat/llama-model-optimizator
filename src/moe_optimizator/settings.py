"""Persistent user settings (paths, context limits)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


def settings_path() -> Path:
    root = Path.home() / ".config" / "moe-optimizator"
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


@dataclass
class UserSettings:
    models_dir: str = ""
    llama_bench: str = "llama-bench"
    llama_server: str = ""
    ctx_min: int = 4096
    ctx_max: int = 131072
    ctx_step: int = 4096
    host: str = "0.0.0.0"
    port: int = 11434
    last_model_label: str = ""
    export_bat_path: str = ""
    export_sh_path: str = ""


def load_settings() -> UserSettings:
    path = settings_path()
    if not path.is_file():
        return UserSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return UserSettings(**{k: data[k] for k in UserSettings.__dataclass_fields__ if k in data})
    except (json.JSONDecodeError, TypeError, KeyError):
        return UserSettings()


def save_settings(settings: UserSettings) -> None:
    path = settings_path()
    path.write_text(
        json.dumps(asdict(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
