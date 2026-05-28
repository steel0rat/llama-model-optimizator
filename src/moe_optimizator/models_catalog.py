"""Scan directories for GGUF models."""

from __future__ import annotations

from pathlib import Path


def list_gguf_models(directory: Path) -> list[Path]:
    """Return sorted ``.gguf`` files under *directory* (recursive)."""
    root = Path(directory)
    if not root.is_dir():
        return []
    files = sorted(root.rglob("*.gguf"), key=lambda p: p.name.lower())
    return [p.resolve() for p in files if p.is_file()]


def format_model_choice(path: Path) -> str:
    """Label for select: filename + parent hint."""
    name = path.name
    parent = path.parent.name
    if parent and parent != path.anchor:
        return f"{name}  ({parent})"
    return name
