"""Progress callbacks for optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProgressEvent:
    phase: str
    message: str
    percent: float | None = None


class ProgressCallback(Protocol):
    def on_progress(self, event: ProgressEvent) -> None: ...

    def on_log(self, line: str) -> None: ...


class NullProgress:
    def on_progress(self, event: ProgressEvent) -> None:
        pass

    def on_log(self, line: str) -> None:
        pass
