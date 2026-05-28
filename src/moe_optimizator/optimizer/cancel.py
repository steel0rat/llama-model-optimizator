"""Cooperative cancellation for long optimization runs."""

from __future__ import annotations

import threading


class OptimizationCancelled(Exception):
    """Raised when the user stops optimization."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def reset(self) -> None:
        self._event.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.is_cancelled:
            raise OptimizationCancelled("Оптимизация остановлена пользователем")
