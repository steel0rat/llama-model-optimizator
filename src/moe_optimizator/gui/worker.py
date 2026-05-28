"""Background optimization worker (Qt thread)."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from moe_optimizator.optimizer.callbacks import ProgressCallback, ProgressEvent
from moe_optimizator.optimizer.cancel import CancelToken, OptimizationCancelled
from moe_optimizator.optimizer.config import OptimizationConfig
from moe_optimizator.optimizer.pipeline import run_optimization


class OptimizationWorker(QThread):
    log_line = Signal(str)
    progress = Signal(str, float)
    finished_ok = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        config: OptimizationConfig,
        cancel: CancelToken,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._cancel = cancel

    def run(self) -> None:
        progress = _ThreadProgress(self)
        try:
            result = run_optimization(
                self._config,
                progress=progress,
                cancel=self._cancel,
            )
            if not self._cancel.is_cancelled:
                self.finished_ok.emit(result)
        except OptimizationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _ThreadProgress(ProgressCallback):
    def __init__(self, worker: OptimizationWorker) -> None:
        self._worker = worker

    def on_progress(self, event: ProgressEvent) -> None:
        pct = event.percent if event.percent is not None else -1.0
        self._worker.progress.emit(event.message, pct)

    def on_log(self, line: str) -> None:
        self._worker.log_line.emit(line)
