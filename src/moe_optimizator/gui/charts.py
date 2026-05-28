"""Matplotlib charts embedded in Qt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from moe_optimizator.optimizer.chart_data import ranking_chart_series

if TYPE_CHECKING:
    from moe_optimizator.optimizer.pipeline import OptimizationResult

# Dark theme aligned with styles.qss
_BG = "#0c0f14"
_AX = "#141a24"
_GRID = "#2e3a50"
_TEXT = "#e8ecf4"
_MUTED = "#a8b4cc"
_COLOR_TG = "#7eb6ff"
_COLOR_PP = "#6bcf7f"
_COLOR_PG = "#e8b86d"


class ChartsPanel(QWidget):
    """Two charts: top configs by tg@ctx; grouped tg/pp/pg for top N."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Выполните оптимизацию — здесь появятся графики.")
        self._placeholder.setObjectName("phaseLabel")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._placeholder)

        self._canvas = None
        self._figure = None

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
            self._figure = None
        self._placeholder.show()

    def plot_result(self, result: OptimizationResult, *, top_n: int = 15) -> None:
        series = ranking_chart_series(result.ranked, limit=top_n)
        if series is None:
            self.clear()
            return

        try:
            import matplotlib
            import numpy as np

            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self.clear()
            self._placeholder.setText(
                "Для графиков установите: pip install -e \".[gui]\" (нужен matplotlib)"
            )
            self._placeholder.show()
            return

        self._placeholder.hide()

        if self._canvas is None:
            self._figure = Figure(figsize=(10, 7), facecolor=_BG)
            self._canvas = FigureCanvasQTAgg(self._figure)
            self.layout().addWidget(self._canvas)
        else:
            self._figure.clear()

        n_grouped = min(8, len(series.labels))
        ax1 = self._figure.add_subplot(2, 1, 1)
        ax2 = self._figure.add_subplot(2, 1, 2)

        for ax in (ax1, ax2):
            ax.set_facecolor(_AX)
            ax.tick_params(colors=_MUTED, labelsize=9)
            ax.xaxis.label.set_color(_TEXT)
            ax.yaxis.label.set_color(_TEXT)
            ax.title.set_color(_TEXT)
            for spine in ax.spines.values():
                spine.set_color(_GRID)
            ax.grid(True, axis="both", alpha=0.25, color=_GRID)

        y_pos = np.arange(len(series.labels))
        ax1.barh(y_pos, series.tg_ctx, color=_COLOR_TG, height=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(series.labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("токенов/с")
        ax1.set_title(
            f"tg — топ-{len(series.labels)} "
            f"(фаза 1: d=0, итог ctx_max={result.ctx_max:,})"
        )

        labels_g = series.labels[:n_grouped]
        x = np.arange(len(labels_g))
        width = 0.25
        ax2.bar(
            x - width,
            series.tg_ctx[:n_grouped],
            width,
            label="tg@ctx",
            color=_COLOR_TG,
        )
        ax2.bar(x, series.pp_ctx[:n_grouped], width, label="pp@ctx", color=_COLOR_PP)
        ax2.bar(
            x + width,
            series.pg_ctx[:n_grouped],
            width,
            label="pg@ctx",
            color=_COLOR_PG,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_g, rotation=35, ha="right", fontsize=7)
        ax2.set_ylabel("токенов/с")
        ax2.set_title(f"Сравнение метрик — топ-{n_grouped}")
        ax2.legend(facecolor=_AX, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)

        self._figure.tight_layout(pad=2.0)
        self._canvas.draw()
