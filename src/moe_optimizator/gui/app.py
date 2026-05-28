"""Desktop GUI (PySide6, без веб-сервера)."""

from __future__ import annotations

import sys
from pathlib import Path

from moe_optimizator.gui.main_window import MainWindow


def run_gui() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("MOE Optimizator")
    app.setStyle("Fusion")

    qss_path = Path(__file__).parent / "styles.qss"
    if qss_path.is_file():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
