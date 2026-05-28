"""PySide6 main window."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from moe_optimizator.executables import (
    discover_llama_bench,
    llama_bench_not_found_message,
    resolve_executable,
)
from moe_optimizator.gui.worker import OptimizationWorker
from moe_optimizator.models_catalog import format_model_choice, list_gguf_models
from moe_optimizator.optimizer.cancel import CancelToken
from moe_optimizator.optimizer.config import OptimizationConfig
from moe_optimizator.optimizer.pipeline import OptimizationResult
from moe_optimizator.optimizer.server_scripts import write_server_bat, write_server_sh
from moe_optimizator.optimizer.table_data import (
    best_record,
    ranking_table_rows,
    records_table_rows,
)
from moe_optimizator.settings import UserSettings, load_settings, save_settings


class ExportDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        *,
        prefs: UserSettings,
        default_bat: str,
        default_sh: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Скрипт запуска llama-server")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.host = QLineEdit(prefs.host)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(prefs.port)
        self.bat_path = QLineEdit(default_bat)
        self.sh_path = QLineEdit(default_sh)
        form.addRow("Host", self.host)
        form.addRow("Port", self.port)
        form.addRow("Файл .bat", self.bat_path)
        form.addRow("Файл .sh", self.sh_path)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        btn_bat = buttons.addButton("Сохранить .bat", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_sh = buttons.addButton("Сохранить .sh", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        btn_bat.clicked.connect(self._accept_bat)
        btn_sh.clicked.connect(self._accept_sh)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.export_kind = ""

    def _accept_bat(self) -> None:
        self.export_kind = "bat"
        self.accept()

    def _accept_sh(self) -> None:
        self.export_kind = "sh"
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MOE Optimizator")
        self.resize(1280, 840)

        self._prefs = load_settings()
        self._model_paths: dict[str, Path] = {}
        self._cancel = CancelToken()
        self._worker: OptimizationWorker | None = None
        self._result: OptimizationResult | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        self._status = QLabel("Готов к работе")
        self._status.setObjectName("statusLabel")
        header.addWidget(self._status)
        header.addStretch()
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- left: settings ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)

        paths_box = QGroupBox("Модель и инструменты")
        paths_form = QGridLayout(paths_box)

        self._models_dir = QLineEdit(self._prefs.models_dir)
        btn_dir = QPushButton("…")
        btn_dir.setFixedWidth(36)
        btn_dir.clicked.connect(self._browse_models_dir)
        paths_form.addWidget(QLabel("Каталог GGUF"), 0, 0)
        row0 = QHBoxLayout()
        row0.addWidget(self._models_dir)
        row0.addWidget(btn_dir)
        paths_form.addLayout(row0, 0, 1)

        self._model_combo = QComboBox()
        paths_form.addWidget(QLabel("Модель"), 1, 0)
        paths_form.addWidget(self._model_combo, 1, 1)

        bench_default = self._prefs.llama_bench
        resolved_bench = resolve_executable(bench_default) or discover_llama_bench()
        self._bench = QLineEdit(
            str(resolved_bench) if resolved_bench else bench_default,
        )
        self._bench.setPlaceholderText("Путь или имя в PATH (например build/bin/llama-bench)")
        btn_bench = QPushButton("…")
        btn_bench.setFixedWidth(36)
        btn_bench.clicked.connect(lambda: self._browse_file(self._bench, "llama-bench"))
        paths_form.addWidget(QLabel("llama-bench"), 2, 0)
        row_b = QHBoxLayout()
        row_b.addWidget(self._bench)
        row_b.addWidget(btn_bench)
        paths_form.addLayout(row_b, 2, 1)

        self._server = QLineEdit(self._prefs.llama_server)
        btn_srv = QPushButton("…")
        btn_srv.setFixedWidth(36)
        btn_srv.clicked.connect(lambda: self._browse_file(self._server, "llama-server"))
        paths_form.addWidget(QLabel("llama-server"), 3, 0)
        row_s = QHBoxLayout()
        row_s.addWidget(self._server)
        row_s.addWidget(btn_srv)
        paths_form.addLayout(row_s, 3, 1)

        btn_refresh = QPushButton("Обновить список моделей")
        btn_refresh.clicked.connect(self._refresh_models)
        paths_form.addWidget(btn_refresh, 4, 0, 1, 2)

        left_layout.addWidget(paths_box)

        ctx_box = QGroupBox("Контекст (фаза 1)")
        ctx_form = QFormLayout(ctx_box)
        self._ctx_min = QSpinBox()
        self._ctx_min.setRange(512, 2_000_000)
        self._ctx_min.setSingleStep(512)
        self._ctx_min.setValue(self._prefs.ctx_min)
        self._ctx_max = QSpinBox()
        self._ctx_max.setRange(512, 2_000_000)
        self._ctx_max.setSingleStep(512)
        self._ctx_max.setValue(self._prefs.ctx_max)
        self._ctx_step = QSpinBox()
        self._ctx_step.setRange(256, 65536)
        self._ctx_step.setValue(self._prefs.ctx_step)
        ctx_form.addRow("ctx_min", self._ctx_min)
        ctx_form.addRow("ctx_max", self._ctx_max)
        ctx_form.addRow("ctx_step", self._ctx_step)
        left_layout.addWidget(ctx_box)

        actions = QVBoxLayout()
        self._btn_start = QPushButton("Запустить оптимизацию")
        self._btn_start.setObjectName("primaryBtn")
        self._btn_start.clicked.connect(self._start)
        self._btn_stop = QPushButton("Стоп")
        self._btn_stop.setObjectName("stopBtn")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        row_a = QHBoxLayout()
        row_a.addWidget(self._btn_start)
        row_a.addWidget(self._btn_stop)
        actions.addLayout(row_a)

        btn_export = QPushButton("Создать скрипт llama-server")
        btn_export.clicked.connect(self._export_script)
        actions.addWidget(btn_export)
        left_layout.addLayout(actions)
        left_layout.addStretch()

        splitter.addWidget(left)

        # --- right: progress, log, tables ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)

        prog_box = QGroupBox("Прогресс")
        prog_layout = QVBoxLayout(prog_box)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._phase = QLabel("")
        self._phase.setObjectName("phaseLabel")
        prog_layout.addWidget(self._progress)
        prog_layout.addWidget(self._phase)
        right_layout.addWidget(prog_box)

        log_box = QGroupBox("Журнал")
        log_layout = QVBoxLayout(log_box)
        self._log = QTextEdit()
        self._log.setObjectName("logView")
        self._log.setReadOnly(True)
        log_layout.addWidget(self._log)
        right_layout.addWidget(log_box, stretch=2)

        tabs = QTabWidget()
        self._rank_table = QTableWidget()
        self._detail_table = QTableWidget()
        tabs.addTab(self._rank_table, "Рейтинг конфигураций")
        tabs.addTab(self._detail_table, "Все прогоны")
        right_layout.addWidget(tabs, stretch=3)

        splitter.addWidget(right)
        splitter.setSizes([380, 860])
        root.addWidget(splitter, stretch=1)

        if self._prefs.models_dir:
            self._refresh_models()

    def _persist(self) -> None:
        self._prefs = UserSettings(
            models_dir=self._models_dir.text().strip(),
            llama_bench=self._bench.text().strip(),
            llama_server=self._server.text().strip(),
            ctx_min=self._ctx_min.value(),
            ctx_max=self._ctx_max.value(),
            ctx_step=self._ctx_step.value(),
            host=self._prefs.host,
            port=self._prefs.port,
            last_model_label=self._model_combo.currentText(),
            export_bat_path=self._prefs.export_bat_path,
            export_sh_path=self._prefs.export_sh_path,
        )
        save_settings(self._prefs)

    def _append_log(self, line: str) -> None:
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _browse_models_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Каталог моделей")
        if path:
            self._models_dir.setText(path)
            self._refresh_models()

    def _browse_file(self, field: QLineEdit, title: str) -> None:
        filt = "Executables (*.exe);;All (*)" if sys.platform == "win32" else "All (*)"
        path, _ = QFileDialog.getOpenFileName(self, title, field.text(), filt)
        if path:
            field.setText(path)
            self._persist()

    def _refresh_models(self) -> None:
        root = Path(self._models_dir.text().strip())
        paths = list_gguf_models(root)
        self._model_paths = {format_model_choice(p): p for p in paths}
        self._model_combo.clear()
        self._model_combo.addItems(list(self._model_paths.keys()))
        if self._prefs.last_model_label in self._model_paths:
            self._model_combo.setCurrentText(self._prefs.last_model_label)
        self._append_log(f"Найдено моделей: {len(paths)}")
        self._persist()

    def _fill_table(self, table: QTableWidget, rows: list[dict]) -> None:
        table.clear()
        if not rows:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        cols = list(rows[0].keys())
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                val = row.get(col, "")
                table.setItem(r, c, QTableWidgetItem(str(val)))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    def _start(self) -> None:
        label = self._model_combo.currentText()
        if not label or label not in self._model_paths:
            QMessageBox.warning(self, "Модель", "Выберите модель из списка.")
            return
        bench_spec = self._bench.text().strip()
        bench = resolve_executable(bench_spec)
        if bench is None:
            QMessageBox.critical(
                self,
                "llama-bench",
                llama_bench_not_found_message(bench_spec),
            )
            return

        self._persist()
        self._log.clear()
        self._cancel = CancelToken()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status.setText("Оптимизация…")
        self._progress.setValue(0)

        config = OptimizationConfig(
            model=self._model_paths[label],
            llama_bench=bench,
            ctx_min=self._ctx_min.value(),
            ctx_max=self._ctx_max.value(),
            ctx_step=self._ctx_step.value(),
            output_dir=Path("optimization_out"),
        )
        self._worker = OptimizationWorker(config, self._cancel, self)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _stop(self) -> None:
        self._cancel.cancel()
        self._append_log("Запрошена остановка…")

    def _on_progress(self, message: str, percent: float) -> None:
        self._phase.setText(message)
        if percent >= 0:
            self._progress.setValue(int(min(100, max(0, percent))))

    def _on_finished(self, result: OptimizationResult) -> None:
        self._result = result
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText(f"Готово · ctx_max = {result.ctx_max:,}")
        self._progress.setValue(100)
        self._fill_table(self._rank_table, ranking_table_rows(result.ranked))
        self._fill_table(self._detail_table, records_table_rows(result.records))
        self._append_log(f"Отчёт: {result.report_path}")

    def _on_failed(self, msg: str) -> None:
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText("Ошибка")
        self._append_log(f"ОШИБКА: {msg}")
        QMessageBox.critical(self, "Ошибка", msg)

    def _on_cancelled(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status.setText("Остановлено")
        self._append_log("— Оптимизация остановлена —")

    def _export_script(self) -> None:
        if not self._result or not self._result.ranked:
            QMessageBox.warning(self, "Экспорт", "Сначала выполните оптимизацию.")
            return
        rec = best_record(self._result.records, self._result.ranked, self._result.ctx_max)
        if not rec:
            QMessageBox.warning(self, "Экспорт", "Нет данных для экспорта.")
            return

        dlg = ExportDialog(
            self,
            prefs=self._prefs,
            default_bat=self._prefs.export_bat_path or str(Path.cwd() / "start-llama-server.bat"),
            default_sh=self._prefs.export_sh_path or str(Path.cwd() / "start-llama-server.sh"),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        srv = self._server.text().strip()
        if not srv:
            QMessageBox.warning(self, "Экспорт", "Укажите путь к llama-server.")
            return
        model = self._model_paths.get(self._model_combo.currentText())
        if not model:
            return

        host = dlg.host.text().strip() or "0.0.0.0"
        port = dlg.port.value()
        self._prefs.host = host
        self._prefs.port = port

        try:
            if dlg.export_kind == "bat":
                out = write_server_bat(
                    Path(dlg.bat_path.text()),
                    llama_server=Path(srv),
                    model=model,
                    record=rec,
                    ctx_size=self._result.ctx_max,
                    host=host,
                    port=port,
                )
                self._prefs.export_bat_path = str(out)
            else:
                out = write_server_sh(
                    Path(dlg.sh_path.text()),
                    llama_server=Path(srv),
                    model=model,
                    record=rec,
                    ctx_size=self._result.ctx_max,
                    host=host,
                    port=port,
                )
                self._prefs.export_sh_path = str(out)
            save_settings(self._prefs)
            QMessageBox.information(self, "Сохранено", str(out))
        except OSError as exc:
            QMessageBox.critical(self, "Ошибка", str(exc))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker and self._worker.isRunning():
            self._cancel.cancel()
            self._worker.wait(3000)
        self._persist()
        super().closeEvent(event)
