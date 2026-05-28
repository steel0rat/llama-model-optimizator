import pytest

pytest.importorskip("PySide6")


def test_main_window_import():
    from moe_optimizator.gui.main_window import MainWindow

    assert MainWindow is not None
