# -*- mode: python ; coding: utf-8 -*-
# pyinstaller pyinstaller/moe-optimizator.spec

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
root = Path(SPECPATH).resolve().parent
src = root / "src"
gui_dir = src / "moe_optimizator" / "gui"

pyside_hidden = collect_submodules("PySide6")
pyside_datas = collect_data_files("PySide6")

a = Analysis(
    [str(gui_dir / "__main__.py")],
    pathex=[str(src)],
    binaries=[],
    datas=pyside_datas + [(str(gui_dir / "styles.qss"), "moe_optimizator/gui")],
    hiddenimports=[
        "moe_optimizator.gui.app",
        "moe_optimizator.gui.main_window",
        "moe_optimizator.gui.worker",
        *pyside_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["nicegui", "uvicorn", "fastapi", "starlette"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="moe-optimizator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
