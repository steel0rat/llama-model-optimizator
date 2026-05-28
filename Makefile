# MOE Optimizator — команды разработки и сборки
# Справка: make help

.DEFAULT_GOAL := help

PYTHON       ?= python3
VENV         := .venv
PROJECT      := moe-optimizator
DIST_EXE     := dist/moe-optimizator.exe
SPEC         := pyinstaller/moe-optimizator.spec

ifeq ($(OS),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
else
	VENV_BIN := $(VENV)/bin
endif

PY   := $(VENV_BIN)/python
PIP  := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
RUFF := $(VENV_BIN)/ruff

.PHONY: help venv install install-dev install-build install-all \
	run gui test lint lint-fix format check build-wheel build-app build-exe build-exe-remote clean distclean ci

help: ## Показать список команд
	@echo "MOE Optimizator — доступные цели make:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Создать виртуальное окружение .venv
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@echo "Окружение: $(VENV)"

install: venv ## Установить пакет в editable-режиме
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: venv ## Установить пакет + dev-зависимости (pytest, ruff, gui)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-gui: venv ## Установить пакет + GUI (PySide6)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[gui]"

gui: install-gui ## Запустить графический интерфейс
	$(PY) -m moe_optimizator.gui

install-build: venv ## Установить пакет + PyInstaller
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[build]"

install-all: venv ## Установить пакет + dev + build extras
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,build]"

run: install-dev ## Запустить CLI приложения
	$(PY) -m moe_optimizator $(ARGS)

test: install-dev ## Запустить тесты
	$(PYTEST) -q $(PYTEST_ARGS)

lint: install-dev ## Проверка кода (ruff)
	$(RUFF) check src tests

lint-fix: install-dev ## Автоисправление ruff
	$(RUFF) check --fix src tests

format: install-dev ## Форматирование ruff
	$(RUFF) format src tests

check: lint test ## Линтер + тесты (локальный CI)

build-wheel: install ## Собрать wheel в dist/
	$(PIP) install build
	$(PY) -m build --outdir dist

build-app: install-all ## Собрать приложение под текущую ОС (dist/moe-optimizator)
	$(PIP) install -e ".[gui,build]"
	$(PY) -m PyInstaller --noconfirm --clean pyinstaller/moe-optimizator.spec
	@echo "Готово: dist/moe-optimizator (или dist/moe-optimizator.exe на Windows)"

build-exe: install-all ## Собрать Windows .exe (только на Windows; см. build-exe-remote)
ifeq ($(OS),Windows_NT)
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_windows.ps1
else
	@echo "Локально .exe на $(shell uname -s) собрать нельзя: PyInstaller упаковывает"
	@echo "бинарники целевой ОС (Windows DLL + Qt), кросс-сборки macOS→.exe нет."
	@echo ""
	@echo "  • macOS/Linux GUI:  make build-app        → dist/moe-optimizator"
	@echo "  • Windows .exe:     make build-exe-remote → dist/moe-optimizator.exe (через GitHub Actions)"
	@echo "  • или Windows VM / машина: make build-exe"
	@exit 1
endif

build-exe-remote: ## Собрать .exe в GitHub Actions и скачать в dist/ (нужны gh + push в репо)
	bash scripts/fetch_windows_exe.sh

clean: ## Удалить артефакты сборки
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

distclean: clean ## clean + удалить .venv
	rm -rf $(VENV)

ci: check ## Алиас для проверки перед коммитом
