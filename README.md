# MOE Optimizator

Python-приложение с поддержкой сборки в standalone `.exe` для Windows.

## Быстрый старт

```bash
make install-gui
make gui
```

CLI (без GUI): `moe-optimizator optimize -m model.gguf --llama-bench ./llama-bench`

Справка: `make help`.

## Документация

Подробная документация в каталоге **[docs/](docs/README.md)**:

- [Быстрый старт](docs/getting-started.md)
- [Разработка и Makefile](docs/development.md)
- [Сборка Windows exe](docs/building-windows-exe.md)
- [Структура проекта](docs/project-structure.md)
- [Процесс работы (согласование фич)](docs/process.md)

## Сборка `.exe` (Windows)

**С Mac/Linux** — через GitHub Actions (нужен `gh` и репозиторий на GitHub):

```bash
brew install gh && gh auth login
make build-exe-remote    # → dist/moe-optimizator.exe
```

Подробнее: [docs/ci.md](docs/ci.md).

**На Windows** локально:

```powershell
make build-exe
```

## Участие

См. [CONTRIBUTING.md](CONTRIBUTING.md).
