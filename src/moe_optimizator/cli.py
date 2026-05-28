import argparse
import sys
from pathlib import Path

from moe_optimizator.executables import discover_llama_bench, resolve_executable
from moe_optimizator.optimizer.config import OptimizationConfig
from moe_optimizator.optimizer.pipeline import run_optimization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe-optimizator",
        description="Подбор параметров llama-server через llama-bench",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("gui", help="Графический интерфейс (по умолчанию)")

    opt = sub.add_parser(
        "optimize",
        help="Двухфазная оптимизация: inference, затем ctx_max (CLI)",
    )
    opt.add_argument("-m", "--model", type=Path, required=True, help="Путь к GGUF")
    opt.add_argument(
        "--llama-bench",
        type=Path,
        default=Path("llama-bench"),
        help="Путь к llama-bench",
    )
    opt.add_argument("--ctx-min", type=int, default=4096)
    opt.add_argument("--ctx-max", type=int, default=131072)
    opt.add_argument("--ctx-step", type=int, default=4096)
    opt.add_argument("--prompt-tokens", type=int, default=512)
    opt.add_argument("--gen-tokens", type=int, default=128)
    opt.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("optimization_out"),
        help="Каталог отчёта",
    )
    opt.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Пропустить фазу 1 (подбор inference), взять значения по умолчанию",
    )
    opt.add_argument(
        "--skip-ctx-search",
        type=int,
        metavar="CTX",
        help="Пропустить фазу 2 (поиск ctx_max), использовать заданное значение",
    )
    opt.add_argument(
        "--skip-phase1",
        type=int,
        metavar="CTX",
        help=argparse.SUPPRESS,
    )

    return parser


def cmd_optimize(args: argparse.Namespace) -> int:
    bench = resolve_executable(str(args.llama_bench))
    if bench is None:
        bench = discover_llama_bench()
    if bench is None:
        print(
            "llama-bench не найден. Укажите --llama-bench /полный/путь "
            "или добавьте бинарник в PATH.",
            file=sys.stderr,
        )
        return 1

    config = OptimizationConfig(
        model=args.model,
        llama_bench=bench,
        ctx_min=args.ctx_min,
        ctx_max=args.ctx_max,
        ctx_step=args.ctx_step,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        server_parallel=1,
        output_dir=args.output_dir,
    )

    def _log(line: str) -> None:
        print(line)

    class _CliProgress:
        def on_progress(self, event) -> None:
            pct = f" [{event.percent:.0f}%]" if event.percent is not None else ""
            print(f"{event.phase}{pct}: {event.message}")

        def on_log(self, line: str) -> None:
            _log(line)

    skip_ctx = args.skip_ctx_search
    if args.skip_phase1 is not None:
        skip_ctx = args.skip_phase1

    result = run_optimization(
        config,
        skip_tuning=args.skip_tuning,
        skip_ctx_search=skip_ctx,
        progress=_CliProgress(),
    )
    print(f"Записей: {len(result.records)}, конфигураций: {len(result.ranked)}")
    print(f"Отчёт: {result.report_path}")
    if result.ranked:
        print("Лучшие метрики:", result.ranked[0][1])
    return 0


def cmd_gui() -> int:
    try:
        from moe_optimizator.gui.app import run_gui
    except ImportError:
        print(
            "GUI не установлен. Выполните: pip install -e '.[gui]'",
            file=sys.stderr,
        )
        return 1
    run_gui()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "optimize":
        return cmd_optimize(args)

    if args.command in (None, "gui"):
        return cmd_gui()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
