from moe_optimizator.cli import build_parser, main


def test_optimize_subcommand_registered():
    parser = build_parser()
    args = parser.parse_args(["optimize", "-m", "x.gguf"])
    assert args.command == "optimize"
    assert args.model.name == "x.gguf"


def test_gui_command_registered():
    parser = build_parser()
    args = parser.parse_args(["gui"])
    assert args.command == "gui"


def test_version_exits_zero():
    try:
        code = main(["--version"])
    except SystemExit as exc:
        code = exc.code
    assert code == 0
