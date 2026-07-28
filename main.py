import argparse
import sys

import parsers
from handlers import handle_preprocess, handle_train


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcos-cli",
        description="PCOS End-to-End Machine Learning System CLI",
        usage="uv run python main.py <command> [options]",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Available Commands",
        description="Select a pipeline workflow to execute",
        help="Use 'uv run python main.py <command> --help' for command-specific help.",
    )

    preprocess_parser = parsers.preprocess(subparsers=subparsers)
    preprocess_parser.set_defaults(func=handle_preprocess)

    train_parser = parsers.train(subparsers=subparsers)
    train_parser.set_defaults(func=handle_train)

    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
