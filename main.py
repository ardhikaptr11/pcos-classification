import argparse
import sys

from deployment import parser as deploy
from deployment.handler import handle as handle_deploy
from preprocessing import parser as preprocess
from preprocessing.handler import handle as handle_preprocess
from training import parser as train
from training.handler import handle as handle_train


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

    preprocess_parser = preprocess.parse(subparsers=subparsers)
    preprocess_parser.set_defaults(func=handle_preprocess)

    train_parser = train.parse(subparsers=subparsers)
    train_parser.set_defaults(func=handle_train)

    deploy_parser = deploy.parse(subparsers=subparsers)
    deploy_parser.set_defaults(func=handle_deploy)

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
