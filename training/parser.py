def parse(subparsers, prog: str = "uv run python main.py train"):
    train_parser = subparsers.add_parser(
        "train", help="Train a machine learning model", prog=prog
    )

    group = train_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--baseline",
        action="store_true",
        help="Train a baseline model",
    )
    group.add_argument(
        "--tuning", action="store_true", help="Train a model with hyperparameter tuning"
    )

    train_parser.add_argument(
        "-W",
        "--watch",
        type=str,
        choices=["local", "dagshub"],
        default="local",
        help="Watch the experiment",
    )
    train_parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials for hyperparameter tuning",
    )
    train_parser.add_argument(
        "-C",
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the configuration file (.yaml)",
    )

    return train_parser
