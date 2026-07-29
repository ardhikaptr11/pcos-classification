def parse(subparsers, prog: str = "uv run python main.py preprocess"):
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Clean and merge raw datasets", prog=prog
    )

    preprocess_parser.add_argument(
        "primary", type=str, help="Path to the primary dataset file (.csv)"
    )

    preprocess_parser.add_argument(
        "secondary",
        type=str,
        nargs="?",
        default=None,
        help="Path to the secondary dataset file (.xlsx/.csv) [Optional]",
    )

    preprocess_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dataset/processed/pcos_data_preprocessed.csv",
        help="Output result file name (.csv)",
    )

    return preprocess_parser
