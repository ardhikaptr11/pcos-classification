def parse(subparsers, prog: str = "uv run python main.py deploy"):
    deploy_parser = subparsers.add_parser(
        "deploy", help="Predict locally or serve API", prog=prog
    )

    deploy_subs = deploy_parser.add_subparsers(dest="command")

    # Downloading model
    download_parser = deploy_subs.add_parser(
        "download-model", help="Download model artifacts", prog=prog
    )
    download_parser.add_argument(
        "-S",
        "--source",
        choices=["dagshub", "gdrive"],
        default="gdrive",
        help="Source of model",
    )

    # Serving
    deploy_subs.add_parser("serve", help="Run FastAPI server locally")

    # Inference/Prediction
    predict_parser = deploy_subs.add_parser("predict", help="Run local inference")
    predict_parser.add_argument(
        "-M",
        "--model-source",
        type=str,
        default="drive",
        choices=["drive", "hub"],
        help="Source of model",
    )
    predict_parser.add_argument(
        "-D",
        "--data",
        type=str,
        default="data/sample_negative.json",
        help="Input data in JSON string format",
    )

    return deploy_parser
