import argparse
import sys

from dotenv import load_dotenv

from training.train import run_training
from .utils import load_config
from env import envs

load_dotenv()


def handle(args: argparse.Namespace):
    if args.baseline and args.trials is not None:
        print("❌ Error: --trials cannot be used with --baseline", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    data_path = config["data_path"]

    tracking_uri = envs["MLFLOW_TRACKING_URI_LOCAL"] if args.watch == "local" else None

    mode = "baseline" if args.baseline else "tuning"
    cfg = config.get(mode, {})

    if args.baseline:
        print("🚀 Executing Baseline Model Training...")

        run_training(
            experiment_name=experiment_name,
            data_path=data_path,
            tracking_uri=tracking_uri,
            config=cfg,
        )

        print("✅ Model Training Completed!")
    elif args.tuning:
        print("⚙️ Executing Model Training with Hyperparameter Tuning...")

        n_trials = args.trials if args.trials is not None else 50

        run_training(
            use_tuning=True,
            experiment_name=experiment_name,
            data_path=data_path,
            tracking_uri=tracking_uri,
            config=cfg,
            n_trials=n_trials,
        )

        print("✅ Hyperparameter Tuning Completed!")
