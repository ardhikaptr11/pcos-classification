import argparse
import sys

from dotenv import load_dotenv

from common.logger import setup_logger
from env import envs
from training.baseline import run_local_train_baseline
from training.train import run_training
from training.tuning import run_local_train_tuning

from .utils import load_config

logger = setup_logger()

load_dotenv()


def handle(args: argparse.Namespace):
    if args.baseline and args.trials is not None:
        logger.error("Error: --trials cannot be used with --baseline")
        sys.exit(1)

    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    data_path = config["data_path"]

    mode = "baseline" if args.baseline else "tuning"
    cfg = config.get(mode, {})

    tracking_uri = envs["MLFLOW_TRACKING_URI_LOCAL"]

    if args.baseline:
        if args.watch == "local":
            is_promoted, _ = run_local_train_baseline(
                data_path=data_path,
                tracking_uri=tracking_uri,
                config=cfg,
                experiment_name=experiment_name,
            )

            if is_promoted:
                logger.info("✅ Model promoted as champion!")
        elif args.watch == "remote":
            run_training(
                experiment_name=experiment_name,
                data_path=data_path,
                config=cfg,
            )

        logger.info("✅ Training Completed!")
    elif args.tuning:
        if args.watch == "local":
            is_promoted, run_id = run_local_train_tuning(
                data_path=data_path,
                tracking_uri=tracking_uri,
                config=cfg,
                experiment_name=experiment_name,
            )

            if is_promoted and run_id:
                from common import upload_to_drive

                logger.info("✅ Model promoted to new champion!")
                logger.info("Uploading artifacts to Google Drive...")

                upload_to_drive(run_id=run_id)
        elif args.watch == "remote":
            run_training(
                use_tuning=True,
                experiment_name=experiment_name,
                data_path=data_path,
                config=cfg,
            )

        logger.info("✅ Hyperparameter Tuning Completed!")
