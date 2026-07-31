import os
import sys
from pathlib import Path
from typing import Any

import dagshub
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from common.logger import setup_logger
from env import envs
from training.baseline import train_baseline
from training.tuning import train_tuning
from training.utils import check_champion, validate_and_promote

logger = setup_logger()


def run_training(
    experiment_name: str,
    data_path: str,
    n_trials: int | None = 30,
    use_tuning: bool = False,
    config: dict[str, Any] | None = None,
):
    logger.info("Connecting to DagsHub...")

    repo_owner = envs["DAGSHUB_REPO_OWNER"]
    repo_name = envs["DAGSHUB_REPO_NAME"]

    if not repo_owner or not repo_name:
        raise ValueError("DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME is missing in .env")

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    if not mlflow.active_run() and not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(experiment_name)
        logger.info(f"Set MLflow experiment name to: {experiment_name}")

    file_path = Path(data_path)
    if not file_path.exists():
        logger.error(f"Dataset not found in path: '{data_path}'")
        sys.exit(1)

    model_name = "pcos-xgboost-prod"

    # Check if champion already exists
    has_champion = check_champion(model_name=model_name)

    if has_champion and not use_tuning:
        logger.info(
            "Champion already exists. Skipping baseline model training to save resources."
        )
        return

    logger.info(f"Loading dataset from: {file_path}")
    data = pd.read_csv(file_path)
    X = data.drop(columns=["pcos_yn"])
    y = data["pcos_yn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        f"Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    if not use_tuning:
        logger.info("Starting baseline model training...")
        trained_model = train_baseline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
        )
    else:
        logger.info(f"Starting hyperparameter tuning with n_trials={n_trials}...")
        trained_model = train_tuning(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            n_trials=n_trials,
        )

    validate_and_promote(
        model_name=model_name,
        challenger_model=trained_model,
        X_test=X_test,
        y_test=y_test,
    )
