import os
import sys
from pathlib import Path
from typing import Any, Optional

import dagshub
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from training.baseline import train_baseline
from training.tuning import train_tuning
from env import envs


def run_training(
    experiment_name: str,
    data_path: str,
    tracking_uri: str | None,
    n_trials: Optional[int] = None,
    use_tuning: bool = False,
    config: dict[str, Any] | None = None,
):
    if tracking_uri:
        mlflow.set_tracking_uri(uri=tracking_uri)
    else:
        print("🌐 Connecting to DagsHub...")

        repo_owner = envs["DAGSHUB_REPO_OWNER"]
        repo_name = envs["DAGSHUB_REPO_NAME"]

        if not repo_owner or not repo_name:
            raise ValueError(
                "DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME is missing in .env"
            )

        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    if not mlflow.active_run() and not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(experiment_name)

    file_path = Path(data_path)
    if not file_path.exists():
        print(f"❌ Error: Dataset not found in path: '{data_path}'", file=sys.stderr)
        sys.exit(1)

    data = pd.read_csv(file_path)
    X = data.drop(columns=["pcos_yn"])
    y = data["pcos_yn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if not use_tuning:
        train_baseline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
        )
    else:
        train_tuning(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            n_trials=n_trials,
        )
