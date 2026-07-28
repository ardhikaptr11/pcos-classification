import os
import sys
from pathlib import Path
from typing import Any

import dagshub
import mlflow
import mlflow.xgboost as mlflow_xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils.env import envs


def _train_and_evaluate(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eval_data: pd.DataFrame,
    run_id: Any,
):
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    model_uri = f"runs:/{run_id}/model"
    mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="label",
        model_type="classifier",
    )


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any] | None = None,
):
    config = config or {}
    run_name = config.get("run_name", "XGBoost_Baseline")

    default_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    params = config.get("params", default_params)

    eval_data = X_test.copy()
    eval_data["label"] = y_test

    mlflow_xgb.autolog()

    env_run_id = os.getenv("MLFLOW_RUN_ID")
    active_run = mlflow.active_run()

    if active_run or env_run_id:
        current_run_id = active_run.info.run_id if active_run else env_run_id
        mlflow.set_tag(key="mlflow.runName", value=run_name)

        _train_and_evaluate(
            params=params,
            X_train=X_train,
            y_train=y_train,
            eval_data=eval_data,
            run_id=current_run_id,
        )

    else:
        with mlflow.start_run(run_name=run_name) as run:
            _train_and_evaluate(
                params=params,
                X_train=X_train,
                y_train=y_train,
                eval_data=eval_data,
                run_id=run.info.run_id,
            )


def run_train_baseline(
    experiment_name: str,
    data_path: str,
    tracking_uri: str | None,
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

    train_baseline(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, config=config
    )


if __name__ == "__main__":
    run_train_baseline(
        experiment_name="PCOS Classification",
        data_path="dataset/pcos_data_preprocessed.csv",
        tracking_uri=envs["MLFLOW_TRACKING_URI_LOCAL"],
    )
