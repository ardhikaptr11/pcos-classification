import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost as mlflow_xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from common.logger import setup_logger
from training.utils import check_champion, validate_and_promote

logger = setup_logger()


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

    return model


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

    def apply_tags():
        is_local_exec = os.getenv("GITHUB_ACTIONS") != "true"
        source = "Local Execution" if is_local_exec else "CI/CD Pipeline"
        commit_sha = os.getenv("GITHUB_SHA", "local_run")[:9]

        mlflow.set_tag("source", source)
        mlflow.set_tag("commit_sha", commit_sha)
        mlflow.set_tag("run_type", "baseline")
        mlflow.set_tag("is_champion", "false")

    if active_run:
        current_run_id = active_run.info.run_id

        apply_tags()
        mlflow.set_tag(key="mlflow.runName", value=run_name)

        trained_model = _train_and_evaluate(
            params=params,
            X_train=X_train,
            y_train=y_train,
            eval_data=eval_data,
            run_id=current_run_id,
        )
    elif env_run_id:
        with mlflow.start_run(run_id=env_run_id) as run:
            apply_tags()
            mlflow.set_tag(key="mlflow.runName", value=run_name)
            
            trained_model = _train_and_evaluate(
                params=params,
                X_train=X_train,
                y_train=y_train,
                eval_data=eval_data,
                run_id=run.info.run_id,
            )
    else:
        with mlflow.start_run(run_name=run_name) as run:
            apply_tags()

            trained_model = _train_and_evaluate(
                params=params,
                X_train=X_train,
                y_train=y_train,
                eval_data=eval_data,
                run_id=run.info.run_id,
            )

    return trained_model


def run_local_train_baseline(
    data_path: str,
    tracking_uri: str,
    config: dict[str, Any] | None = None,
    experiment_name: str = "PCOS Classification",
):
    mlflow.set_tracking_uri(uri=tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow experiment name: {experiment_name}")

    file_path = Path(data_path)
    if not file_path.exists():
        logger.error(f"Dataset not found in path: '{data_path}'")
        sys.exit(1)

    model_name = "pcos-xgboost"

    # Check if champion already exists
    has_champion = check_champion(model_name=model_name)

    if has_champion:
        logger.info(
            "Champion already exists. Skipping baseline model training to save resources."
        )
        return False, None

    data = pd.read_csv(file_path)
    X = data.drop(columns=["pcos_yn"])
    y = data["pcos_yn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    trained_model = train_baseline(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, config=config
    )

    return validate_and_promote(
        model_name=model_name,
        challenger_model=trained_model,
        X_test=X_test,
        y_test=y_test,
    )


# if __name__ == "__main__":
#     run_train_baseline(
#         experiment_name="PCOS Classification",
#         data_path="dataset/pcos_data_preprocessed.csv",
#         tracking_uri=envs["MLFLOW_TRACKING_URI_LOCAL"],
#     )
