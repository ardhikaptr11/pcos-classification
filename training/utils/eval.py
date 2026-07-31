import os

import mlflow
import mlflow.exceptions
import pandas as pd
from mlflow.models import MetricThreshold
from sklearn.metrics import roc_auc_score

from common.logger import setup_logger

logger = setup_logger()


def _get_champion_score(
    model_name: str, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    try:
        model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")
        y_prob = model.predict_proba(X_test)[:, 1]
        score = float(roc_auc_score(y_test, y_prob))
        logger.info(f"Loaded existing champion model from registry. Score: {score:.4f}")
        return score
    except Exception as e:
        logger.info(
            f"No existing champion found or failed to load: {e}. Defaulting score to 0.0."
        )
        return 0.0


def _validate_thresholds(run_id: str, eval_data: pd.DataFrame):
    logger.info("Starting model evaluation against defined thresholds...")
    result = mlflow.evaluate(
        model=f"runs:/{run_id}/model",
        data=eval_data,
        targets="pcos_yn",
        model_type="classifier",
    )

    thresholds = {
        "roc_auc": MetricThreshold(threshold=0.85, greater_is_better=True),
        "accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True),
    }

    mlflow.validate_evaluation_results(
        candidate_result=result, validation_thresholds=thresholds
    )
    logger.info("Model successfully passed all evaluation thresholds.")


def check_champion(model_name: str) -> bool:
    client = mlflow.MlflowClient()
    try:
        client.get_model_version_by_alias(model_name, "champion")
        logger.info(f"Champion alias found for model '{model_name}'.")
        return True
    except Exception:
        logger.info(f"No champion alias found for model '{model_name}'.")
        return False


def validate_and_promote(
    model_name: str, challenger_model, X_test: pd.DataFrame, y_test: pd.Series
):
    logger.info("Initiating model validation and promotion sequence...")
    champion_score = _get_champion_score(
        model_name=model_name, X_test=X_test, y_test=y_test
    )

    y_prob_challenger = challenger_model.predict_proba(X_test)[:, 1]
    challenger_score = float(roc_auc_score(y_test, y_prob_challenger))
    logger.info(
        f"Challenger score: {challenger_score:.4f} vs Champion score: {champion_score:.4f}"
    )

    if challenger_score <= champion_score:
        logger.info(
            f"Champion retained. Score ({champion_score:.4f}) >= Challenger ({challenger_score:.4f})."
        )
        return False, None

    eval_data = X_test.copy()
    eval_data["pcos_yn"] = y_test

    with mlflow.start_run(run_name="Champion_Updated") as run:
        commit_sha = os.getenv("GITHUB_SHA", "local_run")[:9]
        mlflow.set_tag("source", "CI/CD Pipeline")
        mlflow.set_tag("commit_sha", commit_sha)
        mlflow.set_tag("run_type", "champion_promotion")
        mlflow.set_tag("is_champion", "true")

        mlflow.log_metric("roc_auc", challenger_score)

        model_info = mlflow.xgboost.log_model(
            xgb_model=challenger_model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        logger.info(f"Challenger model logged to run ID: {run.info.run_id}")
        try:
            _validate_thresholds(run.info.run_id, eval_data)

            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=model_info.registered_model_version,
            )
            logger.info(
                f"Validation passed. Champion updated to version {model_info.registered_model_version}. "
                f"New ROC-AUC score: {challenger_score:.4f}"
            )

            return True, run.info.run_id

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Validation failed: {e}")

            return False, None
