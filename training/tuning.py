import os
import sys
from pathlib import Path
from typing import Any, TypedDict

import dagshub
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow import xgboost as mlflow_xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import estimator_html_repr

from utils import get_sampler
from utils.artifacts import LogFigures
from utils.env import envs
from utils.metrics import calculate_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Labels(TypedDict):
    true: pd.Series
    pred: pd.Series
    prob: np.ndarray


def _log_artifacts(
    best_params: dict[str, Any],
    opt_target_name: str,
    sampler_name: str,
    best_value: float,
    df: pd.DataFrame,
    labels: Labels,
    best_model: Any,
):
    mlflow.log_params(best_params)
    mlflow.log_param("optimization_target", opt_target_name)
    mlflow.log_param("optuna_sampler", sampler_name)
    mlflow.log_param("optuna_study_best_value", best_value)

    metrics = calculate_metrics(
        y_true=labels["true"], y_pred=labels["pred"], y_prob=labels["prob"]
    )
    mlflow.log_metrics(metrics)

    mlflow_xgb.log_model(xgb_model=best_model, artifact_path="model")

    # Artifacts
    LogFigures.conf_matrix(y_true=labels["true"], y_pred=labels["pred"])
    LogFigures.feature_importance(booster=best_model, figsize=(10, 8))
    LogFigures.roc_curve(estimator=best_model, X=df, y=labels["true"])

    html_content = estimator_html_repr(best_model)
    mlflow.log_text(html_content, "estimator.html")

    metric_info = {"metrics_logged": list(metrics.keys())}
    mlflow.log_dict(metric_info, "metric_info.json")


def _evaluate(
    run_id: Any,
    eval_data: pd.DataFrame,
):
    model_uri = f"runs:/{run_id}/model"
    mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        targets="label",
        model_type="classifier",
    )


def train_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict | None = None,
    n_trials: int | None = 30,
):
    config = config or {}

    sampler_cfg = config.get("sampler", {})
    search_space = config.get("search_space", {})
    run_name = config.get("run_name", "XGBoost_Tuning_Optuna")
    optimization_target = config.get("optimization_target", "f1")

    sampler = get_sampler(sampler_cfg)

    n_neg_samples = float((y_train == 0).sum())
    n_pos_samples = float((y_train == 1).sum())
    scale_pos_weight_default = (
        n_neg_samples / n_pos_samples if n_pos_samples > 0 else 1.0
    )

    def objective(trial):
        params = {
            "random_state": config.get("seed", 42),
            "eval_metric": config.get("eval_metric", "logloss"),
        }

        for param_name, cfg in search_space.items():
            param_type = cfg.get("type")

            if param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, cfg["low"], cfg["high"]
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name, cfg["low"], cfg["high"], log=cfg.get("log", False)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, cfg["choices"]
                )
            elif param_type == "dynamic_scale":
                high_val = scale_pos_weight_default * cfg.get("multiplier", 2.0)
                params[param_name] = trial.suggest_float(
                    param_name, cfg.get("low", 1.0), high_val
                )

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=params["random_state"]
        )
        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=optimization_target,
            n_jobs=-1,
        )

        return scores.mean()

    print(f"🔍 Starting Optuna Search ({n_trials} trials)...")
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials)  # type: ignore

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["eval_metric"] = "logloss"

    print(f"🏆 Best Trial Value: {study.best_value:.4f}")
    print(f"📌 Best Parameters: {best_params}")

    # Retrain the model using the best parameters
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    eval_data = X_test.copy()
    eval_data["label"] = y_test

    env_run_id = os.getenv("MLFLOW_RUN_ID")
    active_run = mlflow.active_run()

    if active_run or env_run_id:
        current_run_id = active_run.info.run_id if active_run else env_run_id
        mlflow.set_tag("mlflow.runName", run_name)

        _log_artifacts(
            best_params=best_params,
            opt_target_name=optimization_target,
            sampler_name=sampler.__class__.__name__,
            best_value=study.best_value,
            df=X_test,
            labels={"true": y_test, "pred": y_pred, "prob": y_prob},
            best_model=best_model,
        )

        _evaluate(run_id=current_run_id, eval_data=eval_data)
    else:
        with mlflow.start_run(run_name=run_name) as run:
            _log_artifacts(
                best_params=best_params,
                opt_target_name=optimization_target,
                sampler_name=sampler.__class__.__name__,
                best_value=study.best_value,
                df=X_test,
                labels={"true": y_test, "pred": y_pred, "prob": y_prob},
                best_model=best_model,
            )

            _evaluate(run_id=run.info.run_id, eval_data=eval_data)


def run_train_tuning(
    experiment_name: str,
    data_path: str,
    tracking_uri: str | None,
    n_trials: int = 30,
    config: dict[str, Any] | None = None,
):
    if tracking_uri:
        mlflow.set_tracking_uri(uri=tracking_uri)
    else:
        print("🌐 Connecting to DagsHub...")

        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")

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

    train_tuning(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_trials=n_trials,
        config=config,
    )


if __name__ == "__main__":
    run_train_tuning(
        experiment_name="PCOS Classification",
        data_path="dataset/pcos_data_preprocessed.csv",
        tracking_uri=envs["MLFLOW_TRACKING_URI_LOCAL"],
    )
