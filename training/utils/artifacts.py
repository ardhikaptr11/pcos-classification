import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import RocCurveDisplay, confusion_matrix


class LogFigures:
    @staticmethod
    def conf_matrix(
        y_true: pd.Series,
        y_pred: pd.Series,
        figsize: tuple = (6, 6),
        title: str = "Confusion Matrix",
        labels: dict[str, str] = {"xlabel": "Predicted Label", "ylabel": "True Label"},
    ):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            cbar=False,
            ax=ax,
        )

        ax.set_title(label=title)
        ax.set_xlabel(xlabel=labels["xlabel"])
        ax.set_ylabel(ylabel=labels["ylabel"])

        save_as_title = title.lower().replace(" ", "_")
        mlflow.log_figure(fig, f"{save_as_title}.png")
        plt.close(fig)

    @staticmethod
    def feature_importance(
        booster: xgb.XGBClassifier,
        figsize: tuple = (6, 6),
        title: str = "Feature Importance",
    ):
        fig, ax = plt.subplots(figsize=figsize)
        xgb.plot_importance(
            booster=booster,
            ax=ax,
            importance_type="weight",
            title=title,
        )

        save_as_title = title.lower().replace(" ", "_")
        mlflow.log_figure(fig, f"{save_as_title}.png")
        plt.close(fig)

    @staticmethod
    def roc_curve(
        estimator: xgb.XGBClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        figsize: tuple = (6, 6),
        title: str = "Receiver Operating Characteristic (ROC) Curve",
        name: str = "XGBoost",
    ):
        fig, ax = plt.subplots(figsize=figsize)
        RocCurveDisplay.from_estimator(estimator=estimator, X=X, y=y, ax=ax, name=name)

        ax.set_title(label=title)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)
