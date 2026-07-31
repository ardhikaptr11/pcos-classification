import os
from typing import Any

import mlflow.xgboost
import numpy as np
import pandas as pd

from common import setup_logger

logger = setup_logger()


class ModelPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = mlflow.xgboost.load_model(self.model_path)
            logger.info("Model successfully loaded into memory.")
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, input_data: dict):
        try:
            # Data conversion (None -> NaN) suitable for XGBoost
            data_dict = {k: (np.nan if v is None else v) for k, v in input_data.items()}
            input_df = pd.DataFrame([data_dict])

            # Dynamic Reordering
            expected_features = self.model.get_booster().feature_names

            # Fallback if get_booster() returns None
            if not expected_features and hasattr(self.model, "feature_names_in_"):
                expected_features = list(self.model.feature_names_in_)

            # If there are missing columns, fill them with NaN
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = np.nan

            input_df = input_df[expected_features]

            prediction_array = self.model.predict_proba(input_df)
            confidence_score = float(prediction_array[0][1])

            pred_class = 1 if confidence_score > 0.5 else 0

            if pred_class == 1:
                label = "PCOS Positive"
                risk_level = None
            else:
                label = "PCOS Negative"
                if confidence_score <= 0.20:
                    risk_level = "Low Risk"
                elif confidence_score <= 0.35:
                    risk_level = "Moderate Risk"
                else:
                    risk_level = "High Risk"

            response = {"label": label}
            key = "probability_to_pcos" if risk_level else "confidence_score"
            if risk_level:
                response["risk_level"] = risk_level

            response[key] = str(round(confidence_score, 4))

            return response
        except Exception as e:
            error_msg = f"❌ Error during inference: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
