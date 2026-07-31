"""
NOTE: In the early stages of development, all types of tests (unit and integration) are combined and written in the simplest way possible.
TODO: Tests should be separated into distinct files for better organization.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from deployment.app import app
from deployment.predictor import ModelPredictor


@pytest.fixture
def mock_xgboost_model():
    mock_model = MagicMock()
    mock_booster = MagicMock()

    mock_booster.feature_names = [
        "age_yrs",
        "weight_kg",
        "height_cm",
        "bmi",
        "pulse_rate_bpm",
        "rr_breathsmin",
        "hb_gdl",
        "cycle_ri",
        "cycle_length_days",
        "pregnant_yn",
        "i_betahcg_miuml",
        "ii_betahcg_miuml",
        "fsh_miuml",
        "lh_miuml",
        "lhfsh_ratio",
        "hip_inch",
        "waist_inch",
        "waisthip_ratio",
        "tsh_miul",
        "amh_ngml",
        "prl_ngml",
        "vit_d3_ngml",
        "prg_ngml",
        "rbs_mgdl",
        "weight_gain_yn",
        "hair_growth_yn",
        "skin_darkening_yn",
        "hair_loss_yn",
        "pimples_yn",
        "fast_food_yn",
        "regexercise_yn",
        "bp_systolic_mmhg",
        "bp_diastolic_mmhg",
        "follicle_no_l",
        "follicle_no_r",
        "total_follicles",
        "follicles_difference",
        "avg_f_size_l_mm",
        "avg_f_size_r_mm",
        "endometrium_mm",
    ]

    mock_model.get_booster.return_value = mock_booster

    return mock_model


def test_model_predictor_initialization_failure():
    with pytest.raises(RuntimeError):
        ModelPredictor(model_path="dummy_path")


@patch("mlflow.xgboost.load_model")
def test_model_predictor_predict_negative(mock_load_model, mock_xgboost_model):
    mock_load_model.return_value = mock_xgboost_model

    mock_xgboost_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    predictor = ModelPredictor(model_path="model_artifacts/drive/model/MLmodel")

    with open("data/sample_negative.json", "r") as fp:
        input_data = json.load(fp=fp)

    response = predictor.predict(input_data)

    assert response["label"] == "PCOS Negative"
    assert response["risk_level"] == "Low Risk"
    assert "probability_to_pcos" in response
    assert "confidence_score" not in response


@patch("mlflow.xgboost.load_model")
def test_model_predictor_predict_positive(mock_load_model, mock_xgboost_model):
    mock_load_model.return_value = mock_xgboost_model

    mock_xgboost_model.predict_proba.return_value = np.array([[0.25, 0.75]])

    predictor = ModelPredictor(model_path="model_artifacts/drive/model/MLmodel")
    with open("data/sample_positive.json", "r") as fp:
        input_data = json.load(fp=fp)

    response = predictor.predict(input_data)

    assert response["label"] == "PCOS Positive"
    assert "confidence_score" in response
    assert "risk_level" not in response
    assert "probability_to_pcos" not in response


def test_api_predict_endpoint():
    client = TestClient(app)

    with patch("deployment.app.predictor") as mock_predictor:
        mock_predictor.predict.return_value = {
            "label": "PCOS Negative",
            "risk_level": "Low Risk",
            "probability_to_pcos": "0.1500",
        }

        sample_payload = {
            "age_yrs": 25.0,
            "weight_kg": 55.0,
            "height_cm": 160.0,
            "bmi": 21.5,
            "pulse_rate_bpm": 72.0,
            "rr_breathsmin": 16.0,
            "hb_gdl": 13.5,
            "cycle_ri": 2.0,
            "cycle_length_days": 28.0,
            "pregnant_yn": 0,
            "i_betahcg_miuml": 1.5,
            "ii_betahcg_miuml": 1.5,
            "fsh_miuml": 6.0,
            "lh_miuml": 6.0,
            "lhfsh_ratio": 1.0,
            "hip_inch": 36.0,
            "waist_inch": 28.0,
            "waisthip_ratio": 0.77,
            "tsh_miul": 2.5,
            "amh_ngml": 2.5,
            "prl_ngml": 12.0,
            "vit_d3_ngml": 35.0,
            "prg_ngml": 0.5,
            "rbs_mgdl": 90.0,
            "weight_gain_yn": 0,
            "hair_growth_yn": 0,
            "skin_darkening_yn": 0,
            "hair_loss_yn": 0,
            "pimples_yn": 0,
            "fast_food_yn": 0,
            "regexercise_yn": 1,
            "bp_systolic_mmhg": 115.0,
            "bp_diastolic_mmhg": 75.0,
            "follicle_no_l": 5,
            "follicle_no_r": 6,
            "total_follicles": 11,
            "follicles_difference": 1,
            "avg_f_size_l_mm": 18.0,
            "avg_f_size_r_mm": 17.0,
            "endometrium_mm": 7.0,
        }

        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 200

        data = response.json()

        if data["label"] == "PCOS Positive":
            assert "risk_level" not in data
            assert "confidence_score" in data
        else:
            assert data["risk_level"] == "Low Risk"
            assert "probability_to_pcos" in data
