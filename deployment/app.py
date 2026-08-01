import logging
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from common import setup_logger

from .predictor import ModelPredictor

logger = setup_logger()

predictor = None

MODEL_PATH = os.getenv("MODEL_PATH", "model_artifacts/drive/model/MLmodel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    try:
        logger.info(f"Model path: {MODEL_PATH}")
        # Class instantiation when the server starts
        predictor = ModelPredictor(model_path=MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to start predictor: {e}")
        raise RuntimeError("Service startup failed.")

    yield
    predictor = None


app = FastAPI(title="PCOS Classification API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)


class PatientData(BaseModel):
    # REQUIRED for user input
    age_yrs: float
    weight_kg: float
    height_cm: float
    bmi: float
    cycle_length_days: float
    weight_gain_yn: int
    hair_growth_yn: int
    skin_darkening_yn: int
    hair_loss_yn: int
    pimples_yn: int
    fast_food_yn: int
    regexercise_yn: int
    pregnant_yn: int

    # OPTIONAL
    pulse_rate_bpm: Optional[float] = None
    rr_breathsmin: Optional[float] = None
    hb_gdl: Optional[float] = None
    cycle_ri: Optional[float] = None
    i_betahcg_miuml: Optional[float] = None
    ii_betahcg_miuml: Optional[float] = None
    fsh_miuml: Optional[float] = None
    lh_miuml: Optional[float] = None
    lhfsh_ratio: Optional[float] = None
    hip_inch: Optional[float] = None
    waist_inch: Optional[float] = None
    waisthip_ratio: Optional[float] = None
    tsh_miul: Optional[float] = None
    amh_ngml: Optional[float] = None
    prl_ngml: Optional[float] = None
    vit_d3_ngml: Optional[float] = None
    prg_ngml: Optional[float] = None
    rbs_mgdl: Optional[float] = None
    bp_systolic_mmhg: Optional[float] = None
    bp_diastolic_mmhg: Optional[float] = None
    follicle_no_l: Optional[int] = None
    follicle_no_r: Optional[int] = None
    total_follicles: Optional[int] = None
    follicles_difference: Optional[int] = None
    avg_f_size_l_mm: Optional[float] = None
    avg_f_size_r_mm: Optional[float] = None
    endometrium_mm: Optional[float] = None


@app.get("/")
async def root():
    return {
        "status": "OK!",
        "code": HTTPStatus.OK,
        "message": "PCOS Classification API",
    }


@app.get("/health")
async def health():
    return {
        "status": "OK!",
        "code": HTTPStatus.OK,
        "message": "Server is up and runinng!",
    }


@app.post("/predict")
def predict_endpoint(data: PatientData):
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Service Unavailable: Model is not loaded."
        )

    try:
        result = predictor.predict(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
