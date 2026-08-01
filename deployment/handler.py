import json
import os
import socket

import uvicorn

from common import setup_logger
from common.download import DagsHub, GoogleDrive
from env import envs

from .predictor import ModelPredictor

logger = setup_logger()


def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3)
        return True
    except OSError:
        return False


def handle(args):
    HOST = envs["HOST"]
    PORT = int(envs["PORT"])

    if args.command == "download-model":
        internet_on = is_connected()

        dest = (
            "model_artifacts/hub"
            if args.source == "dagshub"
            else "model_artifacts/drive"
        )

        if not internet_on:
            raise ConnectionError(
                "Not connected to internet, please check your connection."
            )

        gdrive_client = GoogleDrive(
            credentials=envs["GDRIVE_CREDENTIALS"], folder_id=envs["GDRIVE_FOLDER_ID"]
        )
        dagshub_client = DagsHub(model_name=envs["MODEL_NAME"])

        if args.source not in ["dagshub", "gdrive"]:
            raise ValueError(f"Unknown model source argument: {args.source}")

        (
            dagshub_client.download(dest=dest)
            if args.source == "dagshub"
            else gdrive_client.download(dest=dest)
        )

    if args.command == "serve":
        uvicorn.run("deployment.app:app", host=HOST, port=PORT, reload=True)
        return

    elif args.command == "predict":
        ARTIFACTS_PATH = "model_artifacts"

        if not os.path.exists(ARTIFACTS_PATH):
            logger.error(
                "No model artifacts found, run 'uv run python main.py deploy download-model' to download before proceeding."
            )
            return

        model_source = args.model_source
        source_path = os.path.join(
            ARTIFACTS_PATH, model_source, "model"
        )  # model_artifacts/{drive, hub}/model

        if not os.path.exists(source_path):
            alternative_source = "drive" if model_source == "hub" else "hub"
            alternative_path = os.path.join(ARTIFACTS_PATH, alternative_source, "model")

            if os.path.exists(alternative_path):
                logger.info(
                    f"Model not found in '{source_path}', fallback to use '{alternative_path}'."
                )
                source_path = alternative_path
            else:
                logger.error(
                    f"Failed to find model in both '{source_path}' and '{alternative_path}'."
                )
                return

        model_path = os.path.join(source_path, "MLmodel")

        input_dict = None

        if args.data.endswith(".json") and os.path.exists(args.data):
            logger.info(f"Reading input from file: {args.data}")
            try:
                with open(args.data, "r") as f:
                    input_dict = json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON file: {e}")
                return
        else:
            try:
                input_dict = json.loads(args.data)
            except json.JSONDecodeError:
                logger.error("Invalid JSON string provided.")
                return

        try:
            predictor = ModelPredictor(model_path=model_path)
            response = json.dumps(predictor.predict(input_dict))
            logger.info(f"Prediction result: {response}")
        except Exception as e:
            logger.error(f"Failed to predict: {e}")
