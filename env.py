import os

from dotenv import load_dotenv

load_dotenv()

envs: dict[str, str | None] = {
    "MLFLOW_RUN_ID": os.getenv("MLFLOW_RUN_ID"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
    "MLFLOW_TRACKING_URI_LOCAL": os.getenv("MLFLOW_TRACKING_URI_LOCAL"),
    "DAGSHUB_REPO_OWNER": os.getenv("DAGSHUB_REPO_OWNER"),
    "DAGSHUB_REPO_NAME": os.getenv("DAGSHUB_REPO_NAME"),
    "GDRIVE_FOLDER_ID": os.getenv("GDRIVE_FOLDER_ID"),
    "GDRIVE_CREDENTIALS": os.getenv("GDRIVE_CREDENTIALS"),
}

if __name__ == "__main__":
    print(envs)