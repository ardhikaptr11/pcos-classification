import json
import logging
import os
import sys
import shutil
import tempfile

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from mlflow.artifacts import download_artifacts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

creds_data = json.loads(os.environ["GDRIVE_CREDENTIALS"])
credentials = Credentials.from_authorized_user_info(
    creds_data, scopes=["https://www.googleapis.com/auth/drive"]
)

service = build("drive", "v3", credentials=credentials)
SHARED_DRIVE_ID = os.environ["GDRIVE_FOLDER_ID"]


def main():
    run_id = os.getenv("RUN_ID")

    if not run_id:
        logging.warning("RUN_ID environment variable is not set. Exiting.")
        sys.exit(0)

    logging.info(f"Fetching artifacts for RUN_ID: {run_id}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Put downloaded artifacts in temp dir
            local_artifact_path = download_artifacts(run_id=run_id, dst_path=temp_dir)
            logging.info(f"Artifacts downloaded locally to: {local_artifact_path}")
        except Exception as e:
            logging.error(f"Error downloading artifacts from MLflow: {e}")
            sys.exit(1)

        # 2. Compress folder to ZIP
        zip_base_name = os.path.join(temp_dir, f"model_artifacts_{run_id}")
        shutil.make_archive(
            base_name=zip_base_name, format="zip", root_dir=local_artifact_path
        )

        zip_file_path = f"{zip_base_name}.zip"
        zip_file_name = f"model_artifacts_{run_id}.zip"
        logging.info(f"Compressed artifacts to: {zip_file_name}")

        logging.info(f"Uploading {zip_file_name} to Google Drive...")
        file_meta = {"name": zip_file_name, "parents": [SHARED_DRIVE_ID]}
        media = MediaFileUpload(
            zip_file_path, mimetype="application/zip", resumable=True
        )

        uploaded_file = (
            service.files()
            .create(
                body=file_meta, media_body=media, fields="id", supportsAllDrives=True
            )
            .execute()
        )

        file_id = uploaded_file["id"]
        logging.info(f"Successfully uploaded {zip_file_name} (ID: {file_id})")


if __name__ == "__main__":
    main()
