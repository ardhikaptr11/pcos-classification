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
            # Avoid infinite loop when zipping by creating sub-folder
            download_dir = os.path.join(temp_dir, "raw_artifacts")
            os.makedirs(download_dir)

            # Put downloaded artifacts into sub-folder
            local_artifact_path = download_artifacts(
                run_id=run_id, dst_path=download_dir
            )
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

        # Check file size to avoid blind spot
        file_size_mb = os.path.getsize(zip_file_path) / (1024 * 1024)
        logging.info(
            f"Compressed artifacts to: {zip_file_name} (Size: {file_size_mb:.2f} MB)"
        )

        if file_size_mb > 500:
            logging.warning("⚠️ File is unusually large! Upload might take a while.")

        logging.info(f"Uploading {zip_file_name} to Google Drive...")
        file_meta = {"name": zip_file_name, "parents": [SHARED_DRIVE_ID]}

        # Chunk upload to display progress
        media = MediaFileUpload(
            zip_file_path,
            mimetype="application/zip",
            resumable=True,
            chunksize=5 * 1024 * 1024,  # 5 MB per chunk
        )

        request = service.files().create(
            body=file_meta, media_body=media, fields="id", supportsAllDrives=True
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logging.info(f"Upload progress: {int(status.progress() * 100)}%")

        file_id = response.get("id")
        logging.info(f"✅ Successfully uploaded {zip_file_name} (ID: {file_id})")


if __name__ == "__main__":
    main()
