import json
import os
import sys
import tempfile

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from mlflow.artifacts import download_artifacts

creds_data = json.loads(os.environ["GDRIVE_CREDENTIALS"])
credentials = Credentials.from_authorized_user_info(
    creds_data, scopes=["https://www.googleapis.com/auth/drive"]
)

service = build("drive", "v3", credentials=credentials)
SHARED_DRIVE_ID = os.environ["GDRIVE_FOLDER_ID"]


def upload_directory(local_dir_path, parent_drive_id):
    for item_name in os.listdir(local_dir_path):
        item_path = os.path.join(local_dir_path, item_name)

        # Sub-folder
        if os.path.isdir(item_path):
            folder_meta = {
                "name": item_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_drive_id],
            }
            created_folder = (
                service.files()
                .create(body=folder_meta, fields="id", supportsAllDrives=True)
                .execute()
            )
            new_folder_id = created_folder["id"]
            print(f"📁 Created folder: {item_name} (ID: {new_folder_id})")

            upload_directory(item_path, new_folder_id)

        # File
        else:
            print(f"📄 Uploading file: {item_name}")
            file_meta = {"name": item_name, "parents": [parent_drive_id]}
            media = MediaFileUpload(item_path, resumable=True)
            service.files().create(
                body=file_meta,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()


def main():
    run_id = os.getenv("RUN_ID")

    if not run_id:
        print("⚠️ Warning: RUN_ID environment variable is not set. Exiting.")
        sys.exit(0)

    print(f"🚀 Fetching artifacts for RUN_ID: {run_id}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            local_artifact_path = download_artifacts(run_id=run_id, dst_path=temp_dir)
            print(f"📦 Artifacts downloaded locally to: {local_artifact_path}")
        except Exception as e:
            print(f"❌ Error downloading artifacts from MLflow: {e}")
            sys.exit(1)

        run_id_folder_meta = {
            "name": run_id,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [SHARED_DRIVE_ID],
        }
        run_id_folder = (
            service.files()
            .create(body=run_id_folder_meta, fields="id", supportsAllDrives=True)
            .execute()
        )
        run_id_folder_id = run_id_folder["id"]
        print(
            f"\n=== Created Google Drive folder: {run_id} (ID: {run_id_folder_id}) ==="
        )

        upload_directory(local_artifact_path, run_id_folder_id)

    print("✅ All run_id artifacts have been successfully uploaded to Google Drive!")


if __name__ == "__main__":
    main()
