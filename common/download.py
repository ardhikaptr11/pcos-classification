import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from mlflow import MlflowClient, artifacts

from common import setup_logger

logger = setup_logger()

load_dotenv()


def internal(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        return result

    return wrapper


class GoogleDrive:
    def __init__(self, credentials, folder_id):
        self.credentials = credentials
        self.folder_id = folder_id

        self.SCOPES = ["https://www.googleapis.com/auth/drive"]

    @internal
    def _get_service(self):
        if self.credentials:
            creds_data = json.loads(self.credentials)
            creds = Credentials.from_authorized_user_info(info=creds_data, scopes=self.SCOPES)
        else:
            raise FileNotFoundError("Google Drive credentials not found in .env file.")

        return build("drive", "v3", credentials=creds)

    @internal
    def _get_latest_metadata(self):
        service = self._get_service()
        query = f"'{self.folder_id}' in parents and mimeType='application/zip' and trashed=false"

        try:
            results = (
                service.files()
                .list(
                    q=query,
                    orderBy="modifiedTime desc",
                    pageSize=1,
                    fields="files(id, name, modifiedTime)",
                )
                .execute()
            )
        except Exception as e:
            raise RuntimeError(f"Failed to query Google Drive API: {e}")

        items = results.get("files", [])
        if not items:
            return None

        latest_file = items[0]

        return latest_file

    def download(self, dest: str):
        METADATA_FILE = os.path.join(dest, "metadata.json")

        logger.info(f"Searching for the latest model in folder: {self.folder_id}...")

        latest_file = self._get_latest_metadata()
        if not latest_file:
            raise FileNotFoundError("No zip files found in the specified Google Drive folder.")

        _id = latest_file["id"]
        _name = latest_file["name"]
        _modified_time = latest_file["modifiedTime"]

        # Caching
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE) as f:
                try:
                    local_metadata = json.load(f)
                    if local_metadata.get("modifiedTime") == _modified_time:
                        logger.info("Local model is up to date. Skipping download.")
                except json.JSONDecodeError:
                    logger.error("Metadata file malformed, will be downloaded again.")

        logger.info(f"Found latest model: {latest_file['name']}. Downloading...")

        service = self._get_service()
        request = service.files().get_media(fileId=_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download Progress: {int(status.progress() * 100)}%", end="\r")

        print()

        logger.info("Cleaning up old artifacts and extracting new ones...")
        fh.seek(0)

        if os.path.exists(dest):
            shutil.rmtree(dest)

        os.makedirs(dest, exist_ok=True)

        with zipfile.ZipFile(fh, "r") as zip_ref:
            zip_ref.extractall(path=dest)

        with open(METADATA_FILE, "w") as fp:
            json.dump(
                obj={
                    "id": _id,
                    "name": _name,
                    "modifiedTime": _modified_time,
                    "downloadedAt": datetime.now().isoformat(),
                },
                fp=fp,
                indent=4,
            )

        logger.info(f"✅ Model successfully extracted and synced to {dest}")


class DagsHub:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.alias = "champion"

    @internal
    def _get_client(self):
        return MlflowClient()

    @internal
    def _get_latest_metadata(self):
        client = self._get_client()

        # Get the latest model
        try:
            model_version_details = client.get_model_version_by_alias(name=self.model_name, alias=self.alias)
            return model_version_details
        except Exception as e:
            raise RuntimeError(f"Failed to query MLflow API: {e}")

    def download(self, dest: str):
        logger.info(f"Searching for the latest '{self.alias}' model '{self.model_name}' in DagsHub...")

        METADATA_FILE = os.path.join(dest, "metadata.json")

        latest_model = self._get_latest_metadata()
        if not latest_model:
            raise FileNotFoundError(f"Model '{self.model_name}' with alias '{self.alias}' not found in registry.")

        _version = latest_model.version
        _run_id = latest_model.run_id

        # Caching
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                try:
                    local_metadata = json.load(f)
                    if local_metadata.get("version") == _version:
                        logger.info(f"Local hub model is up to date (Version {_version}). Skipping download.")
                except json.JSONDecodeError:
                    logger.error("Metadata file malformed, will be downloaded again.")

        logger.info(f"Found latest model (Version {_version}). Downloading...")

        model_uri = f"models:/{self.model_name}@{self.alias}"

        try:
            if os.path.exists(dest):
                shutil.rmtree(dest)

            artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dest)
        except Exception as e:
            raise RuntimeError(f"Failed to download model artifacts from MLflow: {e}")

        os.makedirs(dest, exist_ok=True)
        with open(METADATA_FILE, "w") as fp:
            json.dump(
                obj={
                    "model_name": self.model_name,
                    "alias": self.alias,
                    "version": _version,
                    "run_id": _run_id,
                    "downloadedAt": datetime.now().isoformat(),
                },
                fp=fp,
                indent=4,
            )

        logger.info(f"✅ Model successfully downloaded and synced to {dest}")
