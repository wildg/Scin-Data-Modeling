import importlib
from pathlib import Path

BUCKET_NAME = "dx-scin-public-data"
LOCAL_DATA_DIR = Path("data/raw")


def download_bucket(prefix: str = "", local_dir: Path = LOCAL_DATA_DIR):
    """Download all files from the SCIN public GCS bucket."""
    storage = importlib.import_module("google.cloud.storage")
    client = storage.Client.create_anonymous_client()  # public bucket, no auth needed
    bucket = client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        local_path = local_dir / blob.name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():  # skip already downloaded files
            print(f"Downloading {blob.name}...")
            blob.download_to_filename(local_path)
        else:
            print(f"Skipping {blob.name} (already exists)")


if __name__ == "__main__":
    download_bucket()
