from pathlib import Path

from google.cloud import storage
from tqdm import tqdm

BUCKET_NAME = "dx-scin-public-data"
LOCAL_DATA_DIR = Path("data/raw")

# GCS prefixes for the two components of the dataset
CSV_PREFIXES = [
    "dataset/scin_cases.csv",
    "dataset/scin_labels.csv",
    "dataset/scin_label_questions.csv",
    "dataset/scin_app_questions.csv",
]
IMAGES_PREFIX = "dataset/images/"


def download_bucket(prefix: str = "", local_dir: Path = LOCAL_DATA_DIR) -> None:
    """Download files matching *prefix* from the SCIN public GCS bucket.

    Files are saved under *local_dir* preserving the GCS path structure, so a
    blob at ``dataset/images/foo.png`` lands at ``<local_dir>/dataset/images/foo.png``.
    Already-downloaded files are skipped automatically.
    """
    client = storage.Client.create_anonymous_client()  # public bucket – no auth needed
    bucket = client.bucket(BUCKET_NAME)

    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in tqdm(blobs, desc=f"Downloading '{prefix or '(all)'}'", unit="file"):
        local_path = local_dir / blob.name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            blob.download_to_filename(local_path)
        # already-downloaded files are silently skipped


def download_csvs(local_dir: Path = LOCAL_DATA_DIR) -> None:
    """Download only the metadata CSV files (fast, < 10 MB total)."""
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET_NAME)

    for blob_name in tqdm(CSV_PREFIXES, desc="Downloading CSVs", unit="file"):
        blob = bucket.blob(blob_name)
        local_path = local_dir / blob_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            blob.download_to_filename(local_path)


def download_images(local_dir: Path = LOCAL_DATA_DIR) -> None:
    """Download all images (~5 000 cases × up to 3 images).

    This may take several minutes depending on your connection.
    """
    download_bucket(prefix=IMAGES_PREFIX, local_dir=local_dir)


if __name__ == "__main__":
    download_csvs()
    download_images()
