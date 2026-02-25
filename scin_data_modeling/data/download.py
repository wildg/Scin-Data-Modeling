import argparse
import importlib
from pathlib import Path

BUCKET_NAME = "dx-scin-public-data"
LOCAL_DATA_DIR = Path("data/raw")


def download_bucket(prefix: str = "", local_dir: Path = LOCAL_DATA_DIR, csv_only: bool = False):
    """Download all files from the SCIN public GCS bucket."""
    storage = importlib.import_module("google.cloud.storage")
    client = storage.Client.create_anonymous_client()  # public bucket, no auth needed
    bucket = client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if csv_only and not blob.name.lower().endswith(".csv"):
            continue

        local_path = local_dir / blob.name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():  # skip already downloaded files
            print(f"Downloading {blob.name}...")
            blob.download_to_filename(local_path)
        else:
            print(f"Skipping {blob.name} (already exists)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SCIN public data files.")
    parser.add_argument("--prefix", default="", help="Optional blob prefix filter.")
    parser.add_argument("--local-dir", type=Path, default=LOCAL_DATA_DIR, help="Local output directory.")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Download only CSV files and skip all non-CSV assets (e.g., images).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_bucket(prefix=args.prefix, local_dir=args.local_dir, csv_only=args.csv_only)
