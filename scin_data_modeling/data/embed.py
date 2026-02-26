"""Stream images from GCS and extract embeddings via a frozen backbone."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scin_data_modeling.data.download import BUCKET_NAME
from scin_data_modeling.data.preprocess import PROCESSED_DATA_DIR, load_split

# ── GCS image streaming ───────────────────────────────────────────────────────


def _get_bucket() -> storage.Bucket:
    client = storage.Client.create_anonymous_client()
    return client.bucket(BUCKET_NAME)


def stream_image_from_gcs(bucket: storage.Bucket, blob_name: str) -> Image.Image:
    """Fetch a single image from GCS and return it as a PIL Image.

    The image is decoded in memory — nothing is written to disk.
    """
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ── Embedding extraction ──────────────────────────────────────────────────────


def extract_embeddings(
    split_df: pd.DataFrame,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream images from GCS, encode with *model*, and mean-pool per case.

    Parameters
    ----------
    split_df:
        DataFrame from :func:`~scin_data_modeling.data.preprocess.load_split`
        with columns ``case_id``, ``image_paths`` (list of GCS blob names),
        ``label`` (list of skin condition strings).
    model:
        Frozen backbone in eval mode.
    transform:
        torchvision transform matching the backbone's expected input.
    device:
        ``torch.device`` to run inference on.

    Returns
    -------
    (case_ids, embeddings, labels)
        - case_ids:   shape (N,) — string array
        - embeddings: shape (N, embed_dim) — float32
        - labels:     shape (N,) — list of JSON-encoded label arrays
    """
    bucket = _get_bucket()
    model = model.to(device)

    all_case_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_labels: list[str] = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Embedding", unit="case"):
        image_paths: list[str] = row["image_paths"]
        label: list[str] = row["label"]
        case_id: str = str(row["case_id"])

        # Encode each image for this case, then mean-pool
        image_embeddings: list[torch.Tensor] = []
        for gcs_path in image_paths:
            try:
                pil_img = stream_image_from_gcs(bucket, gcs_path)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(tensor).squeeze(0).cpu()
                image_embeddings.append(emb)
            except Exception as exc:
                tqdm.write(f"  Skipping {gcs_path}: {exc}")

        if not image_embeddings:
            tqdm.write(f"  No valid images for case {case_id}, skipping.")
            continue

        # Mean-pool across the 1-3 images for this case
        case_emb = torch.stack(image_embeddings).mean(dim=0).numpy()

        all_case_ids.append(case_id)
        all_embeddings.append(case_emb)
        all_labels.append(json.dumps(label))

    return (
        np.array(all_case_ids),
        np.array(all_embeddings, dtype=np.float32),
        np.array(all_labels),  # array of JSON strings
    )


# ── Save / load ───────────────────────────────────────────────────────────────


def save_embeddings(
    out_path: Path,
    case_ids: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    backbone_name: str,
) -> None:
    """Persist embeddings to a ``.npz`` file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        case_ids=case_ids,
        embeddings=embeddings,
        labels=labels,
        backbone_name=np.array(backbone_name),
    )
    print(f"Saved {len(case_ids):,} embeddings ({embeddings.shape[1]}-dim) → {out_path}")


def load_embeddings(path: Path) -> dict[str, np.ndarray | str | list]:
    """Load a ``.npz`` embedding file.

    Returns a dict with keys: ``case_ids``, ``embeddings``, ``labels``
    (list of lists of condition strings), ``backbone_name``.
    """
    data = np.load(path, allow_pickle=True)
    return {
        "case_ids": data["case_ids"],
        "embeddings": data["embeddings"],
        "labels": [json.loads(l) for l in data["labels"]],
        "backbone_name": str(data["backbone_name"]),
    }


# ── High-level convenience ────────────────────────────────────────────────────


def embed_split(
    split: str,
    backbone_name: str,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """End-to-end: load a split CSV → stream & embed → save .npz.

    Returns the path of the saved ``.npz`` file.
    """
    split_df = load_split(split, processed_dir=processed_dir)
    case_ids, embeddings, labels = extract_embeddings(split_df, model, transform, device)
    out_path = processed_dir / f"embeddings_{split}.npz"
    save_embeddings(out_path, case_ids, embeddings, labels, backbone_name)
    return out_path
