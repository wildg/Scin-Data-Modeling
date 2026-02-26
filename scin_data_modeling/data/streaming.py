"""GCS-streaming PyTorch Dataset for end-to-end fine-tuning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from scin_data_modeling.data.embed import _get_bucket, stream_image_from_gcs
from scin_data_modeling.data.preprocess import load_split


class GCSStreamingDataset(Dataset):
    """A PyTorch Dataset that streams images directly from GCS on each access.

    Each item returns ``(images, label)`` where *images* is a list of 1-3
    transformed tensors (one per image for the case) and *label* is a list of
    skin condition name strings.

    Used for finetune-mode training where the backbone is unfrozen and we
    need the raw pixels, not pre-computed embeddings.
    """

    def __init__(
        self,
        split: str,
        transform: transforms.Compose,
        processed_dir: Path = Path("data/processed"),
    ) -> None:
        self.df = load_split(split, processed_dir=processed_dir)
        self.transform = transform
        self.bucket = _get_bucket()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> tuple[list[torch.Tensor], list[str]]:
        row = self.df.iloc[index]
        label: list[str] = row["label"]
        image_paths: list[str] = row["image_paths"]

        tensors: list[torch.Tensor] = []
        for gcs_path in image_paths:
            pil_img = stream_image_from_gcs(self.bucket, gcs_path)
            tensors.append(self.transform(pil_img))

        return tensors, label


def collate_variable_images(
    batch: list[tuple[list[torch.Tensor], list[str]]],
) -> tuple[torch.Tensor, list[list[str]], torch.Tensor]:
    """Custom collate that mean-pools variable-count images per case.

    Returns
    -------
    (pooled_images, labels, num_images)
        - pooled_images: (B, C, H, W) — mean of each case's 1-3 images
        - labels:        list of B label lists (skin condition names)
        - num_images:    (B,) int tensor — how many images per case
    """
    pooled: list[torch.Tensor] = []
    labels: list[list[str]] = []
    counts: list[int] = []

    for images, label in batch:
        pooled.append(torch.stack(images).mean(dim=0))
        labels.append(label)
        counts.append(len(images))

    return (
        torch.stack(pooled),
        labels,
        torch.tensor(counts, dtype=torch.long),
    )
