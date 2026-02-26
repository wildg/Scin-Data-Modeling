"""Inception-style neural classifier trained on cached image embeddings."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class InceptionMLPBlock(nn.Module):
    """Inception-like block for vector inputs using parallel MLP branches."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        if out_dim < 8:
            msg = f"out_dim must be >= 8, got {out_dim}"
            raise ValueError(msg)

        b1 = out_dim // 4
        b2 = out_dim // 4
        b3 = out_dim // 4
        b4 = out_dim - (b1 + b2 + b3)

        self.branch_1 = nn.Sequential(nn.Linear(in_dim, b1), nn.GELU())
        self.branch_2 = nn.Sequential(nn.Linear(in_dim, b2), nn.GELU(), nn.Linear(b2, b2), nn.GELU())
        self.branch_3 = nn.Sequential(
            nn.Linear(in_dim, b3),
            nn.GELU(),
            nn.Linear(b3, b3),
            nn.GELU(),
            nn.Linear(b3, b3),
            nn.GELU(),
        )
        self.branch_4 = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, b4), nn.GELU())
        self.fuse = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat(
            [
                self.branch_1(x),
                self.branch_2(x),
                self.branch_3(x),
                self.branch_4(x),
            ],
            dim=1,
        )
        return self.fuse(out)


class EmbeddingInceptionNet(nn.Module):
    """Multi-label classifier for embedding vectors."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.25) -> None:
        super().__init__()
        head_dim = max(hidden_dim // 2, 64)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block_1 = InceptionMLPBlock(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.block_2 = InceptionMLPBlock(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block_1(x)
        x = self.block_2(x)
        return self.head(x)


def train_inception_embedding(
    processed_dir: Path,
    model_dir: Path,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 512,
    dropout: float = 0.25,
    validation_split: float = 0.1,
    threshold: float = 0.5,
    seed: int = 42,
    device_str: str = "cpu",
) -> Path:
    """Train an Inception-style classifier on cached embeddings and save an artifact."""
    from sklearn.preprocessing import MultiLabelBinarizer

    train_npz = np.load(Path(processed_dir) / "embeddings_train.npz", allow_pickle=True)
    X = train_npz["embeddings"].astype(np.float32)
    y_labels = [json.loads(label) for label in train_npz["labels"]]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_labels).astype(np.float32)
    classes = [str(class_name) for class_name in mlb.classes_]
    num_classes = len(classes)

    if X.ndim != 2:
        msg = f"Expected 2D embeddings array, got shape {X.shape}"
        raise ValueError(msg)
    if X.shape[0] != Y.shape[0]:
        msg = f"Feature/label row mismatch: X={X.shape[0]}, Y={Y.shape[0]}"
        raise ValueError(msg)
    if not 0.0 <= validation_split < 1.0:
        msg = f"validation_split must be in [0, 1), got {validation_split}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    val_size = int(X.shape[0] * validation_split)
    if X.shape[0] > 1 and validation_split > 0 and val_size == 0:
        val_size = 1

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    if train_idx.size == 0:
        train_idx = indices
        val_idx = np.array([], dtype=np.int64)

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val = X[val_idx] if val_idx.size else np.empty((0, X.shape[1]), dtype=np.float32)
    Y_val = Y[val_idx] if val_idx.size else np.empty((0, num_classes), dtype=np.float32)

    feature_mean = X_train.mean(axis=0, keepdims=True)
    feature_std = X_train.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0

    X_train = ((X_train - feature_mean) / feature_std).astype(np.float32)
    X_val = ((X_val - feature_mean) / feature_std).astype(np.float32)

    device = torch.device(device_str)
    model = EmbeddingInceptionNet(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    pos = Y_train.sum(axis=0)
    neg = Y_train.shape[0] - pos
    pos_weight = torch.tensor((neg + 1e-6) / (pos + 1e-6), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader: DataLoader | None = None
    if X_val.shape[0] > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch_idx in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = train_loss

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss_sum += float(loss.detach().cpu())
                    val_batches += 1
            val_loss = val_loss_sum / max(val_batches, 1)

        history.append(
            {
                "epoch": float(epoch_idx + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "inception_embeddings.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "input_dim": int(X.shape[1]),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "threshold": float(threshold),
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
            "history": history,
        },
        artifact_path,
    )
    return artifact_path


def predict_inception_embedding(
    model_path: Path,
    X: np.ndarray,
    *,
    batch_size: int = 256,
    device_str: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict labels from embeddings using a saved Inception-style artifact."""
    artifact = torch.load(Path(model_path), map_location="cpu")
    classes = [str(class_name) for class_name in artifact["classes"]]
    threshold = float(artifact.get("threshold", 0.5))

    model = EmbeddingInceptionNet(
        input_dim=int(artifact["input_dim"]),
        num_classes=len(classes),
        hidden_dim=int(artifact["hidden_dim"]),
        dropout=float(artifact["dropout"]),
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    feature_mean = artifact["feature_mean"].astype(np.float32)
    feature_std = artifact["feature_std"].astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    X = X.astype(np.float32)
    X = (X - feature_mean) / feature_std

    device = torch.device(device_str)
    model = model.to(device)

    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            stop = min(start + batch_size, X.shape[0])
            xb = torch.from_numpy(X[start:stop]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_chunks.append(probs)

    Y_proba = np.vstack(probs_chunks) if probs_chunks else np.empty((0, len(classes)), dtype=np.float32)
    Y_pred = (Y_proba >= threshold).astype(np.int64)
    return Y_pred, Y_proba, classes


def labels_to_binary_matrix(y_labels: list[list[str]], classes: list[str]) -> np.ndarray:
    """Convert multi-label string lists to a binary matrix using known classes."""
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    matrix = np.zeros((len(y_labels), len(classes)), dtype=np.int64)
    for row_idx, row_labels in enumerate(y_labels):
        for label in row_labels:
            col_idx = class_to_idx.get(label)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1
    return matrix
