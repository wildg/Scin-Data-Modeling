"""Tests for the embedding-based Inception model workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scin_data_modeling.evaluation.metrics import evaluate_inception_embedding
from scin_data_modeling.models.inception_embedding import predict_inception_embedding, train_inception_embedding


def _write_embedding_split(path: Path, num_rows: int, embed_dim: int, labels: list[list[str]], seed: int) -> None:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(num_rows, embed_dim)).astype(np.float32)
    case_ids = np.array([f"case_{idx}" for idx in range(num_rows)])
    label_json = np.array([json.dumps(row) for row in labels])
    np.savez(
        path,
        case_ids=case_ids,
        embeddings=embeddings,
        labels=label_json,
        backbone_name=np.array("resnet50"),
    )


def test_inception_embedding_train_predict_evaluate(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    model_dir = tmp_path / "models"
    processed_dir.mkdir()
    model_dir.mkdir()

    train_labels = [
        ["eczema", "acne"],
        ["acne"],
        ["psoriasis"],
        ["eczema"],
        ["acne", "rosacea"],
        ["rosacea"],
        ["eczema", "psoriasis"],
        ["acne"],
        ["rosacea", "psoriasis"],
        ["eczema"],
        ["acne", "psoriasis"],
        ["rosacea"],
    ]
    test_labels = [
        ["eczema"],
        ["acne"],
        ["psoriasis"],
        ["rosacea", "acne"],
    ]

    _write_embedding_split(processed_dir / "embeddings_train.npz", 12, 16, train_labels, seed=7)
    _write_embedding_split(processed_dir / "embeddings_test.npz", 4, 16, test_labels, seed=11)

    artifact_path = train_inception_embedding(
        processed_dir=processed_dir,
        model_dir=model_dir,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        hidden_dim=32,
        dropout=0.1,
        validation_split=0.25,
        seed=123,
        device_str="cpu",
    )
    assert artifact_path.exists()

    test_npz = np.load(processed_dir / "embeddings_test.npz", allow_pickle=True)
    Y_pred, Y_proba, class_names = predict_inception_embedding(
        model_path=artifact_path,
        X=test_npz["embeddings"],
        device_str="cpu",
    )
    assert Y_pred.shape == (4, len(class_names))
    assert Y_proba.shape == (4, len(class_names))
    assert set(class_names) == {"acne", "eczema", "psoriasis", "rosacea"}

    metrics = evaluate_inception_embedding(
        processed_dir=processed_dir,
        model_path=artifact_path,
        device_str="cpu",
    )
    assert metrics["num_test_samples"] == 4
    assert metrics["num_classes"] == 4
