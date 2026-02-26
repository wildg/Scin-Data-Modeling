"""Evaluation metrics for multi-label skin condition classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _compute_multilabel_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import (
        classification_report,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    metrics: dict[str, Any] = {
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "f1_micro": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "num_classes": len(class_names),
        "num_test_samples": int(Y_true.shape[0]),
    }

    report = classification_report(
        Y_true,
        Y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    return metrics


def evaluate_baseline(
    processed_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Evaluate a saved logistic baseline model on the test set."""
    import joblib

    test_npz = np.load(processed_dir / "embeddings_test.npz", allow_pickle=True)
    X_test = test_npz["embeddings"]
    y_labels_test = [json.loads(label) for label in test_npz["labels"]]

    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    mlb = artifact["binarizer"]

    Y_test = mlb.transform(y_labels_test)
    Y_pred = clf.predict(X_test)
    class_names = [str(class_name) for class_name in mlb.classes_]
    return _compute_multilabel_metrics(Y_test, Y_pred, class_names)


def evaluate_inception_embedding(
    processed_dir: Path,
    model_path: Path,
    *,
    device_str: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a saved Inception-style embedding model on the test set."""
    from scin_data_modeling.models.inception_embedding import (
        labels_to_binary_matrix,
        predict_inception_embedding,
    )

    test_npz = np.load(processed_dir / "embeddings_test.npz", allow_pickle=True)
    X_test = test_npz["embeddings"]
    y_labels_test = [json.loads(label) for label in test_npz["labels"]]

    Y_pred, _, class_names = predict_inception_embedding(model_path=model_path, X=X_test, device_str=device_str)
    Y_test = labels_to_binary_matrix(y_labels_test, class_names)
    return _compute_multilabel_metrics(Y_test, Y_pred, class_names)


def print_metrics(metrics: dict[str, Any], console: Any) -> None:
    """Pretty-print evaluation metrics using a Rich console."""
    from rich.table import Table

    console.print(
        f"\n[bold]Test set:[/bold] {metrics['num_test_samples']} samples, {metrics['num_classes']} label classes\n"
    )

    table = Table(title="Multi-Label Classification Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Hamming Loss", f"{metrics['hamming_loss']:.4f}")
    table.add_row("F1 (micro)", f"{metrics['f1_micro']:.4f}")
    table.add_row("F1 (macro)", f"{metrics['f1_macro']:.4f}")
    table.add_row("F1 (weighted)", f"{metrics['f1_weighted']:.4f}")
    table.add_row("Precision (micro)", f"{metrics['precision_micro']:.4f}")
    table.add_row("Recall (micro)", f"{metrics['recall_micro']:.4f}")
    table.add_row("Precision (macro)", f"{metrics['precision_macro']:.4f}")
    table.add_row("Recall (macro)", f"{metrics['recall_macro']:.4f}")

    console.print(table)
