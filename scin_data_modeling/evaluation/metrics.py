"""Evaluation metrics for multi-label skin condition classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_baseline(
    processed_dir: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Evaluate a saved baseline model on the test set.

    Loads test embeddings, binarizes labels using the saved (fitted) binarizer,
    and computes multi-label classification metrics.
    """
    import joblib
    from sklearn.metrics import (
        classification_report,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    test_npz = np.load(processed_dir / "embeddings_test.npz", allow_pickle=True)
    X_test = test_npz["embeddings"]
    y_labels_test = [json.loads(label) for label in test_npz["labels"]]

    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    mlb = artifact["binarizer"]

    Y_test = mlb.transform(y_labels_test)
    Y_pred = clf.predict(X_test)

    metrics: dict[str, Any] = {
        "hamming_loss": float(hamming_loss(Y_test, Y_pred)),
        "f1_micro": float(f1_score(Y_test, Y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y_test, Y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(Y_test, Y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(Y_test, Y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(Y_test, Y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(Y_test, Y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(Y_test, Y_pred, average="macro", zero_division=0)),
        "num_classes": len(mlb.classes_),
        "num_test_samples": X_test.shape[0],
    }

    report = classification_report(
        Y_test,
        Y_pred,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    return metrics


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
