"""Evaluation metrics for multi-label skin condition classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def evaluate_baseline(
    processed_dir: Path,
    model_path: Path,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a saved model on a given split (default: test).

    Supports both legacy models (full 370-class binarizer) and tuned models
    that include ``top_k_indices``, ``thresholds``, and ``top_k_names``.
    """
    import joblib
    from sklearn.metrics import (
        classification_report,
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    npz = np.load(processed_dir / f"embeddings_{split}.npz", allow_pickle=True)
    X = npz["embeddings"]
    y_labels = [json.loads(label) for label in npz["labels"]]

    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    mlb = artifact["binarizer"]

    Y_full = mlb.transform(y_labels)

    # Determine whether this is a tuned model with top-K filtering
    top_k_idx = artifact.get("top_k_indices")
    thresholds = artifact.get("thresholds")
    top_k_names = artifact.get("top_k_names")

    if top_k_idx is not None:
        # Tuned model: restrict labels to top-K classes
        Y = Y_full[:, top_k_idx]
        class_names = top_k_names
    else:
        # Legacy model: use all classes
        Y = Y_full
        class_names = list(mlb.classes_)

    # Predict
    if thresholds is not None:
        # Use optimised per-class thresholds
        Y_proba = clf.predict_proba(X)
        if isinstance(Y_proba, list):
            Y_proba = np.column_stack(
                [
                    cp[:, 1] if cp.ndim == 2 and cp.shape[1] > 1 else cp.ravel()
                    for cp in Y_proba
                ]
            )
        Y_pred = (np.asarray(Y_proba) >= thresholds).astype(int)
    else:
        Y_pred = clf.predict(X)

    metrics: dict[str, Any] = {
        "hamming_loss": float(hamming_loss(Y, Y_pred)),
        "f1_micro": float(f1_score(Y, Y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(Y, Y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(Y, Y_pred, average="weighted", zero_division=0)),
        "precision_micro": float(precision_score(Y, Y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(Y, Y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(Y, Y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(Y, Y_pred, average="macro", zero_division=0)),
        "num_classes": len(class_names),
        "num_test_samples": X.shape[0],
        "model_name": model_path.stem,
        "split": split,
    }

    report = classification_report(
        Y,
        Y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    return metrics


def print_metrics(metrics: dict[str, Any], console: Any) -> None:
    """Pretty-print evaluation metrics using a Rich console."""
    from rich.table import Table

    split_name = metrics.get("split", "test")
    console.print(
        f"\n[bold]{split_name.title()} set:[/bold] {metrics['num_test_samples']} samples, "
        f"{metrics['num_classes']} label classes\n"
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

    # Save metrics JSON to results/ using model name (no timestamp)
    try:
        model_name = metrics.get("model_name") or "metrics"
        # sanitize model_name to a filesystem-safe string
        model_name = str(model_name).replace(" ", "_")
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"metrics_{model_name}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        console.print(f"[green]Saved metrics to[/green] {out_path}")
    except Exception as e:  # pragma: no cover - best-effort save
        console.print(f"[red]Failed to save metrics:[/red] {e}")
