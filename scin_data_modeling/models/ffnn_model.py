"""Feedforward neural network model for multi-label skin condition classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def train_ffnn(
    processed_dir: Path,
    model_dir: Path,
    hidden_layer_sizes: tuple[int, ...] = (768, 256),
    alpha: float = 1e-5,
    learning_rate_init: float = 5e-4,
    batch_size: int = 64,
    max_iter: int = 300,
    random_state: int = 42,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 20,
) -> Path:
    """Train an sklearn MLPClassifier feedforward neural network on cached embeddings."""
    import joblib
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    train_npz = np.load(processed_dir / "embeddings_train.npz", allow_pickle=True)
    X_train = train_npz["embeddings"]
    y_labels_train = [json.loads(label) for label in train_npz["labels"]]

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_labels_train)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )
    clf.fit(X_train, Y_train)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "ffnn_mlp.joblib"
    joblib.dump(
        {
            "classifier": clf,
            "binarizer": mlb,
            "model_type": "sklearn_mlp_ffnn",
        },
        artifact_path,
    )

    return artifact_path


def predict_ffnn(
    model_path: Path,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a saved FFNN model and predict on embeddings."""
    import joblib

    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    mlb = artifact["binarizer"]

    Y_pred = clf.predict(X)
    Y_proba = clf.predict_proba(X)
    if isinstance(Y_proba, list):
        Y_proba = np.column_stack(
            [
                class_proba[:, 1] if class_proba.ndim == 2 and class_proba.shape[1] > 1 else class_proba.ravel()
                for class_proba in Y_proba
            ]
        )

    return Y_pred, np.asarray(Y_proba), list(mlb.classes_)
