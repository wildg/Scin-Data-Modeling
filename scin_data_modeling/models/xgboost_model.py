"""XGBoost model for multi-label skin condition classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def train_xgboost(
    processed_dir: Path,
    model_dir: Path,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    scale_pos_weight: float = 1.0,
    min_child_weight: int = 1,
) -> Path:
    """Train a OneVsRestClassifier(XGBClassifier) on cached embeddings.

    Loads embeddings_train.npz, fits a MultiLabelBinarizer on the training
    labels, trains the classifier, and saves both as a single joblib artifact.

    Returns the path to the saved model artifact.
    """
    import joblib
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from xgboost import XGBClassifier

    train_npz = np.load(processed_dir / "embeddings_train.npz", allow_pickle=True)
    X_train = train_npz["embeddings"]
    y_labels_train = [json.loads(label) for label in train_npz["labels"]]

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_labels_train)

    clf = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=min_child_weight,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "xgboost_model.joblib"
    joblib.dump({"classifier": clf, "binarizer": mlb}, artifact_path)

    return artifact_path


def predict_xgboost(
    model_path: Path,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a saved XGBoost model and predict on embeddings.

    Returns (Y_pred_binary, Y_pred_proba, class_names).
    """
    import joblib

    artifact = joblib.load(model_path)
    clf = artifact["classifier"]
    mlb = artifact["binarizer"]

    Y_pred = clf.predict(X)
    Y_proba = clf.predict_proba(X)

    return Y_pred, Y_proba, list(mlb.classes_)
