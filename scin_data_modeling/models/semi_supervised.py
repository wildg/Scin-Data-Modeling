from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

from scin_data_modeling.data.preprocess import run_pipeline

METADATA_COLUMNS = ("case_id", "image_path")


def _build_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=350,
        random_state=random_state,
        n_jobs=-1,
        class_weight=None,
        min_samples_leaf=2,
    )


def _metric_row(method: str, y_true: Any, y_pred: Any, n_train: int) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    return {
        "method": method,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "train_rows_used": int(n_train),
    }


def _split_views(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint = X.shape[1] // 2
    if midpoint == 0:
        return X, X
    return X[:, :midpoint], X[:, midpoint:]


def _cluster_pseudo_labels(
    X_train_all: np.ndarray,
    y_masked: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    pseudo = y_masked.copy()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto", batch_size=2048)
    cluster_ids = kmeans.fit_predict(X_train_all)

    for cluster_id in np.unique(cluster_ids):
        in_cluster = cluster_ids == cluster_id
        known = pseudo[in_cluster]
        known = known[known != -1]
        if known.size == 0:
            continue
        majority = int(np.bincount(known).argmax())
        pseudo[(in_cluster) & (pseudo == -1)] = majority
    return pseudo


def _co_training_consensus(
    X_train_all: np.ndarray,
    y_masked: np.ndarray,
    random_state: int,
    confidence_threshold: float = 0.9,
    max_rounds: int = 3,
    max_new_per_round: int = 150,
) -> np.ndarray:
    left, right = _split_views(X_train_all)
    pseudo = y_masked.copy()

    for round_idx in range(max_rounds):
        labeled_mask = pseudo != -1
        unlabeled_idx = np.flatnonzero(~labeled_mask)
        if unlabeled_idx.size == 0:
            break

        left_model = _build_rf(random_state + round_idx)
        right_model = _build_rf(random_state + 100 + round_idx)
        left_model.fit(left[labeled_mask], pseudo[labeled_mask])
        right_model.fit(right[labeled_mask], pseudo[labeled_mask])

        left_proba = left_model.predict_proba(left[unlabeled_idx])
        right_proba = right_model.predict_proba(right[unlabeled_idx])
        left_pred = left_proba.argmax(axis=1)
        right_pred = right_proba.argmax(axis=1)
        left_conf = left_proba.max(axis=1)
        right_conf = right_proba.max(axis=1)

        consensus = (
            (left_pred == right_pred)
            & (left_conf >= confidence_threshold)
            & (right_conf >= confidence_threshold)
        )
        if not np.any(consensus):
            break

        cand_idx = unlabeled_idx[consensus]
        cand_labels = left_pred[consensus]
        cand_score = np.minimum(left_conf[consensus], right_conf[consensus])
        order = np.argsort(-cand_score)[:max_new_per_round]
        chosen_idx = cand_idx[order]
        chosen_labels = cand_labels[order]
        pseudo[chosen_idx] = chosen_labels.astype(np.int64)

    return pseudo


def _load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
    train_labeled = pd.read_csv(processed_dir / "train_labeled.csv")
    train_unlabeled = pd.read_csv(processed_dir / "train_unlabeled.csv")
    test_labeled = pd.read_csv(processed_dir / "test_labeled.csv")
    return train_labeled, train_unlabeled, test_labeled, manifest


def _build_label_mapping(train_labels: pd.Series, top_k: int) -> dict[str, str]:
    counts = train_labels.value_counts()
    top = set(counts.head(top_k).index.tolist())
    return {label: (label if label in top else "OTHER") for label in counts.index.tolist()}


def _ensure_processed(processed_dir: Path) -> None:
    required = (
        processed_dir / "train_labeled.csv",
        processed_dir / "train_unlabeled.csv",
        processed_dir / "test_labeled.csv",
        processed_dir / "manifest.json",
    )
    if all(p.exists() for p in required):
        return

    run_pipeline(
        cases_path=Path("data/raw/dataset/scin_cases.csv"),
        labels_path=Path("data/raw/dataset/scin_labels.csv"),
        output_dir=processed_dir,
        test_size=0.2,
        random_state=42,
        scale_numeric=True,
    )


def run_benchmark(
    *,
    processed_dir: Path,
    output_dir: Path,
    random_state: int,
    self_training_threshold: float,
    pseudo_threshold: float,
    pseudo_margin: float,
    top_k_classes: int,
    min_label_confidence: float,
) -> None:
    _ensure_processed(processed_dir)
    train_labeled, train_unlabeled, test_labeled, manifest = _load_processed(processed_dir)
    feature_columns = manifest["feature_columns"]

    train_labeled = train_labeled[train_labeled["target_max_prob"] >= min_label_confidence].reset_index(drop=True)
    test_labeled = test_labeled[test_labeled["target_max_prob"] >= min_label_confidence].reset_index(drop=True)
    if train_labeled.empty or test_labeled.empty:
        raise ValueError("No labeled rows left after confidence filtering. Lower --min-label-confidence.")

    label_map = _build_label_mapping(train_labeled["target_hard_label"].astype(str), top_k=top_k_classes)
    train_labeled["target_grouped_label"] = (
        train_labeled["target_hard_label"].astype(str).map(label_map).fillna("OTHER")
    )
    test_labeled["target_grouped_label"] = (
        test_labeled["target_hard_label"].astype(str).map(label_map).fillna("OTHER")
    )

    X_labeled = train_labeled[feature_columns].to_numpy(dtype=np.float64)
    X_unlabeled = train_unlabeled[feature_columns].to_numpy(dtype=np.float64)
    X_test_all = test_labeled[feature_columns].to_numpy(dtype=np.float64)

    le = LabelEncoder()
    y_labeled = np.asarray(le.fit_transform(train_labeled["target_grouped_label"].astype(str)), dtype=np.int64)
    seen_class_set = set(le.classes_.tolist())
    test_seen_mask = np.asarray(
        test_labeled["target_grouped_label"].astype(str).isin(seen_class_set).to_numpy(),
        dtype=bool,
    )
    if not np.any(test_seen_mask):
        raise ValueError("No test rows have labels seen in training; cannot evaluate.")
    X_test = X_test_all[test_seen_mask]
    y_test = np.asarray(
        le.transform(test_labeled.loc[test_seen_mask, "target_grouped_label"].astype(str)),
        dtype=np.int64,
    )

    w_labeled = np.asarray(train_labeled["sample_weight_case_inverse"].to_numpy(dtype=np.float64), dtype=np.float64)
    w_unlabeled = np.asarray(train_unlabeled["sample_weight_case_inverse"].to_numpy(dtype=np.float64), dtype=np.float64)
    X_train_all = np.vstack([X_labeled, X_unlabeled])
    y_masked = np.full(X_train_all.shape[0], -1, dtype=np.int64)
    y_masked[: y_labeled.shape[0]] = y_labeled

    metrics: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    supervised = _build_rf(random_state)
    supervised.fit(X_labeled, y_labeled, sample_weight=w_labeled)
    supervised_pred = np.asarray(supervised.predict(X_test), dtype=np.int64)
    metrics.append(_metric_row("Supervised_Only_RF", y_test, supervised_pred, len(y_labeled)))
    models["Supervised_Only_RF"] = supervised

    # Positional arg keeps compatibility with sklearn versions that use
    # either `base_estimator` (older) or `estimator` (newer).
    self_training = SelfTrainingClassifier(
        _build_rf(random_state + 1),
        threshold=self_training_threshold,
        max_iter=6,
        verbose=False,
    )
    self_training.fit(X_train_all, y_masked)
    st_pred = np.asarray(self_training.predict(X_test), dtype=np.int64)
    labeled_after_st = int(np.sum(np.asarray(self_training.transduction_) != -1))
    metrics.append(_metric_row("SelfTraining_RF", y_test, st_pred, labeled_after_st))
    models["SelfTraining_RF"] = self_training

    label_spreading = LabelSpreading(kernel="knn", n_neighbors=9, alpha=0.25, max_iter=40)
    label_spreading.fit(X_train_all, y_masked)
    ls_pred = np.asarray(label_spreading.predict(X_test), dtype=np.int64)
    labeled_after_ls = int(np.sum(np.asarray(label_spreading.transduction_) != -1))
    metrics.append(_metric_row("LabelSpreading", y_test, ls_pred, labeled_after_ls))
    models["LabelSpreading"] = label_spreading

    n_clusters = int(max(30, min(300, len(np.unique(y_labeled)) * 3)))
    cluster_pseudo = _cluster_pseudo_labels(X_train_all, y_masked, n_clusters=n_clusters, random_state=random_state + 2)
    cluster_mask = cluster_pseudo != -1
    cluster_model = _build_rf(random_state + 2)
    cluster_weights = np.concatenate([w_labeled, w_unlabeled])[cluster_mask]
    cluster_model.fit(X_train_all[cluster_mask], cluster_pseudo[cluster_mask], sample_weight=cluster_weights)
    cluster_pred = np.asarray(cluster_model.predict(X_test), dtype=np.int64)
    metrics.append(_metric_row("ClusterPseudoLabel_RF", y_test, cluster_pred, int(cluster_mask.sum())))
    models["ClusterPseudoLabel_RF"] = cluster_model

    co_pseudo = _co_training_consensus(
        X_train_all,
        y_masked,
        random_state=random_state + 3,
        confidence_threshold=self_training_threshold,
    )
    co_mask = co_pseudo != -1
    co_model = _build_rf(random_state + 3)
    co_weights = np.concatenate([w_labeled, w_unlabeled])[co_mask]
    co_model.fit(X_train_all[co_mask], co_pseudo[co_mask], sample_weight=co_weights)
    co_pred = np.asarray(co_model.predict(X_test), dtype=np.int64)
    metrics.append(_metric_row("CoTraining_Consensus_RF", y_test, co_pred, int(co_mask.sum())))
    models["CoTraining_Consensus_RF"] = co_model

    metrics_df = pd.DataFrame(metrics).sort_values(by=["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)
    best_method = str(metrics_df.iloc[0]["method"])
    best_model = models[best_method]

    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_unlabeled)
    elif hasattr(best_model, "estimator_") and hasattr(best_model.estimator_, "predict_proba"):
        proba = best_model.estimator_.predict_proba(X_unlabeled)
    else:
        raise ValueError(f"Best model {best_method} does not expose predict_proba.")

    case_series = train_unlabeled.get("case_id", pd.Series(index=train_unlabeled.index, dtype=object))
    case_ids = case_series.astype(str).to_numpy()
    unique_cases, inv = np.unique(case_ids, return_inverse=True)

    case_proba_sum = np.zeros((unique_cases.shape[0], proba.shape[1]), dtype=np.float64)
    case_counts = np.zeros(unique_cases.shape[0], dtype=np.int64)
    np.add.at(case_proba_sum, inv, proba)
    np.add.at(case_counts, inv, 1)
    case_proba = case_proba_sum / case_counts[:, None]

    top_idx_case = np.argsort(-case_proba, axis=1)
    top1_case = top_idx_case[:, 0]
    top2_case = top_idx_case[:, 1] if case_proba.shape[1] > 1 else top_idx_case[:, 0]
    conf_case = case_proba[np.arange(case_proba.shape[0]), top1_case]
    second_case = case_proba[np.arange(case_proba.shape[0]), top2_case]
    margin_case = conf_case - second_case
    accepted_case = (conf_case >= pseudo_threshold) & (margin_case >= pseudo_margin)
    case_label = le.inverse_transform(top1_case.astype(np.int64))

    case_table = pd.DataFrame(
        {
            "case_id": unique_cases,
            "pseudo_label": case_label,
            "pseudo_confidence": conf_case,
            "pseudo_margin": margin_case,
            "accepted": accepted_case.astype(int),
        }
    )
    pseudo_df = train_unlabeled.copy()
    pseudo_df["case_id"] = case_series.astype(str)
    pseudo_df = pseudo_df.merge(case_table, on="case_id", how="left")
    pseudo_df["pseudo_source_method"] = best_method
    keep_cols = [
        "case_id",
        "image_path",
        "pseudo_label",
        "pseudo_confidence",
        "pseudo_margin",
        "accepted",
        "pseudo_source_method",
    ]
    pseudo_df = pseudo_df[keep_cols]
    accepted_df = pseudo_df[pseudo_df["accepted"] == 1].reset_index(drop=True)

    summary = {
        "best_method": best_method,
        "n_labeled_train_rows": int(len(train_labeled)),
        "n_unlabeled_train_rows": int(len(train_unlabeled)),
        "n_unlabeled_train_cases": int(len(unique_cases)),
        "n_test_labeled_rows_total": int(len(test_labeled)),
        "n_test_labeled_rows_seen_classes": int(test_seen_mask.sum()),
        "n_test_labeled_rows_unseen_classes": int((~test_seen_mask).sum()),
        "top_k_classes": int(top_k_classes),
        "min_label_confidence": float(min_label_confidence),
        "n_grouped_classes": int(len(le.classes_)),
        "n_pseudo_accepted_rows": int(len(accepted_df)),
        "n_pseudo_accepted_cases": int(case_table["accepted"].sum()),
        "pseudo_accept_rate": float(len(accepted_df) / max(1, len(pseudo_df))),
        "pseudo_accept_rate_cases": float(case_table["accepted"].sum() / max(1, len(case_table))),
        "pseudo_threshold": float(pseudo_threshold),
        "pseudo_margin": float(pseudo_margin),
        "self_training_threshold": float(self_training_threshold),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "semi_supervised_metrics.csv", index=False)
    pseudo_df.to_csv(output_dir / "unlabeled_pseudo_labels_all.csv", index=False)
    accepted_df.to_csv(output_dir / "unlabeled_pseudo_labels_accepted.csv", index=False)
    (output_dir / "semi_supervised_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Best method: {best_method}")
    print(f"Saved: {output_dir / 'semi_supervised_metrics.csv'}")
    print(f"Saved: {output_dir / 'unlabeled_pseudo_labels_all.csv'}")
    print(f"Saved: {output_dir / 'unlabeled_pseudo_labels_accepted.csv'}")
    print(f"Saved: {output_dir / 'semi_supervised_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate semi-supervised SCIN tabular models.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/semi_supervised"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--self-training-threshold", type=float, default=0.6)
    parser.add_argument("--pseudo-threshold", type=float, default=0.6)
    parser.add_argument("--pseudo-margin", type=float, default=0.1)
    parser.add_argument("--top-k-classes", type=int, default=15)
    parser.add_argument("--min-label-confidence", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        self_training_threshold=args.self_training_threshold,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_margin=args.pseudo_margin,
        top_k_classes=args.top_k_classes,
        min_label_confidence=args.min_label_confidence,
    )


if __name__ == "__main__":
    main()
