from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from scin_data_modeling.models import semi_supervised as ss


@dataclass(frozen=True)
class Experiment:
    name: str
    random_state: int
    top_k_classes: int
    min_label_confidence: float
    pseudo_threshold: float
    pseudo_margin: float
    validation_size: float
    cv_folds: int
    self_training_threshold: float = 0.6


def _compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def _fit_temperature_grid(proba: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-12
    temperatures = np.linspace(0.5, 3.0, 51)
    log_proba = np.log(np.clip(proba, eps, 1.0))
    best_t = 1.0
    best_nll = float("inf")
    for t in temperatures:
        scaled = np.exp(log_proba / t)
        scaled /= scaled.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(np.clip(scaled[np.arange(len(y_true)), y_true], eps, 1.0)))
        if nll < best_nll:
            best_nll = float(nll)
            best_t = float(t)
    return best_t


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-12
    log_proba = np.log(np.clip(proba, eps, 1.0))
    scaled = np.exp(log_proba / temperature)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return scaled


def _aggregate_case_probabilities(case_ids: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_cases, inv = np.unique(case_ids.astype(str), return_inverse=True)
    case_proba_sum = np.zeros((unique_cases.shape[0], proba.shape[1]), dtype=np.float64)
    case_counts = np.zeros(unique_cases.shape[0], dtype=np.int64)
    np.add.at(case_proba_sum, inv, proba)
    np.add.at(case_counts, inv, 1)
    case_proba = case_proba_sum / case_counts[:, None]
    return unique_cases, case_proba


def _run_experiment(exp: Experiment, processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "semi_supervised_summary.json"
    model_path = output_dir / "best_semi_supervised_model.joblib"
    if not (summary_path.exists() and model_path.exists()):
        ss.run_benchmark(
            processed_dir=processed_dir,
            output_dir=output_dir,
            random_state=exp.random_state,
            self_training_threshold=exp.self_training_threshold,
            pseudo_threshold=exp.pseudo_threshold,
            pseudo_margin=exp.pseudo_margin,
            top_k_classes=exp.top_k_classes,
            min_label_confidence=exp.min_label_confidence,
            validation_size=exp.validation_size,
            cv_folds=exp.cv_folds,
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "experiment": exp.name,
        "random_state": exp.random_state,
        "top_k_classes": exp.top_k_classes,
        "min_label_confidence": exp.min_label_confidence,
        "pseudo_threshold": exp.pseudo_threshold,
        "pseudo_margin": exp.pseudo_margin,
        "validation_size": exp.validation_size,
        "cv_folds": exp.cv_folds,
        **summary,
    }


def _build_case_vs_image_and_calibration_report(
    *,
    exp: Experiment,
    experiment_dir: Path,
    processed_dir: Path,
    report_dir: Path,
) -> None:
    artifact = joblib.load(experiment_dir / "best_semi_supervised_model.joblib")
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    classes = np.asarray(artifact["label_classes"])

    train_labeled, train_unlabeled, test_labeled, _ = ss._load_processed(processed_dir)  # noqa: SLF001
    train_labeled = train_labeled[train_labeled["target_max_prob"] >= exp.min_label_confidence].reset_index(drop=True)
    test_labeled = test_labeled[test_labeled["target_max_prob"] >= exp.min_label_confidence].reset_index(drop=True)

    label_map = ss._build_label_mapping(train_labeled["target_hard_label"].astype(str), top_k=exp.top_k_classes)  # noqa: SLF001
    train_labeled = train_labeled.copy()
    test_labeled = test_labeled.copy()
    train_labeled["target_grouped_label"] = (
        train_labeled["target_hard_label"].astype(str).map(label_map).fillna("OTHER")
    )
    test_labeled["target_grouped_label"] = (
        test_labeled["target_hard_label"].astype(str).map(label_map).fillna("OTHER")
    )

    train_fit, val_labeled = ss._grouped_validation_split(  # noqa: SLF001
        train_labeled,
        validation_size=exp.validation_size,
        random_state=exp.random_state,
    )

    le = LabelEncoder()
    le.fit(classes)

    val_seen = val_labeled["target_grouped_label"].astype(str).isin(set(classes.tolist())).to_numpy()
    val_labeled_seen = val_labeled[val_seen].reset_index(drop=True)
    X_val = val_labeled_seen[feature_columns].to_numpy(dtype=np.float64)
    y_val = le.transform(val_labeled_seen["target_grouped_label"].astype(str))

    test_seen = test_labeled["target_grouped_label"].astype(str).isin(set(classes.tolist())).to_numpy()
    test_seen_df = test_labeled[test_seen].reset_index(drop=True)
    X_test = test_seen_df[feature_columns].to_numpy(dtype=np.float64)
    y_test = le.transform(test_seen_df["target_grouped_label"].astype(str))

    proba_val = model.predict_proba(X_val)
    temperature = _fit_temperature_grid(proba_val, y_val)

    proba_test = model.predict_proba(X_test)
    proba_test_cal = _apply_temperature(proba_test, temperature)
    y_pred_image = proba_test.argmax(axis=1)
    y_pred_image_cal = proba_test_cal.argmax(axis=1)

    case_ids = test_seen_df["case_id"].astype(str).to_numpy()
    unique_cases, case_proba = _aggregate_case_probabilities(case_ids, proba_test)
    _, case_proba_cal = _aggregate_case_probabilities(case_ids, proba_test_cal)
    case_label_map = (
        test_seen_df.assign(case_id_str=test_seen_df["case_id"].astype(str))[
            ["case_id_str", "target_grouped_label"]
        ]
        .drop_duplicates(subset=["case_id_str"])
        .set_index("case_id_str")
    )
    case_true = case_label_map.reindex(unique_cases)["target_grouped_label"].fillna("OTHER").astype(str).to_numpy()
    y_true_case = le.transform(case_true)
    y_pred_case = case_proba.argmax(axis=1)
    y_pred_case_cal = case_proba_cal.argmax(axis=1)

    image_conf = proba_test.max(axis=1)
    image_margin = image_conf - np.partition(proba_test, -2, axis=1)[:, -2]
    image_conf_cal = proba_test_cal.max(axis=1)
    image_margin_cal = image_conf_cal - np.partition(proba_test_cal, -2, axis=1)[:, -2]
    accepted_uncal = (image_conf >= exp.pseudo_threshold) & (image_margin >= exp.pseudo_margin)
    accepted_cal = (image_conf_cal >= exp.pseudo_threshold) & (image_margin_cal >= exp.pseudo_margin)

    case_eval = pd.DataFrame(
        {
            "case_id": unique_cases,
            "true_label": le.inverse_transform(y_true_case),
            "pred_label": le.inverse_transform(y_pred_case),
            "pred_label_calibrated": le.inverse_transform(y_pred_case_cal),
        }
    )
    image_eval = pd.DataFrame(
        {
            "case_id": case_ids,
            "true_label": le.inverse_transform(y_test),
            "pred_label": le.inverse_transform(y_pred_image),
            "pred_label_calibrated": le.inverse_transform(y_pred_image_cal),
            "confidence": image_conf,
            "confidence_calibrated": image_conf_cal,
            "accepted_uncalibrated": accepted_uncal.astype(int),
            "accepted_calibrated": accepted_cal.astype(int),
        }
    )

    cm_image = confusion_matrix(y_test, y_pred_image, labels=np.arange(len(classes)))
    cm_case = confusion_matrix(y_true_case, y_pred_case, labels=np.arange(len(classes)))
    cm_image_df = pd.DataFrame(cm_image, index=classes, columns=classes)
    cm_case_df = pd.DataFrame(cm_case, index=classes, columns=classes)

    row_sums = cm_image.sum(axis=1)
    recalls = np.divide(np.diag(cm_image), np.maximum(row_sums, 1))
    per_class_error = pd.DataFrame(
        {
            "label": classes,
            "support_image_rows": row_sums,
            "recall_image": recalls,
            "error_rate_image": 1.0 - recalls,
        }
    ).sort_values(by=["error_rate_image", "support_image_rows"], ascending=[False, False])

    metrics = {
        "image_accuracy": float(accuracy_score(y_test, y_pred_image)),
        "image_f1_macro": float(f1_score(y_test, y_pred_image, average="macro")),
        "case_accuracy": float(accuracy_score(y_true_case, y_pred_case)),
        "case_f1_macro": float(f1_score(y_true_case, y_pred_case, average="macro")),
        "image_accuracy_calibrated": float(accuracy_score(y_test, y_pred_image_cal)),
        "image_f1_macro_calibrated": float(f1_score(y_test, y_pred_image_cal, average="macro")),
        "case_accuracy_calibrated": float(accuracy_score(y_true_case, y_pred_case_cal)),
        "case_f1_macro_calibrated": float(f1_score(y_true_case, y_pred_case_cal, average="macro")),
        "ece_image_uncalibrated": _compute_ece(y_test, proba_test),
        "ece_image_calibrated": _compute_ece(y_test, proba_test_cal),
        "temperature": float(temperature),
        "accept_rate_uncalibrated": float(accepted_uncal.mean()),
        "accept_rate_calibrated": float(accepted_cal.mean()),
        "n_test_rows": int(len(y_test)),
        "n_test_cases": int(len(y_true_case)),
        "n_val_rows": int(len(y_val)),
        "n_train_fit_rows": int(len(train_fit)),
        "experiment": exp.name,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "case_vs_image_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm_image_df.to_csv(report_dir / "confusion_matrix_image_level.csv")
    cm_case_df.to_csv(report_dir / "confusion_matrix_case_level.csv")
    per_class_error.to_csv(report_dir / "error_slices_per_class.csv", index=False)
    image_eval.to_csv(report_dir / "test_image_predictions.csv", index=False)
    case_eval.to_csv(report_dir / "test_case_predictions.csv", index=False)


def _ci95(series: pd.Series) -> tuple[float, float, float]:
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    half = float(1.96 * std / np.sqrt(max(len(series), 1)))
    return mean, mean - half, mean + half


def main() -> None:
    processed_dir = Path("data/processed")
    base_dir = Path("artifacts/semi_supervised_next_steps")
    base_dir.mkdir(parents=True, exist_ok=True)

    experiments: list[Experiment] = []
    seeds = [42, 52, 62]
    tune_grid = [
        ("k15_c055_t060_m010", 15, 0.55, 0.60, 0.10),
        ("k20_c055_t060_m010", 20, 0.55, 0.60, 0.10),
        ("k30_c050_t055_m005", 30, 0.50, 0.55, 0.05),
    ]
    for grid_name, top_k, min_conf, p_thr, p_margin in tune_grid:
        for seed in seeds:
            experiments.append(
                Experiment(
                    name=f"{grid_name}_seed{seed}",
                    random_state=seed,
                    top_k_classes=top_k,
                    min_label_confidence=min_conf,
                    pseudo_threshold=p_thr,
                    pseudo_margin=p_margin,
                    validation_size=0.2,
                    cv_folds=3,
                )
            )

    for val_size, cv_folds in [(0.20, 5), (0.25, 3), (0.25, 5)]:
        experiments.append(
            Experiment(
                name=f"val{int(val_size*100)}_cv{cv_folds}_seed42",
                random_state=42,
                top_k_classes=15,
                min_label_confidence=0.55,
                pseudo_threshold=0.60,
                pseudo_margin=0.10,
                validation_size=val_size,
                cv_folds=cv_folds,
            )
        )

    rows: list[dict[str, Any]] = []
    for exp in experiments:
        exp_dir = base_dir / exp.name
        rows.append(_run_experiment(exp, processed_dir, exp_dir))

    summary_df = pd.DataFrame(rows).sort_values(by=["experiment"]).reset_index(drop=True)
    summary_df.to_csv(base_dir / "experiments_summary.csv", index=False)

    summary_df["grid_key"] = summary_df["experiment"].str.extract(r"^(k\d+_c\d+_t\d+_m\d+|val\d+_cv\d+)_?")[0]
    ci_rows: list[dict[str, Any]] = []
    for grid_key, group in summary_df.groupby("grid_key"):
        mean_f1, lo_f1, hi_f1 = _ci95(group["best_test_f1_macro"])
        mean_acc, lo_acc, hi_acc = _ci95(group["best_test_accuracy"])
        ci_rows.append(
            {
                "grid_key": grid_key,
                "runs": int(len(group)),
                "best_method_mode": group["best_method"].mode().iat[0],
                "test_f1_macro_mean": mean_f1,
                "test_f1_macro_ci95_low": lo_f1,
                "test_f1_macro_ci95_high": hi_f1,
                "test_accuracy_mean": mean_acc,
                "test_accuracy_ci95_low": lo_acc,
                "test_accuracy_ci95_high": hi_acc,
                "pseudo_accept_rate_mean": float(group["pseudo_accept_rate"].mean()),
            }
        )
    ci_df = pd.DataFrame(ci_rows).sort_values(by=["test_f1_macro_mean"], ascending=False).reset_index(drop=True)
    ci_df.to_csv(base_dir / "config_ci_summary.csv", index=False)

    method_counts_raw = summary_df.groupby("grid_key")["best_method"].value_counts()
    method_counts = (
        method_counts_raw.to_frame("count")
        .reset_index()
        .sort_values(by=["grid_key", "count"], ascending=[True, False])
    )
    method_counts.to_csv(base_dir / "best_method_frequency.csv", index=False)

    winner_key = str(ci_df.iloc[0]["grid_key"])
    winner_row = (
        summary_df[summary_df["grid_key"] == winner_key]
        .sort_values(by="best_test_f1_macro", ascending=False)
        .iloc[0]
    )
    winner_exp = next(exp for exp in experiments if exp.name == winner_row["experiment"])
    _build_case_vs_image_and_calibration_report(
        exp=winner_exp,
        experiment_dir=base_dir / winner_exp.name,
        processed_dir=processed_dir,
        report_dir=base_dir / "analysis",
    )

    report_index = {
        "experiments_summary_csv": str(base_dir / "experiments_summary.csv"),
        "config_ci_summary_csv": str(base_dir / "config_ci_summary.csv"),
        "best_method_frequency_csv": str(base_dir / "best_method_frequency.csv"),
        "analysis_case_vs_image_metrics_json": str(base_dir / "analysis" / "case_vs_image_metrics.json"),
        "analysis_confusion_matrix_image_csv": str(base_dir / "analysis" / "confusion_matrix_image_level.csv"),
        "analysis_confusion_matrix_case_csv": str(base_dir / "analysis" / "confusion_matrix_case_level.csv"),
        "analysis_error_slices_csv": str(base_dir / "analysis" / "error_slices_per_class.csv"),
        "analysis_test_image_predictions_csv": str(base_dir / "analysis" / "test_image_predictions.csv"),
        "analysis_test_case_predictions_csv": str(base_dir / "analysis" / "test_case_predictions.csv"),
        "selected_analysis_experiment": winner_exp.name,
        "selected_grid_key": winner_key,
    }
    (base_dir / "REPORT_INDEX.json").write_text(json.dumps(report_index, indent=2), encoding="utf-8")
    print(json.dumps(report_index, indent=2))


if __name__ == "__main__":
    main()
