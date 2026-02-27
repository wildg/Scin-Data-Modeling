"""Hyperparameter tuning for multi-label skin condition classifiers.

Provides tuning functions for Logistic Regression, XGBoost, and FFNN models.
Uses the validation split for evaluation (no cross-validation) and optimises
for macro F1 to ensure equal class weighting.  Includes top-K class filtering
and per-class threshold optimisation.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


# ── Shared helpers ────────────────────────────────────────────────────────────


def _load_split(
    processed_dir: Path,
    split_name: str,
) -> tuple[np.ndarray, list[list[str]]]:
    """Load embeddings and raw label lists for *split_name*."""
    npz = np.load(processed_dir / f"embeddings_{split_name}.npz", allow_pickle=True)
    X = npz["embeddings"]
    labels = [json.loads(lbl) for lbl in npz["labels"]]
    return X, labels


def filter_top_k_classes(
    Y_train: np.ndarray,
    mlb: Any,
    k: int = 30,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Select the *k* most frequent classes from the binarised training matrix.

    Returns
    -------
    Y_filtered : ndarray of shape (n_samples, k)
        Training label matrix restricted to the top-k columns.
    top_k_names : list[str]
        Class names for the selected columns (in frequency order).
    top_k_indices : list[int]
        Column indices into the original ``mlb.classes_`` array.
    """
    class_counts = Y_train.sum(axis=0)
    top_k_indices = np.argsort(class_counts)[::-1][:k].tolist()
    top_k_names = [mlb.classes_[i] for i in top_k_indices]
    Y_filtered = Y_train[:, top_k_indices]
    return Y_filtered, top_k_names, top_k_indices


def _apply_top_k_filter(
    Y: np.ndarray,
    top_k_indices: list[int],
) -> np.ndarray:
    """Restrict a binarised label matrix to previously selected top-k columns."""
    return Y[:, top_k_indices]


def optimize_thresholds(
    Y_true: np.ndarray,
    Y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Find the per-class decision threshold that maximises F1 on *Y_true*.

    Parameters
    ----------
    Y_true : ndarray of shape (n_samples, n_classes)
    Y_proba : ndarray of shape (n_samples, n_classes)
    thresholds : candidate thresholds to sweep (default 0.05 – 0.90 in steps of 0.05)

    Returns
    -------
    best_thresholds : ndarray of shape (n_classes,)
    """
    from sklearn.metrics import f1_score

    if thresholds is None:
        thresholds = np.arange(0.05, 0.91, 0.05)

    n_classes = Y_true.shape[1]
    best_thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        best_f1 = -1.0
        for t in thresholds:
            preds = (Y_proba[:, c] >= t).astype(int)
            f1 = f1_score(Y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = t

    return best_thresholds


def _macro_f1(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))


def _print_results_table(
    results: list[dict[str, Any]],
    console: Any,
    title: str,
) -> None:
    """Print a Rich table summarising tuning results."""
    from rich.table import Table

    table = Table(title=title)
    table.add_column("#", style="dim")
    table.add_column("Params")
    table.add_column("Macro F1 (val)", style="green")
    table.add_column("Time (s)", style="cyan")

    for i, r in enumerate(results, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        table.add_row(
            str(i),
            params_str,
            f"{r['macro_f1']:.4f}",
            f"{r['time']:.1f}",
        )

    console.print(table)


# ── Logistic Regression tuning ────────────────────────────────────────────────


def tune_baseline(
    processed_dir: Path,
    model_dir: Path,
    top_k: int = 30,
) -> Path:
    """Grid search over C and class_weight for Logistic Regression.

    Returns the path to the saved best-model artifact.
    """
    import joblib
    from rich.console import Console
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    console = Console()
    console.print("\n[bold]Tuning Logistic Regression[/bold]")

    # Load data
    X_train, y_train = _load_split(processed_dir, "train")
    X_val, y_val = _load_split(processed_dir, "validate")

    mlb = MultiLabelBinarizer()
    Y_train_full = mlb.fit_transform(y_train)
    Y_val_full = mlb.transform(y_val)

    # Filter to top-K classes
    Y_train, top_k_names, top_k_idx = filter_top_k_classes(Y_train_full, mlb, k=top_k)
    Y_val = _apply_top_k_filter(Y_val_full, top_k_idx)

    console.print(f"  Classes: {len(top_k_names)} (top-{top_k} from {len(mlb.classes_)})")
    console.print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Parameter grid
    param_grid = list(
        itertools.product(
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # C
            ["balanced", None],  # class_weight
        )
    )

    results: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_params: dict[str, Any] = {}

    for C_val, cw in param_grid:
        params = {"C": C_val, "class_weight": cw}
        t0 = time.time()

        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=C_val, solver="lbfgs", class_weight=cw),
            n_jobs=-1,
        )
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_val)
        f1 = _macro_f1(Y_val, Y_pred)
        elapsed = time.time() - t0

        results.append({"params": params, "macro_f1": f1, "time": elapsed})
        console.print(f"  C={C_val:<8} cw={str(cw):<10} → macro F1={f1:.4f}  ({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    _print_results_table(results, console, "Logistic Regression Tuning Results")
    console.print(f"\n[bold green]Best:[/bold green] {best_params}  macro F1={best_f1:.4f}")

    # Retrain best model and optimise thresholds
    best_clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            C=best_params["C"],
            solver="lbfgs",
            class_weight=best_params["class_weight"],
        ),
        n_jobs=-1,
    )
    best_clf.fit(X_train, Y_train)
    Y_val_proba = best_clf.predict_proba(X_val)
    best_thresholds = optimize_thresholds(Y_val, Y_val_proba)

    # Evaluate with optimised thresholds
    Y_val_pred_tuned = (Y_val_proba >= best_thresholds).astype(int)
    tuned_f1 = _macro_f1(Y_val, Y_val_pred_tuned)
    console.print(f"  After threshold tuning: macro F1={tuned_f1:.4f}")

    # Save artifact
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "baseline_logreg.joblib"
    joblib.dump(
        {
            "classifier": best_clf,
            "binarizer": mlb,
            "top_k_indices": top_k_idx,
            "top_k_names": top_k_names,
            "thresholds": best_thresholds,
            "best_params": best_params,
        },
        artifact_path,
    )
    console.print(f"[green]Saved to {artifact_path}[/green]\n")
    return artifact_path


# ── XGBoost tuning ────────────────────────────────────────────────────────────


def tune_xgboost(
    processed_dir: Path,
    model_dir: Path,
    top_k: int = 30,
    n_iter: int = 15,
    seed: int = 42,
) -> Path:
    """Randomised search over XGBoost hyperparameters.

    Returns the path to the saved best-model artifact.
    """
    import joblib
    from rich.console import Console
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from xgboost import XGBClassifier

    console = Console()
    console.print("\n[bold]Tuning XGBoost[/bold]")

    X_train, y_train = _load_split(processed_dir, "train")
    X_val, y_val = _load_split(processed_dir, "validate")

    mlb = MultiLabelBinarizer()
    Y_train_full = mlb.fit_transform(y_train)
    Y_val_full = mlb.transform(y_val)

    Y_train, top_k_names, top_k_idx = filter_top_k_classes(Y_train_full, mlb, k=top_k)
    Y_val = _apply_top_k_filter(Y_val_full, top_k_idx)

    console.print(f"  Classes: {len(top_k_names)} (top-{top_k} from {len(mlb.classes_)})")
    console.print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Full parameter grid (sample n_iter random combos)
    param_space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1, 0.2],
        "scale_pos_weight": [1, 5, 10],
        "min_child_weight": [1, 3],
    }

    rng = np.random.RandomState(seed)
    all_combos = list(itertools.product(*param_space.values()))
    sample_indices = rng.choice(len(all_combos), size=min(n_iter, len(all_combos)), replace=False)
    sampled_combos = [all_combos[i] for i in sample_indices]

    results: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_params: dict[str, Any] = {}

    for combo in sampled_combos:
        params = dict(zip(param_space.keys(), combo))
        t0 = time.time()

        clf = OneVsRestClassifier(
            XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                scale_pos_weight=params["scale_pos_weight"],
                min_child_weight=params["min_child_weight"],
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
            ),
            n_jobs=-1,
        )
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_val)
        f1 = _macro_f1(Y_val, Y_pred)
        elapsed = time.time() - t0

        results.append({"params": params, "macro_f1": f1, "time": elapsed})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        console.print(f"  {params_str} → macro F1={f1:.4f}  ({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    _print_results_table(results, console, "XGBoost Tuning Results")
    console.print(f"\n[bold green]Best:[/bold green] {best_params}  macro F1={best_f1:.4f}")

    # Retrain best and optimise thresholds
    best_clf = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            scale_pos_weight=best_params["scale_pos_weight"],
            min_child_weight=best_params["min_child_weight"],
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
        n_jobs=-1,
    )
    best_clf.fit(X_train, Y_train)
    Y_val_proba = best_clf.predict_proba(X_val)
    best_thresholds = optimize_thresholds(Y_val, Y_val_proba)

    Y_val_pred_tuned = (Y_val_proba >= best_thresholds).astype(int)
    tuned_f1 = _macro_f1(Y_val, Y_val_pred_tuned)
    console.print(f"  After threshold tuning: macro F1={tuned_f1:.4f}")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "xgboost_model.joblib"
    joblib.dump(
        {
            "classifier": best_clf,
            "binarizer": mlb,
            "top_k_indices": top_k_idx,
            "top_k_names": top_k_names,
            "thresholds": best_thresholds,
            "best_params": best_params,
        },
        artifact_path,
    )
    console.print(f"[green]Saved to {artifact_path}[/green]\n")
    return artifact_path


# ── FFNN tuning ───────────────────────────────────────────────────────────────


def tune_ffnn(
    processed_dir: Path,
    model_dir: Path,
    top_k: int = 30,
    n_iter: int = 12,
    seed: int = 42,
) -> Path:
    """Randomised search over MLPClassifier hyperparameters.

    Returns the path to the saved best-model artifact.
    """
    import joblib
    from rich.console import Console
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    console = Console()
    console.print("\n[bold]Tuning FFNN (MLPClassifier)[/bold]")

    X_train, y_train = _load_split(processed_dir, "train")
    X_val, y_val = _load_split(processed_dir, "validate")

    mlb = MultiLabelBinarizer()
    Y_train_full = mlb.fit_transform(y_train)
    Y_val_full = mlb.transform(y_val)

    Y_train, top_k_names, top_k_idx = filter_top_k_classes(Y_train_full, mlb, k=top_k)
    Y_val = _apply_top_k_filter(Y_val_full, top_k_idx)

    console.print(f"  Classes: {len(top_k_names)} (top-{top_k} from {len(mlb.classes_)})")
    console.print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    param_space = {
        "hidden_layer_sizes": [(512,), (768, 256), (1024, 512), (512, 256, 128)],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-4, 5e-4, 1e-3],
        "batch_size": [32, 64],
    }

    rng = np.random.RandomState(seed)
    all_combos = list(itertools.product(*param_space.values()))
    sample_indices = rng.choice(len(all_combos), size=min(n_iter, len(all_combos)), replace=False)
    sampled_combos = [all_combos[i] for i in sample_indices]

    results: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_params: dict[str, Any] = {}

    for combo in sampled_combos:
        params = dict(zip(param_space.keys(), combo))
        t0 = time.time()

        clf = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=params["alpha"],
            batch_size=params["batch_size"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
        clf.fit(X_train, Y_train)

        Y_proba = clf.predict_proba(X_val)
        if isinstance(Y_proba, list):
            Y_proba = np.column_stack(
                [
                    cp[:, 1] if cp.ndim == 2 and cp.shape[1] > 1 else cp.ravel()
                    for cp in Y_proba
                ]
            )

        Y_pred = (np.asarray(Y_proba) >= 0.5).astype(int)
        f1 = _macro_f1(Y_val, Y_pred)
        elapsed = time.time() - t0

        results.append({"params": params, "macro_f1": f1, "time": elapsed})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        console.print(f"  {params_str} → macro F1={f1:.4f}  ({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    _print_results_table(results, console, "FFNN Tuning Results")
    console.print(f"\n[bold green]Best:[/bold green] {best_params}  macro F1={best_f1:.4f}")

    # Retrain best and optimise thresholds
    best_clf = MLPClassifier(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=best_params["alpha"],
        batch_size=best_params["batch_size"],
        learning_rate_init=best_params["learning_rate_init"],
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    best_clf.fit(X_train, Y_train)

    Y_val_proba = best_clf.predict_proba(X_val)
    if isinstance(Y_val_proba, list):
        Y_val_proba = np.column_stack(
            [
                cp[:, 1] if cp.ndim == 2 and cp.shape[1] > 1 else cp.ravel()
                for cp in Y_val_proba
            ]
        )
    Y_val_proba = np.asarray(Y_val_proba)

    best_thresholds = optimize_thresholds(Y_val, Y_val_proba)

    Y_val_pred_tuned = (Y_val_proba >= best_thresholds).astype(int)
    tuned_f1 = _macro_f1(Y_val, Y_val_pred_tuned)
    console.print(f"  After threshold tuning: macro F1={tuned_f1:.4f}")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / "ffnn_mlp.joblib"
    joblib.dump(
        {
            "classifier": best_clf,
            "binarizer": mlb,
            "model_type": "sklearn_mlp_ffnn",
            "top_k_indices": top_k_idx,
            "top_k_names": top_k_names,
            "thresholds": best_thresholds,
            "best_params": best_params,
        },
        artifact_path,
    )
    console.print(f"[green]Saved to {artifact_path}[/green]\n")
    return artifact_path
