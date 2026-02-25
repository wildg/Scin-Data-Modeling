"""Streamlit dashboard for the SCIN skin condition prediction pipeline."""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scin_data_modeling.evaluation.metrics import evaluate_baseline
from scin_data_modeling.models.baseline import predict_baseline

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline_logreg.joblib")

STATUS_COLORS = {
    "Correct": "#0E9F6E",
    "Incorrect": "#D64545",
    "Below threshold": "#94A3B8",
}


# ── cached data loaders (run once per session) ────────────────────────────────


@st.cache_data
def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_df["label_parsed"] = train_df["label"].apply(json.loads)
    test_df["label_parsed"] = test_df["label"].apply(json.loads)
    return train_df, test_df


@st.cache_data
def load_metrics() -> dict:
    return evaluate_baseline(processed_dir=PROCESSED_DIR, model_path=MODEL_PATH)


@st.cache_data
def load_predictions() -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz = np.load(PROCESSED_DIR / "embeddings_test.npz", allow_pickle=True)
    X_test = npz["embeddings"]
    return predict_baseline(MODEL_PATH, X_test)


# ── helpers ───────────────────────────────────────────────────────────────────


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        html, body, .stApp, .block-container {
            font-family: 'Space Grotesk', 'Avenir Next', 'Segoe UI', sans-serif;
        }
        span.material-symbols-rounded {
            font-family: "Material Symbols Rounded" !important;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 35%),
                radial-gradient(circle at bottom left, rgba(45, 212, 191, 0.10), transparent 40%),
                #f8fafc;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }
        .app-banner {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            border: 1px solid rgba(15, 23, 42, 0.12);
            background: rgba(255, 255, 255, 0.88);
            margin-bottom: 1rem;
        }
        .app-banner h1 {
            margin: 0;
            font-size: 1.8rem;
            line-height: 1.2;
            color: #0f172a;
        }
        .app-banner p {
            margin: 0.3rem 0 0;
            color: #334155;
            font-size: 0.98rem;
        }
        .token-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.4rem 0 0.8rem;
        }
        .token {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.84rem;
            border: 1px solid;
            white-space: nowrap;
        }
        .token-success {
            background: rgba(16, 185, 129, 0.14);
            border-color: rgba(5, 150, 105, 0.35);
            color: #065f46;
        }
        .token-danger {
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(220, 38, 38, 0.3);
            color: #7f1d1d;
        }
        .token-neutral {
            background: rgba(100, 116, 139, 0.12);
            border-color: rgba(71, 85, 105, 0.25);
            color: #334155;
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _check_artifacts(required: list[Path] | None = None) -> bool:
    required_paths = required or [MODEL_PATH, PROCESSED_DIR / "embeddings_test.npz"]
    missing = [path for path in required_paths if not path.exists()]
    if not missing:
        return True

    st.error("Missing required files for this view.")
    st.code("\n".join(str(path) for path in missing), language="bash")
    st.info("Run the training pipeline before opening this section.")
    st.code("uv run scin_data_modeling train --mode frozen", language="bash")
    return False


def _render_banner(title: str, subtitle: str) -> None:
    st.markdown(
        (
            "<section class='app-banner'>"
            f"<h1>{escape(title)}</h1>"
            f"<p>{escape(subtitle)}</p>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _render_metric_cards(metrics: dict) -> None:
    metric_specs = [
        ("Hamming Loss", "hamming_loss", "Fraction of label slots predicted incorrectly. Lower is better, but this can look deceptively low for sparse labels."),
        ("F1 (micro)", "f1_micro", "Best single headline metric for imbalanced multi-label classification."),
        ("Precision (micro)", "precision_micro", "When a label is predicted, how often it is correct."),
        ("Recall (micro)", "recall_micro", "Fraction of true labels successfully retrieved."),
        ("F1 (macro)", "f1_macro", "Averages F1 equally across all classes, highlighting rare-class performance."),
        ("F1 (weighted)", "f1_weighted", "Per-class F1 weighted by class frequency."),
        ("Precision (macro)", "precision_macro", "Average precision across classes with equal weight."),
        ("Recall (macro)", "recall_macro", "Average recall across classes with equal weight."),
    ]
    for metric_row in [metric_specs[:4], metric_specs[4:]]:
        columns = st.columns(4)
        for column, (label, key, help_text) in zip(columns, metric_row):
            column.metric(label, f"{metrics[key]:.2%}", help=help_text)


def _render_label_tokens(items: list[str], tone: str = "neutral") -> None:
    if not items:
        st.caption("None")
        return
    tokens = "".join(
        f"<span class='token token-{tone}'>{escape(str(item))}</span>" for item in items
    )
    st.markdown(f"<div class='token-wrap'>{tokens}</div>", unsafe_allow_html=True)


def _format_case_option(idx: int, row: pd.Series) -> str:
    labels = row.get("label_parsed", [])
    primary_label = labels[0] if labels else "No label"
    age_group = row.get("age_group", "—")
    sex_at_birth = row.get("sex_at_birth", "—")
    return f"Case {idx:03d} | {primary_label} | {age_group} | {sex_at_birth}"


def _build_true_label_matrix(
    label_lists: list[list[str]], class_names: list[str]
) -> np.ndarray:
    class_to_idx = {condition: idx for idx, condition in enumerate(class_names)}
    y_true = np.zeros((len(label_lists), len(class_names)), dtype=np.uint8)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            col_idx = class_to_idx.get(label)
            if col_idx is not None:
                y_true[row_idx, col_idx] = 1
    return y_true


def _micro_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1_score


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ── app layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SCIN Model Dashboard", layout="wide")
_inject_theme()
_render_banner(
    "SCIN Skin Condition Prediction Dashboard",
    "Logistic regression baseline with ResNet50 embeddings",
)

with st.sidebar:
    st.subheader("Navigation")
    page = st.radio(
        "Navigate",
        ["Model Performance", "Data Explorer", "Prediction Explorer"],
        label_visibility="collapsed",
    )
    st.caption("UI tuned for quick metric scan, data profiling, and case-level prediction review.")


# ── Page 1: Model Performance ─────────────────────────────────────────────────

if page == "Model Performance":
    st.header("Model Performance")
    st.markdown(
        "Held-out evaluation for multi-label skin condition prediction. "
        "Explore summary metrics, per-class variation, and interpretation driven by current run outputs."
    )

    if not _check_artifacts([MODEL_PATH, PROCESSED_DIR / "embeddings_test.npz", PROCESSED_DIR / "test.csv"]):
        st.stop()

    metrics = load_metrics()
    _, test_df = load_csvs()
    Y_pred, Y_proba, class_names = load_predictions()
    y_true = _build_true_label_matrix(
        test_df["label_parsed"].tolist(),
        class_names,
    )
    y_pred = np.asarray(Y_pred).astype(np.uint8)
    y_proba = np.asarray(Y_proba)
    if y_true.shape != y_pred.shape or y_true.shape != y_proba.shape:
        st.error(
            "Prediction output shape does not match test labels. "
            "Please rerun training artifacts."
        )
        st.stop()
    class_to_idx = {condition: idx for idx, condition in enumerate(class_names)}
    fairness_fields = {
        "Age Group": "age_group",
        "Fitzpatrick Skin Type": "fitzpatrick_skin_type",
        "Race": "combined_race",
        "Sex at Birth": "sex_at_birth",
    }

    _render_metric_cards(metrics)
    st.divider()

    st.subheader("Threshold tuning")
    st.caption(
        "Use this to pick a global probability cutoff for multi-label predictions. "
        "Recommendation below is based on held-out test data for analysis only."
    )
    threshold_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    threshold_rows = []
    for threshold_value in threshold_grid:
        threshold_pred = (y_proba >= threshold_value).astype(np.uint8)
        threshold_precision, threshold_recall, threshold_f1 = _micro_precision_recall_f1(
            y_true, threshold_pred
        )
        threshold_rows.append(
            {
                "threshold": float(threshold_value),
                "precision": threshold_precision,
                "recall": threshold_recall,
                "f1": threshold_f1,
                "avg_pred_labels_per_case": float(threshold_pred.sum() / len(test_df)),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    best_threshold_row = threshold_df.loc[threshold_df["f1"].idxmax()]
    recommended_threshold = float(best_threshold_row["threshold"])

    selected_threshold = st.slider(
        "Inspect threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
        key="model_threshold_selected",
    )
    selected_threshold_pred = (y_proba >= selected_threshold).astype(np.uint8)
    selected_precision, selected_recall, selected_f1 = _micro_precision_recall_f1(
        y_true, selected_threshold_pred
    )
    selected_avg_pred_labels = float(selected_threshold_pred.sum() / len(test_df))

    threshold_metric_cols = st.columns(4)
    threshold_metric_cols[0].metric(
        "Recommended threshold (max F1)",
        f"{recommended_threshold:.2f}",
    )
    threshold_metric_cols[1].metric("Precision @ threshold", f"{selected_precision:.2%}")
    threshold_metric_cols[2].metric("Recall @ threshold", f"{selected_recall:.2%}")
    threshold_metric_cols[3].metric("F1 @ threshold", f"{selected_f1:.2%}")
    st.caption(f"Avg predicted labels per case at {selected_threshold:.2f}: {selected_avg_pred_labels:.2f}")

    threshold_plot_df = threshold_df.melt(
        id_vars=["threshold"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    threshold_plot_df["metric"] = threshold_plot_df["metric"].map(
        {
            "precision": "Precision (micro)",
            "recall": "Recall (micro)",
            "f1": "F1 (micro)",
        }
    )

    threshold_fig = px.line(
        threshold_plot_df,
        x="threshold",
        y="score",
        color="metric",
        markers=True,
        color_discrete_sequence=["#1D4ED8", "#0EA5E9", "#14B8A6"],
        labels={"threshold": "Threshold", "score": "Score"},
        title="Micro metrics vs probability threshold",
    )
    threshold_fig.update_layout(yaxis_tickformat=".0%")
    threshold_fig.add_vline(
        x=selected_threshold,
        line_dash="dash",
        line_color="#334155",
        annotation_text=f"Selected {selected_threshold:.2f}",
        annotation_position="top",
    )
    if abs(recommended_threshold - selected_threshold) > 1e-9:
        threshold_fig.add_vline(
            x=recommended_threshold,
            line_dash="dot",
            line_color="#0F766E",
            annotation_text=f"Recommended {recommended_threshold:.2f}",
            annotation_position="top right",
        )
    st.plotly_chart(threshold_fig, use_container_width=True)
    with st.expander("Threshold metrics table"):
        threshold_display_df = threshold_df.assign(
            precision_pct=threshold_df["precision"] * 100,
            recall_pct=threshold_df["recall"] * 100,
            f1_pct=threshold_df["f1"] * 100,
        ).rename(
            columns={
                "threshold": "Threshold",
                "precision_pct": "Precision (%)",
                "recall_pct": "Recall (%)",
                "f1_pct": "F1 (%)",
                "avg_pred_labels_per_case": "Avg predicted labels/case",
            }
        )[
            [
                "Threshold",
                "Precision (%)",
                "Recall (%)",
                "F1 (%)",
                "Avg predicted labels/case",
            ]
        ]
        st.download_button(
            "Download threshold metrics CSV",
            data=_to_csv_bytes(threshold_display_df),
            file_name="threshold_metrics.csv",
            mime="text/csv",
            key="download_threshold_metrics_csv",
        )
        st.dataframe(
            threshold_display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Per-Class F1 Score")
    top_n = st.slider(
        "Number of classes to display",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        key="model_top_n_classes",
    )
    report = metrics["classification_report"]
    skip = {"micro avg", "macro avg", "weighted avg", "samples avg"}
    per_class = [
        {
            "condition": condition,
            "f1": values["f1-score"],
            "precision": values["precision"],
            "recall": values["recall"],
            "support": values["support"],
        }
        for condition, values in report.items()
        if condition not in skip
    ]
    class_df = pd.DataFrame(per_class).sort_values("f1", ascending=False).head(top_n)
    class_df_display = class_df.assign(
        f1_pct=class_df["f1"] * 100,
        precision_pct=class_df["precision"] * 100,
        recall_pct=class_df["recall"] * 100,
    )

    fig = px.bar(
        class_df_display,
        x="f1_pct",
        y="condition",
        orientation="h",
        color="f1_pct",
        color_continuous_scale="Teal",
        hover_data={"precision_pct": ":.2f", "recall_pct": ":.2f", "support": True},
        labels={"f1_pct": "F1 score (%)", "condition": "Skin condition"},
        title=f"Top {top_n} conditions by F1",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Top-class performance table"):
        top_class_export_df = class_df_display.rename(
            columns={
                "condition": "Condition",
                "f1_pct": "F1 (%)",
                "precision_pct": "Precision (%)",
                "recall_pct": "Recall (%)",
                "support": "Support",
            }
        )[["Condition", "F1 (%)", "Precision (%)", "Recall (%)", "Support"]]
        st.download_button(
            "Download top-class table CSV",
            data=_to_csv_bytes(top_class_export_df),
            file_name="top_class_performance.csv",
            mime="text/csv",
            key="download_top_class_csv",
        )
        st.dataframe(
            top_class_export_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("What is your model predicting most?")
    predicted_counts = y_pred.sum(axis=0).astype(int)

    true_counts = Counter()
    for labels in test_df["label_parsed"]:
        true_counts.update(labels)

    predicted_df = pd.DataFrame(
        {
            "condition": class_names,
            "predicted_count": predicted_counts,
            "predicted_rate": predicted_counts / len(test_df),
            "true_count": [true_counts.get(condition, 0) for condition in class_names],
        }
    )
    predicted_df = predicted_df[predicted_df["predicted_count"] > 0].sort_values(
        "predicted_count", ascending=False
    )

    if predicted_df.empty:
        st.info("The model did not produce any positive predictions on the test set.")
    else:
        top_pred_n = st.slider(
            "Number of predicted conditions to display",
            min_value=10,
            max_value=60,
            value=20,
            step=5,
            key="model_top_predicted",
        )
        top_pred_df = predicted_df.head(top_pred_n)
        pred_fig = px.bar(
            top_pred_df,
            x="predicted_count",
            y="condition",
            orientation="h",
            color="predicted_rate",
            color_continuous_scale="Aggrnyl",
            hover_data={"predicted_rate": ":.1%", "true_count": True},
            labels={
                "predicted_count": "Number of predictions",
                "condition": "Skin condition",
                "predicted_rate": "Prediction rate",
            },
            title=f"Top {top_pred_n} most frequently predicted conditions",
        )
        pred_fig.update_layout(
            yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        with st.expander("Predicted frequency table"):
            predicted_export_df = top_pred_df.rename(
                columns={
                    "condition": "Condition",
                    "predicted_count": "Predicted count",
                    "predicted_rate": "Predicted rate",
                    "true_count": "True count",
                }
            )
            st.download_button(
                "Download predicted-frequency CSV",
                data=_to_csv_bytes(predicted_export_df),
                file_name="predicted_frequency.csv",
                mime="text/csv",
                key="download_predicted_frequency_csv",
            )
            st.dataframe(
                predicted_export_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Where does the model fail most?")
    top_error_n = st.slider(
        "Number of error classes to display",
        min_value=10,
        max_value=60,
        value=20,
        step=5,
        key="model_top_error_classes",
    )

    false_positive_counts = np.logical_and(y_pred == 1, y_true == 0).sum(axis=0)
    missed_true_counts = np.logical_and(y_pred == 0, y_true == 1).sum(axis=0)
    error_df = pd.DataFrame(
        {
            "condition": class_names,
            "false_positives": false_positive_counts.astype(int),
            "missed_true_labels": missed_true_counts.astype(int),
        }
    )

    fp_top = error_df[error_df["false_positives"] > 0].nlargest(
        top_error_n, "false_positives"
    )
    fn_top = error_df[error_df["missed_true_labels"] > 0].nlargest(
        top_error_n, "missed_true_labels"
    )

    error_col_a, error_col_b = st.columns(2)

    with error_col_a:
        st.markdown("**Most common false positives**")
        if fp_top.empty:
            st.info("No false positives recorded.")
        else:
            fp_fig = px.bar(
                fp_top.sort_values("false_positives"),
                x="false_positives",
                y="condition",
                orientation="h",
                color_discrete_sequence=["#D64545"],
                labels={
                    "false_positives": "False positive count",
                    "condition": "Skin condition",
                },
                title=f"Top {len(fp_top)} false positives",
            )
            st.plotly_chart(fp_fig, use_container_width=True)

    with error_col_b:
        st.markdown("**Most common missed true labels**")
        if fn_top.empty:
            st.info("No missed labels recorded.")
        else:
            fn_fig = px.bar(
                fn_top.sort_values("missed_true_labels"),
                x="missed_true_labels",
                y="condition",
                orientation="h",
                color_discrete_sequence=["#F97316"],
                labels={
                    "missed_true_labels": "Missed true-label count",
                    "condition": "Skin condition",
                },
                title=f"Top {len(fn_top)} missed true labels",
            )
            st.plotly_chart(fn_fig, use_container_width=True)

    with st.expander("Error hotspot table"):
        error_export_df = error_df.sort_values(
            "missed_true_labels", ascending=False
        ).rename(
            columns={
                "condition": "Condition",
                "false_positives": "False positives",
                "missed_true_labels": "Missed true labels",
            }
        )
        st.download_button(
            "Download error-hotspot CSV",
            data=_to_csv_bytes(error_export_df),
            file_name="error_hotspots.csv",
            mime="text/csv",
            key="download_error_hotspots_csv",
        )
        st.dataframe(
            error_export_df,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Per-condition drilldown")
    condition_support_df = pd.DataFrame(
        {
            "condition": class_names,
            "support": y_true.sum(axis=0).astype(int),
        }
    ).sort_values("support", ascending=False)
    condition_options = condition_support_df["condition"].tolist()
    selected_condition = st.selectbox(
        "Select condition",
        options=condition_options,
        index=0,
        key="model_condition_selected",
    )
    condition_threshold = st.slider(
        "Condition threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(selected_threshold),
        step=0.01,
        key="model_condition_threshold",
    )
    condition_slice_label = st.selectbox(
        "Break down condition by",
        options=list(fairness_fields.keys()),
        key="model_condition_slice",
    )
    min_condition_slice_cases = st.slider(
        "Minimum cases per condition slice",
        min_value=5,
        max_value=120,
        value=20,
        step=5,
        key="model_condition_slice_min_cases",
    )

    condition_idx = class_to_idx[selected_condition]
    condition_true = y_true[:, condition_idx].astype(np.uint8)
    condition_proba = y_proba[:, condition_idx]
    condition_pred = (condition_proba >= condition_threshold).astype(np.uint8)

    condition_tp = int(np.logical_and(condition_true == 1, condition_pred == 1).sum())
    condition_fp = int(np.logical_and(condition_true == 0, condition_pred == 1).sum())
    condition_fn = int(np.logical_and(condition_true == 1, condition_pred == 0).sum())
    condition_tn = int(np.logical_and(condition_true == 0, condition_pred == 0).sum())
    condition_precision = (
        condition_tp / (condition_tp + condition_fp)
        if (condition_tp + condition_fp)
        else 0.0
    )
    condition_recall = (
        condition_tp / (condition_tp + condition_fn)
        if (condition_tp + condition_fn)
        else 0.0
    )
    condition_f1 = (
        2 * condition_precision * condition_recall / (condition_precision + condition_recall)
        if (condition_precision + condition_recall)
        else 0.0
    )

    condition_cols = st.columns(6)
    condition_cols[0].metric("Support", f"{int(condition_true.sum())}")
    condition_cols[1].metric("TP", f"{condition_tp}")
    condition_cols[2].metric("FP", f"{condition_fp}")
    condition_cols[3].metric("FN", f"{condition_fn}")
    condition_cols[4].metric("Precision", f"{condition_precision:.2%}")
    condition_cols[5].metric("Recall", f"{condition_recall:.2%}")
    st.caption(
        f"Condition F1 at threshold {condition_threshold:.2f}: {condition_f1:.2%} | TN: {condition_tn}"
    )

    condition_slice_col = fairness_fields[condition_slice_label]
    condition_slice_base = test_df.assign(
        _slice=test_df[condition_slice_col].fillna("Unknown").astype(str)
    )
    condition_slice_rows = []
    for group_name, group_df in condition_slice_base.groupby("_slice", dropna=False):
        group_case_count = len(group_df)
        if group_case_count < min_condition_slice_cases:
            continue
        group_idx = group_df.index.to_numpy()
        group_true = condition_true[group_idx]
        group_pred = condition_pred[group_idx]
        group_tp = int(np.logical_and(group_true == 1, group_pred == 1).sum())
        group_fp = int(np.logical_and(group_true == 0, group_pred == 1).sum())
        group_fn = int(np.logical_and(group_true == 1, group_pred == 0).sum())
        group_precision = group_tp / (group_tp + group_fp) if (group_tp + group_fp) else 0.0
        group_recall = group_tp / (group_tp + group_fn) if (group_tp + group_fn) else 0.0
        group_f1 = (
            2 * group_precision * group_recall / (group_precision + group_recall)
            if (group_precision + group_recall)
            else 0.0
        )
        condition_slice_rows.append(
            {
                "group": str(group_name),
                "cases": group_case_count,
                "support": int(group_true.sum()),
                "predicted_positive": int(group_pred.sum()),
                "tp": group_tp,
                "fp": group_fp,
                "fn": group_fn,
                "precision": group_precision,
                "recall": group_recall,
                "f1": group_f1,
            }
        )

    if not condition_slice_rows:
        st.warning(
            "No condition slices meet the minimum-case filter. "
            "Lower the slice threshold to inspect subgroup behavior."
        )
    else:
        condition_slice_df = pd.DataFrame(condition_slice_rows).sort_values(
            "f1", ascending=False
        )
        condition_slice_long = condition_slice_df.melt(
            id_vars=["group", "cases", "support"],
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="score",
        )
        condition_slice_long["metric"] = condition_slice_long["metric"].map(
            {
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
            }
        )
        condition_slice_fig = px.bar(
            condition_slice_long,
            x="group",
            y="score",
            color="metric",
            barmode="group",
            labels={"group": condition_slice_label, "score": "Score"},
            hover_data={"cases": True, "support": True, "score": ":.2%"},
            color_discrete_sequence=["#1D4ED8", "#0EA5E9", "#14B8A6"],
            title=f"{selected_condition}: metrics by {condition_slice_label}",
        )
        condition_slice_fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-20)
        st.plotly_chart(condition_slice_fig, use_container_width=True)

        with st.expander("Condition slice table"):
            condition_slice_export_df = condition_slice_df.assign(
                precision_pct=condition_slice_df["precision"] * 100,
                recall_pct=condition_slice_df["recall"] * 100,
                f1_pct=condition_slice_df["f1"] * 100,
            ).rename(
                columns={
                    "group": condition_slice_label,
                    "cases": "Cases",
                    "support": "Support",
                    "predicted_positive": "Predicted positive",
                    "tp": "TP",
                    "fp": "FP",
                    "fn": "FN",
                    "precision_pct": "Precision (%)",
                    "recall_pct": "Recall (%)",
                    "f1_pct": "F1 (%)",
                }
            )[
                [
                    condition_slice_label,
                    "Cases",
                    "Support",
                    "Predicted positive",
                    "TP",
                    "FP",
                    "FN",
                    "Precision (%)",
                    "Recall (%)",
                    "F1 (%)",
                ]
            ]
            st.download_button(
                "Download condition-slice CSV",
                data=_to_csv_bytes(condition_slice_export_df),
                file_name="condition_slice_metrics.csv",
                mime="text/csv",
                key="download_condition_slice_csv",
            )
            st.dataframe(
                condition_slice_export_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Performance by demographic slice")
    fairness_label = st.selectbox(
        "Slice metric by",
        options=list(fairness_fields.keys()),
        key="model_fairness_slice",
    )
    min_slice_cases = st.slider(
        "Minimum cases per slice",
        min_value=5,
        max_value=120,
        value=20,
        step=5,
        key="model_fairness_min_cases",
    )

    fairness_col = fairness_fields[fairness_label]
    grouped_test_df = test_df.assign(_slice=test_df[fairness_col].fillna("Unknown").astype(str))
    slice_rows = []
    for group_name, group_df in grouped_test_df.groupby("_slice", dropna=False):
        case_count = len(group_df)
        if case_count < min_slice_cases:
            continue
        group_idx = group_df.index.to_numpy()
        group_true = y_true[group_idx]
        group_pred = y_pred[group_idx]
        group_precision, group_recall, group_f1 = _micro_precision_recall_f1(
            group_true, group_pred
        )
        slice_rows.append(
            {
                "group": str(group_name),
                "cases": case_count,
                "precision_micro": group_precision,
                "recall_micro": group_recall,
                "f1_micro": group_f1,
                "avg_true_labels_per_case": float(group_true.sum() / case_count),
                "avg_pred_labels_per_case": float(group_pred.sum() / case_count),
            }
        )

    if not slice_rows:
        st.warning(
            "No slice has enough cases for this minimum threshold. "
            "Lower the minimum-cases slider to view subgroup metrics."
        )
    else:
        slice_df = pd.DataFrame(slice_rows).sort_values("f1_micro", ascending=False)
        st.caption(
            f"Showing {len(slice_df)} groups with at least {min_slice_cases} cases."
        )
        best_slice = slice_df.iloc[0]
        worst_slice = slice_df.iloc[-1]
        slice_gap_pp = (best_slice["f1_micro"] - worst_slice["f1_micro"]) * 100

        bias_col1, bias_col2 = st.columns([1, 2])
        bias_col1.metric(
            "Largest group F1 gap (percentage points)",
            f"{slice_gap_pp:.2f}",
        )
        bias_col2.caption(
            f"Best: {best_slice['group']} ({best_slice['f1_micro']:.2%}) | "
            f"Lowest: {worst_slice['group']} ({worst_slice['f1_micro']:.2%})"
        )
        if slice_gap_pp >= 10:
            st.warning(
                "Bias alert: subgroup F1 spread is >= 10 percentage points. "
                "Review thresholding or class weighting before deployment decisions."
            )
        elif slice_gap_pp >= 5:
            st.info(
                "Moderate subgroup spread detected (>= 5 percentage points). "
                "Monitor this gap as you iterate."
            )

        slice_long_df = slice_df.melt(
            id_vars=["group", "cases"],
            value_vars=["precision_micro", "recall_micro", "f1_micro"],
            var_name="metric",
            value_name="score",
        )
        slice_long_df["metric"] = slice_long_df["metric"].map(
            {
                "precision_micro": "Precision (micro)",
                "recall_micro": "Recall (micro)",
                "f1_micro": "F1 (micro)",
            }
        )

        fairness_fig = px.bar(
            slice_long_df,
            x="group",
            y="score",
            color="metric",
            barmode="group",
            labels={"group": fairness_label, "score": "Score"},
            hover_data={"cases": True, "score": ":.2%"},
            color_discrete_sequence=["#1D4ED8", "#0EA5E9", "#14B8A6"],
            title=f"Micro metrics across {fairness_label} groups",
        )
        fairness_fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-20)
        st.plotly_chart(fairness_fig, use_container_width=True)

        with st.expander("Slice metrics table"):
            slice_export_df = slice_df.assign(
                precision_pct=slice_df["precision_micro"] * 100,
                recall_pct=slice_df["recall_micro"] * 100,
                f1_pct=slice_df["f1_micro"] * 100,
            ).rename(
                columns={
                    "group": fairness_label,
                    "cases": "Cases",
                    "precision_pct": "Precision (%)",
                    "recall_pct": "Recall (%)",
                    "f1_pct": "F1 (%)",
                    "avg_true_labels_per_case": "Avg true labels/case",
                    "avg_pred_labels_per_case": "Avg predicted labels/case",
                }
            )[
                [
                    fairness_label,
                    "Cases",
                    "Precision (%)",
                    "Recall (%)",
                    "F1 (%)",
                    "Avg true labels/case",
                    "Avg predicted labels/case",
                ]
            ]
            st.download_button(
                "Download slice-metrics CSV",
                data=_to_csv_bytes(slice_export_df),
                file_name="demographic_slice_metrics.csv",
                mime="text/csv",
                key="download_slice_metrics_csv",
            )
            st.dataframe(
                slice_export_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Interpretation")
    micro_f1 = float(metrics["f1_micro"])
    macro_f1 = float(metrics["f1_macro"])
    micro_precision = float(metrics["precision_micro"])
    micro_recall = float(metrics["recall_micro"])
    hamming_loss = float(metrics["hamming_loss"])
    f1_gap = micro_f1 - macro_f1
    precision_recall_gap = micro_precision - micro_recall

    interpretation_cols = st.columns(3)
    interpretation_cols[0].metric(
        "Micro-Macro F1 Gap (percentage points)", f"{f1_gap * 100:.2f}"
    )
    interpretation_cols[1].metric(
        "Precision-Recall Gap (micro, percentage points)",
        f"{precision_recall_gap * 100:.2f}",
    )
    interpretation_cols[2].metric("Headline F1 (micro)", f"{micro_f1:.2%}")

    model_behavior = "conservative" if precision_recall_gap > 0 else "aggressive"
    st.markdown(
        f"""
- **Hamming loss is {hamming_loss:.2%}.** This stays low in sparse-label settings, so it should be read alongside F1.
- **Micro F1 is {micro_f1:.2%} and macro F1 is {macro_f1:.2%}.** The gap ({f1_gap * 100:.2f} percentage points) indicates stronger performance on common classes than rare classes.
- **Micro precision ({micro_precision:.2%}) vs recall ({micro_recall:.2%})** suggests the model is currently **{model_behavior}** in its label predictions.
        """
    )


# ── Page 2: Data Explorer ─────────────────────────────────────────────────────

elif page == "Data Explorer":
    st.header("Data Explorer")
    st.markdown(
        "Profile the training and test splits, inspect label distribution shape, and review demographic coverage."
    )

    if not _check_artifacts([PROCESSED_DIR / "train.csv", PROCESSED_DIR / "test.csv"]):
        st.stop()

    train_df, test_df = load_csvs()
    train_df = train_df.copy()
    train_df["num_labels"] = train_df["label_parsed"].apply(len)

    label_counts: Counter[str] = Counter()
    for labels in train_df["label_parsed"]:
        label_counts.update(labels)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Train cases", f"{len(train_df):,}")
    summary_cols[1].metric("Test cases", f"{len(test_df):,}")
    summary_cols[2].metric("Unique conditions", f"{len(label_counts):,}")
    summary_cols[3].metric("Avg labels / case", f"{train_df['num_labels'].mean():.2f}")

    st.divider()
    tab_a, tab_b, tab_c = st.tabs(
        ["Condition Frequency", "Label Density", "Demographics"]
    )

    with tab_a:
        top_n_conditions = st.slider(
            "Top conditions to display",
            min_value=10,
            max_value=60,
            value=20,
            step=5,
            key="data_top_n_conditions",
        )
        label_freq_df = pd.DataFrame(
            label_counts.most_common(top_n_conditions), columns=["condition", "count"]
        )
        fig = px.bar(
            label_freq_df,
            x="count",
            y="condition",
            orientation="h",
            color="count",
            color_continuous_scale="Tealgrn",
            labels={"count": "Number of cases", "condition": "Skin condition"},
            title=f"Top {top_n_conditions} conditions in training set",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        label_dist = train_df["num_labels"].value_counts().reset_index()
        label_dist.columns = ["num_labels", "count"]
        label_dist = label_dist.sort_values("num_labels")
        label_dist["share"] = label_dist["count"] / len(train_df)

        show_pct = st.toggle("Show percentage labels", key="show_label_pct")
        y_col = "share" if show_pct else "count"
        y_axis_label = "Share of cases" if show_pct else "Number of cases"
        text_template = "%{y:.1%}" if show_pct else "%{y}"

        fig2 = px.bar(
            label_dist,
            x="num_labels",
            y=y_col,
            labels={"num_labels": "Number of labels per case", y_col: y_axis_label},
            title="How many conditions appear per case?",
            color_discrete_sequence=["#0EA5E9"],
        )
        fig2.update_traces(texttemplate=text_template, textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_c:
        demographic_fields = {
            "Age Group": "age_group",
            "Fitzpatrick Skin Type": "fitzpatrick_skin_type",
            "Race": "combined_race",
            "Sex at Birth": "sex_at_birth",
        }
        demographic_label = st.selectbox(
            "Demographic variable",
            options=list(demographic_fields.keys()),
        )
        demographic_col = demographic_fields[demographic_label]
        demographic_df = (
            train_df[demographic_col]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        demographic_df.columns = [demographic_col, "count"]
        demographic_df["share"] = demographic_df["count"] / len(train_df)

        show_demo_pct = st.toggle("Show percentages", key="show_demo_pct")
        demo_y_col = "share" if show_demo_pct else "count"
        demo_y_axis_label = "Share of cases" if show_demo_pct else "Number of cases"
        demo_text_template = "%{y:.1%}" if show_demo_pct else "%{y}"

        fig3 = px.bar(
            demographic_df,
            x=demographic_col,
            y=demo_y_col,
            labels={demographic_col: demographic_label, demo_y_col: demo_y_axis_label},
            title=f"Distribution of {demographic_label}",
            color_discrete_sequence=["#14B8A6"],
        )
        fig3.update_traces(texttemplate=demo_text_template, textposition="outside")
        fig3.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig3, use_container_width=True)


# ── Page 3: Prediction Explorer ───────────────────────────────────────────────

elif page == "Prediction Explorer":
    st.header("Prediction Explorer")
    st.markdown(
        "Search for a test case, set the confidence threshold, and compare predicted labels against ground truth."
    )

    if not _check_artifacts([MODEL_PATH, PROCESSED_DIR / "embeddings_test.npz", PROCESSED_DIR / "test.csv"]):
        st.stop()

    _, test_df = load_csvs()
    Y_pred, Y_proba, class_names = load_predictions()

    case_options = {
        _format_case_option(i, test_df.iloc[i]): i for i in range(len(test_df))
    }
    selected_case = st.selectbox(
        "Search/select test case",
        options=list(case_options.keys()),
        index=0,
        help="Type here to search by case id, primary label, age group, or sex.",
    )
    case_idx = case_options[selected_case]

    controls_col1, controls_col2 = st.columns(2)
    top_k = controls_col1.slider(
        "Top predictions to inspect",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
    )
    threshold = controls_col2.slider(
        "Confidence threshold",
        min_value=0.00,
        max_value=1.00,
        value=0.20,
        step=0.01,
    )

    row = test_df.iloc[case_idx]
    true_labels = list(row["label_parsed"])
    true_label_set = set(true_labels)
    proba_row = Y_proba[case_idx]

    ranked_idx = np.argsort(proba_row)[::-1]
    top_idx = ranked_idx[:top_k]
    top_predictions_df = pd.DataFrame(
        [
            {
                "condition": class_names[i],
                "confidence": float(proba_row[i]),
                "is_true_label": class_names[i] in true_label_set,
            }
            for i in top_idx
        ]
    )

    top_predictions_df["status"] = np.select(
        [
            top_predictions_df["confidence"] < threshold,
            top_predictions_df["is_true_label"],
        ],
        ["Below threshold", "Correct"],
        default="Incorrect",
    )

    above_threshold_df = top_predictions_df[top_predictions_df["confidence"] >= threshold]
    predicted_set = set(above_threshold_df["condition"])
    recovered_labels = sorted(predicted_set & true_label_set)
    missed_labels = sorted(true_label_set - predicted_set)
    extra_labels = sorted(predicted_set - true_label_set)

    default_pred_labels = [
        class_names[i]
        for i, value in enumerate(Y_pred[case_idx])
        if int(value) == 1
    ]

    summary_cols = st.columns(4)
    summary_cols[0].metric("True labels", f"{len(true_label_set)}")
    summary_cols[1].metric("Recovered labels", f"{len(recovered_labels)}")
    summary_cols[2].metric("Missed labels", f"{len(missed_labels)}")
    summary_cols[3].metric("Extra labels", f"{len(extra_labels)}")

    st.divider()
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Ground truth**")
        _render_label_tokens(sorted(true_label_set), tone="success")

        st.markdown(f"**Recovered above threshold ({threshold:.2f})**")
        _render_label_tokens(recovered_labels, tone="success")

        st.markdown("**Missed labels**")
        _render_label_tokens(missed_labels, tone="danger")

        st.markdown("**Default model predictions (0.50 cutoff)**")
        _render_label_tokens(sorted(default_pred_labels), tone="neutral")

        with st.expander("Case metadata"):
            st.json(
                {
                    "Age group": row.get("age_group", "—"),
                    "Sex at birth": row.get("sex_at_birth", "—"),
                    "Fitzpatrick type": row.get("fitzpatrick_skin_type", "—"),
                    "Race": row.get("combined_race", "—"),
                    "Num images": row.get("num_images", "—"),
                }
            )

    with col_b:
        fig = px.bar(
            top_predictions_df.sort_values("confidence"),
            x="confidence",
            y="condition",
            orientation="h",
            color="status",
            color_discrete_map=STATUS_COLORS,
            labels={"confidence": "Model confidence", "condition": "Predicted condition"},
            title=f"Top {top_k} predictions for case {case_idx}",
            range_x=[0, 1],
        )
        fig.add_vline(
            x=threshold,
            line_width=2,
            line_dash="dash",
            line_color="#334155",
            annotation_text=f"Threshold {threshold:.2f}",
            annotation_position="top",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green: true label. Red: false label. Gray: below selected threshold.")

        with st.expander("Prediction table"):
            st.dataframe(
                top_predictions_df.rename(
                    columns={
                        "condition": "Condition",
                        "confidence": "Confidence",
                        "status": "Status",
                    }
                )[["Condition", "Confidence", "Status"]],
                use_container_width=True,
                hide_index=True,
            )
