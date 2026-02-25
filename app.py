"""Streamlit dashboard for the SCIN skin condition prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scin_data_modeling.evaluation.metrics import evaluate_baseline
from scin_data_modeling.models.baseline import predict_baseline

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("models/baseline_logreg.joblib")

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


def _check_artifacts() -> bool:
    missing = [p for p in [MODEL_PATH, PROCESSED_DIR / "embeddings_test.npz"] if not p.exists()]
    if missing:
        st.error(
            "Missing required files:\n"
            + "\n".join(f"- `{p}`" for p in missing)
            + "\n\nRun the pipeline first:\n"
            "```\nuv run scin_data_modeling train --mode frozen\n```"
        )
        return False
    return True


# ── app layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SCIN Model Dashboard", layout="wide")
st.title("SCIN Skin Condition Prediction Dashboard")
st.caption("Logistic regression baseline — ResNet50 embeddings")

page = st.sidebar.radio(
    "Navigate",
    ["Model Performance", "Data Explorer", "Prediction Explorer"],
)

# ── Page 1: Model Performance ─────────────────────────────────────────────────

if page == "Model Performance":
    st.header("Model Performance")
    st.markdown(
        "Evaluation of the logistic regression baseline on the **613-case held-out test set**. "
        "The model predicts across **370 unique skin conditions** using 2048-dimensional ResNet50 image embeddings."
    )

    if not _check_artifacts():
        st.stop()

    metrics = load_metrics()

    # Summary metric cards
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hamming Loss", f"{metrics['hamming_loss']:.4f}", help="Fraction of label slots predicted incorrectly. Low values are good, but misleading for sparse labels.")
    col2.metric("F1 (micro)", f"{metrics['f1_micro']:.4f}", help="F1 weighted by label frequency — the best single headline metric for imbalanced multi-label problems.")
    col3.metric("Precision (micro)", f"{metrics['precision_micro']:.4f}", help="When the model predicts a label, how often it is correct.")
    col4.metric("Recall (micro)", f"{metrics['recall_micro']:.4f}", help="Fraction of all true labels the model successfully finds.")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("F1 (macro)", f"{metrics['f1_macro']:.4f}", help="Per-class F1 averaged equally across all 370 conditions — penalises poor performance on rare classes.")
    col6.metric("F1 (weighted)", f"{metrics['f1_weighted']:.4f}", help="Per-class F1 weighted by class frequency.")
    col7.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
    col8.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")

    st.divider()

    # Per-class F1 bar chart
    st.subheader("Per-Class F1 Score — Top 20 Conditions")
    report = metrics["classification_report"]
    skip = {"micro avg", "macro avg", "weighted avg", "samples avg"}
    per_class = [
        {"condition": k, "f1": v["f1-score"], "precision": v["precision"], "recall": v["recall"], "support": v["support"]}
        for k, v in report.items()
        if k not in skip
    ]
    class_df = pd.DataFrame(per_class).sort_values("f1", ascending=False).head(20)

    fig = px.bar(
        class_df,
        x="f1",
        y="condition",
        orientation="h",
        color="f1",
        color_continuous_scale="Blues",
        hover_data={"precision": ":.3f", "recall": ":.3f", "support": True},
        labels={"f1": "F1 Score", "condition": "Skin Condition"},
        title="Top 20 Conditions by F1 Score",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Interpreting these results")
    st.markdown(
        """
- **Hamming Loss (0.0083)** looks excellent but is deceptive. With 370 classes and only 1–3 true labels per case,
  most label slots are 0. Predicting all zeros would also score well here. Micro F1 is the more honest headline number.

- **Micro F1 (0.19)** shows the model has learned something real from the image embeddings. A random guesser
  on 370 classes would score near zero. Performance is modest but expected for a linear model on this problem.

- **The micro/macro F1 gap (0.19 vs 0.02)** reveals severe class imbalance. Common conditions drive most correct
  predictions; rare conditions are nearly always missed, dragging macro F1 down.

- **Precision > Recall (0.30 vs 0.14)** means the model is conservative — it only predicts a label when fairly
  confident, but misses many true labels in the process.
        """
    )

# ── Page 2: Data Explorer ─────────────────────────────────────────────────────

elif page == "Data Explorer":
    st.header("Data Explorer")
    st.markdown("Exploring the **training set** (2,448 cases) for label distribution and demographics.")

    if not (PROCESSED_DIR / "train.csv").exists():
        st.error("Training data not found. Run the preprocessing pipeline first.")
        st.stop()

    train_df, test_df = load_csvs()

    # Label frequency
    st.subheader("Top 20 Most Common Skin Conditions")
    from collections import Counter
    label_counts: Counter = Counter()
    for labels in train_df["label_parsed"]:
        label_counts.update(labels)
    label_freq_df = pd.DataFrame(label_counts.most_common(20), columns=["condition", "count"])

    fig = px.bar(
        label_freq_df,
        x="count",
        y="condition",
        orientation="h",
        color="count",
        color_continuous_scale="Teal",
        labels={"count": "Number of Cases", "condition": "Skin Condition"},
        title="Top 20 Conditions in Training Set",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Labels per case
    st.subheader("Labels per Case Distribution")
    train_df["num_labels"] = train_df["label_parsed"].apply(len)
    label_dist = train_df["num_labels"].value_counts().reset_index()
    label_dist.columns = ["num_labels", "count"]
    label_dist = label_dist.sort_values("num_labels")

    fig2 = px.bar(
        label_dist,
        x="num_labels",
        y="count",
        text="count",
        labels={"num_labels": "Number of Labels per Case", "count": "Number of Cases"},
        title="How many conditions does each case have?",
        color_discrete_sequence=["#0096C7"],
    )
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Demographics
    st.subheader("Demographic Breakdowns")
    demo_col = st.selectbox(
        "Select demographic variable",
        ["age_group", "fitzpatrick_skin_type", "combined_race", "sex_at_birth"],
    )

    demo_counts = train_df[demo_col].value_counts().reset_index()
    demo_counts.columns = [demo_col, "count"]

    fig3 = px.bar(
        demo_counts,
        x=demo_col,
        y="count",
        text="count",
        labels={"count": "Number of Cases"},
        title=f"Distribution of {demo_col.replace('_', ' ').title()}",
        color_discrete_sequence=["#48CAE4"],
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig3, use_container_width=True)

# ── Page 3: Prediction Explorer ───────────────────────────────────────────────

elif page == "Prediction Explorer":
    st.header("Prediction Explorer")
    st.markdown(
        "Select a test case to see the model's predicted skin conditions vs the true labels. "
        "The bar chart shows the model's confidence score for each of its top predictions."
    )

    if not _check_artifacts():
        st.stop()

    _, test_df = load_csvs()
    Y_pred, Y_proba, class_names = load_predictions()

    case_idx = st.number_input(
        "Test case index", min_value=0, max_value=len(test_df) - 1, value=0, step=1
    )

    true_labels = test_df.iloc[case_idx]["label_parsed"]
    proba_row = Y_proba[case_idx]

    # Top 5 predictions by confidence
    top5_idx = np.argsort(proba_row)[::-1][:5]
    top5_data = [
        {
            "condition": class_names[i],
            "confidence": float(proba_row[i]),
            "correct": class_names[i] in true_labels,
        }
        for i in top5_idx
    ]
    top5_df = pd.DataFrame(top5_data)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**True labels**")
        for label in true_labels:
            st.success(f"✓ {label}")

        st.markdown("**Case metadata**")
        row = test_df.iloc[case_idx]
        st.write({
            "Age group": row.get("age_group", "—"),
            "Sex at birth": row.get("sex_at_birth", "—"),
            "Fitzpatrick type": row.get("fitzpatrick_skin_type", "—"),
            "Num images": row.get("num_images", "—"),
        })

    with col_b:
        top5_df["label"] = top5_df.apply(
            lambda r: f"✓ {r['condition']}" if r["correct"] else r["condition"], axis=1
        )
        fig = px.bar(
            top5_df,
            x="confidence",
            y="label",
            orientation="h",
            color="correct",
            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
            labels={"confidence": "Model Confidence", "label": "Predicted Condition"},
            title=f"Top 5 Predictions — Test Case {case_idx}",
            range_x=[0, 1],
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = correct prediction (matches a true label). Red = incorrect.")
