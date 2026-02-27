# SCIN Data Modeling Pipeline

**INSY 674 Group Project**

## Project Overview

This project builds an end-to-end pipeline for the SCIN (Skin Condition Image Network) dataset to predict multi-label skin condition diagnoses:
- Download raw SCIN metadata and images
- Merge and clean case + label data
- Extract image embeddings via a frozen ResNet50 backbone
- Train and evaluate classification models

## Objectives

- Standardized ingestion from the SCIN public bucket
- Consistent cleaned dataset generation with multi-label support
- Image feature extraction via frozen CNN backbones
- Modular model training and evaluation pipeline

## Dataset

The SCIN dataset contains dermatological cases with comprehensive clinical information:

### Files
- `dataset_scin_cases.csv` - Patient demographics, symptoms, affected body parts, and clinical metadata (5,033 cases, 57 features)
- `dataset_scin_labels.csv` - Dermatologist-assigned diagnoses, confidence scores, and skin type assessments (5,033 cases, 17 features)

### Key Features
**Demographics:**
- Age groups, sex at birth
- Fitzpatrick skin types (self-reported and dermatologist-assessed)
- Monk skin tone scales (India and US)
- Race and ethnicity information

**Clinical Characteristics:**
- Skin texture (raised/bumpy, flat, rough/flaky, fluid-filled)
- Affected body parts (head/neck, arms, legs, torso, etc.)
- Condition symptoms (itching, pain, bleeding, darkening, etc.)
- Other systemic symptoms (fever, fatigue, joint pain, etc.)
- Condition duration and category

**Target Variables:**
- Weighted skin condition labels (dermatologist consensus)
- Skin condition confidence scores
- Image gradability assessments

## Pipeline Stages

### 1) Download
- Download SCIN CSV files (`scin_cases.csv`, `scin_labels.csv`, etc.)
- Optionally download all images

### 2) Preprocess (Cleaning)
- Merge `scin_cases.csv` and `scin_labels.csv` on `case_id`
- Parse `dermatologist_skin_condition_on_label_name`
- Drop rows without usable labels or without image paths
- Build:
  - `image_paths` (JSON list)
  - `num_images`
  - `label_all` (deduplicated full condition list)
  - `label` (first 3 labels, JSON list)
- Save cleaned output to `data/processed/cleaned.csv`
- Optional: create `train.csv` / `test.csv` split (80/20, seed 42)
- Result: 3,061 cleaned cases → 2,448 train / 613 test

### 3) Embed
- Stream images directly from GCS (no local download needed)
- Pass each image through a frozen ResNet50 backbone (ImageNet pretrained)
- Mean-pool embeddings across 1–3 images per case
- Save as `.npz` files: `embeddings_train.npz` (2448 × 2048), `embeddings_test.npz` (613 × 2048)

### 4) Train
Three models are available, all using the same multi-label approach:
- Binarize the multi-label targets: fit a `MultiLabelBinarizer` on training labels across all 370 unique skin conditions, producing a binary matrix of shape (2448 × 370)
- Train a `OneVsRestClassifier` (one binary classifier per condition, parallelized via `n_jobs=-1`)
- Save the classifier and binarizer together as a single `.joblib` artifact

**Logistic Regression** (`--model logreg`): `LogisticRegression(solver="lbfgs")` — fast linear baseline. Saved to `models/baseline_logreg.joblib`.

**XGBoost** (`--model xgboost`): `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist")` — gradient-boosted trees that can learn non-linear interactions in the embedding space. Saved to `models/xgboost_model.joblib`.

**LightGBM** (`--model lightgbm`): `LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1)` - gradient-boosted trees with efficient histogram-based training. Saved to `models/lightgbm_model.joblib`.

### 5) Evaluate
- Load test embeddings and the saved model artifact
- Binarize test labels using the **training** binarizer (no label leakage)
- Compute multi-label metrics and print a results table

## Repository Structure

```
Scin-Data-Modeling/
├── README.md
├── data/
│   ├── raw/
│   │   └── dataset/
│   │       ├── scin_cases.csv
│   │       ├── scin_labels.csv
│   │       └── images/
│   └── processed/
│       ├── cleaned.csv
│       ├── train.csv
│       ├── test.csv
│       ├── embeddings_train.npz
│       └── embeddings_test.npz
├── models/
│   ├── baseline_logreg.joblib   # saved after training logreg
│   └── xgboost_model.joblib     # saved after training xgboost
└── scin_data_modeling/
    ├── cli.py
    ├── data/
    │   ├── download.py
    │   ├── preprocess.py
    │   ├── embed.py
    │   └── streaming.py
    ├── models/
    │   ├── backbone.py          # ResNet50 / EfficientNet-B0
    │   ├── baseline.py          # logistic regression baseline
    │   └── xgboost_model.py     # XGBoost model
    └── evaluation/
        └── metrics.py           # multi-label evaluation metrics
```

## Technical Requirements

### Python Libraries
- `pandas`, `numpy`
- `google-cloud-storage`, `tqdm`
- `typer`, `rich`
- `scikit-learn`, `xgboost`, `lightgbm`
- `torch`, `torchvision`

### Installation
```bash
pip install -e .
```

## Usage

### Full pipeline from scratch

```bash
# 1. Download CSVs
uv run scin_data_modeling download --no-images

# 2. Preprocess and create train/test split
uv run scin_data_modeling preprocess --create-split --test-size 0.2 --seed 42

# 3. Generate image embeddings (streams from GCS, no local images needed)
uv run scin_data_modeling embed --backbone resnet50 --device cpu

# 4. Train the logistic regression baseline
uv run scin_data_modeling train --mode frozen

# 5. Evaluate on the test set
uv run scin_data_modeling evaluate
```

### Run a specific model (if data and embeddings already exist)

All three models use the same `--mode frozen` flag (train on cached embeddings) and the same `--model` flag to select which model to train or evaluate.

```bash
# Logistic regression (default)
uv run scin_data_modeling train --mode frozen --model logreg
uv run scin_data_modeling evaluate --model logreg

# XGBoost
uv run scin_data_modeling train --mode frozen --model xgboost
uv run scin_data_modeling evaluate --model xgboost

# LightGBM
uv run scin_data_modeling train --mode frozen --model lightgbm
uv run scin_data_modeling evaluate --model lightgbm
```

### Options

```bash
# Use a different device for embedding generation (e.g. Apple Silicon)
uv run scin_data_modeling embed --device mps

# Custom data/model directories
uv run scin_data_modeling train --mode frozen --model xgboost --processed-dir data/processed --model-dir models
uv run scin_data_modeling evaluate --model xgboost --processed-dir data/processed --model-dir models

# Same pattern for LightGBM
uv run scin_data_modeling train --mode frozen --model lightgbm --processed-dir data/processed --model-dir models
uv run scin_data_modeling evaluate --model lightgbm --processed-dir data/processed --model-dir models
```

### Output artifacts

| File | Description |
|------|-------------|
| `data/processed/cleaned.csv` | All 3,061 cleaned cases |
| `data/processed/train.csv` | 2,448 training cases |
| `data/processed/test.csv` | 613 test cases |
| `data/processed/embeddings_train.npz` | ResNet50 features (2448 × 2048) |
| `data/processed/embeddings_test.npz` | ResNet50 features (613 × 2048) |
| `models/baseline_logreg.joblib` | Logistic regression classifier + label binarizer |
| `models/xgboost_model.joblib` | XGBoost classifier + label binarizer |
| `models/lightgbm_model.joblib` | LightGBM classifier + label binarizer |

## Baseline Model Results

The logistic regression baseline was evaluated on the 613-case held-out test set. It predicts across 370 unique skin condition classes using ResNet50 image embeddings as features.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Hamming Loss** | 0.0083 | On average, 0.83% of the 370 label slots are wrong per case — low because most labels are correctly predicted as absent |
| **F1 (micro)** | 0.1857 | Aggregate F1 across all label occurrences; the model captures roughly 1 in 5 correct label predictions overall |
| **F1 (macro)** | 0.0163 | Per-class F1 averaged equally across all 370 conditions; very low because many rare conditions get zero predictions |
| **F1 (weighted)** | 0.1583 | Per-class F1 weighted by class frequency; closer to micro F1, dominated by the more common conditions |
| **Precision (micro)** | 0.2973 | When the model predicts a label, it is correct ~30% of the time |
| **Recall (micro)** | 0.1350 | The model finds ~14% of all true labels in the test set |
| **Precision (macro)** | 0.0290 | Averaged per-class; low due to many rare classes with sparse predictions |
| **Recall (macro)** | 0.0128 | Averaged per-class; the model misses most instances of rare conditions |

### Interpreting the results

**Hamming Loss (0.0083)** appears excellent but is misleading for sparse multi-label problems. With 370 classes and only 1–3 true labels per case, the vast majority of label slots are 0 — predicting all zeros would also score well on this metric. Micro F1 is a better headline number.

**Micro F1 (0.1857)** reflects overall performance weighted by label frequency. The model has learned something meaningful from the embeddings — a random baseline on 370 classes would score near zero — but performance is limited, which is expected for a simple linear model on a highly imbalanced 370-class problem with only 2,448 training examples.

**Macro F1 (0.0163)** is low because it treats all 370 conditions equally, including extremely rare ones the model effectively never predicts. This gap between micro (0.19) and macro (0.02) F1 reveals severe class imbalance: a handful of common conditions drive most correct predictions while rare conditions are missed almost entirely.

**Precision > Recall (0.30 vs 0.14)**: The model is conservative — when it does predict a label it is more often right than wrong, but it misses many true labels. This is typical for logistic regression with class imbalance: the classifier learns a high threshold before committing to a positive prediction.

These results represent an expected baseline for a linear model on a difficult 370-class multi-label problem. They establish a performance floor for comparing against more powerful models (neural classification heads, fine-tuned backbones).

## XGBoost Model Results

The XGBoost model was evaluated on the same 613-case held-out test set using `OneVsRestClassifier(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist"))`.

| Metric | Logistic Regression | XGBoost | Change |
|--------|--------------------:|--------:|-------:|
| **Hamming Loss** | 0.0083 | 0.0070 | -15% |
| **F1 (micro)** | 0.1857 | 0.0959 | -48% |
| **F1 (macro)** | 0.0163 | 0.0051 | -69% |
| **F1 (weighted)** | 0.1583 | 0.0777 | -51% |
| **Precision (micro)** | 0.2973 | 0.5180 | +74% |
| **Recall (micro)** | 0.1350 | 0.0528 | -61% |
| **Precision (macro)** | 0.0290 | 0.0237 | -18% |
| **Recall (macro)** | 0.0128 | 0.0036 | -72% |

### Interpreting the XGBoost results

**XGBoost trades recall for precision drastically.** Micro precision jumps from 0.30 to 0.52 — when XGBoost predicts a label it is correct over half the time, a meaningful improvement over logistic regression. However, micro recall falls from 0.14 to 0.05, meaning the model only finds about 1 in 20 true labels. This is an extreme conservative shift: XGBoost raises its internal prediction threshold very high before committing to a positive label.

**F1 (micro) drops from 0.19 to 0.10.** F1 is the harmonic mean of precision and recall, so the large recall drop outweighs the precision gain. Overall, XGBoost is less useful than logistic regression on this dataset despite being a more powerful model.

**Why does this happen?** XGBoost with default `OneVsRestClassifier` wrapping trains each binary classifier independently on highly imbalanced data (most conditions appear in fewer than 5% of cases). XGBoost's decision trees tend to converge on high-confidence predictions only, suppressing positive predictions for rare classes even more aggressively than logistic regression. The result is high precision but very low recall.

**Hamming Loss improves slightly (0.0083 → 0.0070)** because XGBoost predicts fewer positive labels overall — making fewer false positives at the cost of far more false negatives. As noted for the logistic regression, this metric is misleading for sparse multi-label problems.

**Takeaway:** On this dataset, logistic regression outperforms XGBoost overall (higher F1). XGBoost would benefit from class-weight tuning (`scale_pos_weight`) or threshold calibration to rebalance precision and recall. These remain directions for future work.

## LightGBM Model Results

Evaluate LightGBM with:

```bash
uv run scin_data_modeling train --mode frozen --model lightgbm
uv run scin_data_modeling evaluate --model lightgbm
```

LightGBM uses the same multi-label setup:
`OneVsRestClassifier(LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1))`.

| Metric | Value |
|--------|------:|
| **Hamming Loss** | 0.0083 |
| **F1 (micro)** | 0.1847 |
| **F1 (macro)** | 0.0163 |
| **F1 (weighted)** | 0.1576 |
| **Precision (micro)** | 0.2956 |
| **Recall (micro)** | 0.1343 |
| **Precision (macro)** | 0.0299 |
| **Recall (macro)** | 0.0127 |

### Interpreting the LightGBM results

**LightGBM is effectively tied with logistic regression.** Micro F1 is 0.1847 vs 0.1857 for logistic regression, and precision/recall are also nearly identical (0.2956/0.1343 vs 0.2973/0.1350).

**Compared with XGBoost, LightGBM keeps a much better precision-recall balance.** XGBoost reached higher precision (0.5180) but collapsed recall (0.0528), while LightGBM stays close to the baseline trade-off and therefore much higher micro F1.

**Takeaway:** With the current settings (`n_estimators=300`, `max_depth=4`, `learning_rate=0.1`), LightGBM does not materially improve over logistic regression on this dataset, but it avoids the severe recall drop observed with XGBoost. Next steps are class weighting, threshold tuning, and per-label calibration.

## Notes on Labels

- Raw target source: `dermatologist_skin_condition_on_label_name` in `scin_labels.csv`
- Cleaned `label_all`: deduplicated full list of condition names per case
- Cleaned `label`: first 3 items from `label_all` for downstream top-3 prediction workflows

## Contributors

INSY 674 Group Project Team

## License

This project is for educational purposes as part of INSY 674 coursework.

## Acknowledgments

- SCIN (Skin Condition Image Network) dataset providers
- Course instructors and teaching assistants

---

*Last Updated: February 2026*

