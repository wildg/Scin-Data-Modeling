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
- Optional: create `train.csv` / `test.csv` / `validate.csv` split (configurable via `--test-size` and `--validate-size`, seed 42)
- Default split (no validation): 3,061 cleaned cases → 2,448 train / 613 test
- Example 70/20/10 split: → ~2,143 train / 613 test / 306 validate

### 3) Embed
- Stream images directly from GCS (no local download needed)
- Pass each image through a frozen ResNet50 backbone (ImageNet pretrained)
- Mean-pool embeddings across 1–3 images per case
- Save as `.npz` files per split: `embeddings_train.npz`, `embeddings_test.npz`, and optionally `embeddings_validate.npz`

### 4) Train
Three models are available, all using the same multi-label approach:
- Binarize the multi-label targets: fit a `MultiLabelBinarizer` on training labels across all unique skin conditions, producing a binary matrix
- Train a classifier (with `OneVsRestClassifier` for LogReg, XGBoost, and LightGBM, or natively multi-output for FFNN)
- Save the classifier and binarizer together as a single `.joblib` artifact

**Logistic Regression** (`--model logreg`): `LogisticRegression(solver="lbfgs", class_weight="balanced")` — fast linear baseline with automatic inverse-frequency class weighting. Saved to `models/baseline_logreg.joblib`.

**XGBoost** (`--model xgboost`): `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, tree_method="hist")` — gradient-boosted trees with configurable `scale_pos_weight` and `min_child_weight` for class imbalance handling. Saved to `models/xgboost_model.joblib`.

**LightGBM** (`--model lightgbm`): `LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.1)` — gradient-boosted trees with efficient histogram-based training and configurable `scale_pos_weight` and `min_child_weight` for class imbalance handling. Saved to `models/lightgbm_model.joblib`.

**Feedforward Neural Network (FFNN)** (`--model ffnn`): `sklearn.neural_network.MLPClassifier` — a small fully-connected classification head trained on embeddings. Typical configuration used in experiments:
- **Architecture:** hidden layers `(768, 256)`
- **Training:** `adam`, `learning_rate_init=5e-4`, `batch_size=64`, `max_iter=300`, `early_stopping=True`, `n_iter_no_change=20`, `random_state=42`
- **Artifact:** `models/ffnn_mlp.joblib` (sklearn MLP classifier + label binarizer)

### 5) Tune (Hyperparameter Tuning)
- Requires a **validation split** (`embeddings_validate.npz`)
- Filters to the **top-K most frequent classes** (default 30) to focus on actionable conditions and improve macro metrics
- Searches hyperparameter combinations, evaluating **macro F1** on the validation set
- Applies **per-class threshold optimisation** after finding the best hyperparameters — sweeps thresholds per class to maximise each class's F1, ensuring balanced predictions across rare and common conditions
- Saves the best model artifact with tuned thresholds, top-K class list, and best parameters

**Logistic Regression:** Grid search over `C` ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0} and `class_weight` ∈ {balanced, None} (12 configurations).

**XGBoost:** Randomised search (default 15 iterations) over `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`, and `min_child_weight`.

**LightGBM:** Randomised search (default 15 iterations) over `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`, and `min_child_weight`.

**FFNN:** Randomised search (default 12 iterations) over `hidden_layer_sizes`, `alpha`, `learning_rate_init`, and `batch_size`.

### 6) Evaluate
- Load test embeddings and the saved model artifact
- Binarize test labels using the **training** binarizer (no label leakage)
- Automatically detects tuned models (top-K filtering + per-class thresholds) vs legacy models
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
│       ├── validate.csv              # only created when --validate-size > 0
│       ├── embeddings_train.npz
│       ├── embeddings_test.npz
│       └── embeddings_validate.npz  # only created when validate split exists
├── models/
│   ├── baseline_logreg.joblib   # saved after training logreg
│   ├── xgboost_model.joblib     # saved after training xgboost
│   ├── lightgbm_model.joblib    # saved after training lightgbm
│   └── ffnn_mlp.joblib          # saved after training ffnn (sklearn MLP)
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
    │   ├── xgboost_model.py     # XGBoost model
    │   ├── lightgbm_model.py    # LightGBM model
    │   ├── ffnn_model.py        # feedforward neural network (MLP) model
    │   └── tune.py              # hyperparameter tuning (all models)
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
- `streamlit`, `plotly` (dashboard only)

### Installation
```bash
# Core pipeline
uv sync

# Core pipeline + Streamlit dashboard
uv sync --group dashboard
```

## Usage

### Quick Start (Recommended Pipeline)

```bash
# 1. Download CSVs
uv run scin_data_modeling download --no-images

# 2. Preprocess and create train/test/validate split (70/20/10)
uv run scin_data_modeling preprocess --create-split --test-size 0.2 --validate-size 0.1 --seed 42

# 3. Generate image embeddings for all splits (streams from GCS, no local images needed)
uv run scin_data_modeling embed --backbone resnet50 --device cpu --split all

# 4. Tune models using the validation set (recommended)
uv run scin_data_modeling tune --model logreg
uv run scin_data_modeling tune --model xgboost
uv run scin_data_modeling tune --model lightgbm
uv run scin_data_modeling tune --model ffnn

# 5. Evaluate on the test set
uv run scin_data_modeling evaluate --model logreg
uv run scin_data_modeling evaluate --model xgboost
uv run scin_data_modeling evaluate --model lightgbm
uv run scin_data_modeling evaluate --model ffnn
```

> **Note:** You can also train models without tuning using `uv run scin_data_modeling train --mode frozen --model logreg`. Tuning is recommended as it searches for optimal hyperparameters, applies top-K class filtering, and optimises per-class decision thresholds.

### Command Reference

| Command | Key Flags | Description |
|---------|-----------|-------------|
| `download` | `--raw-dir`, `--images/--no-images` | Download SCIN CSVs (and optionally images) from GCS |
| `preprocess` | `--create-split`, `--test-size`, `--validate-size`, `--seed` | Merge and clean CSVs; optionally create train/test/validate splits |
| `embed` | `--backbone` (`resnet50`, `efficientnet_b0`), `--split` (`train`, `test`, `validate`, `both`, `all`), `--device` (`cpu`, `cuda`, `mps`) | Stream images from GCS and extract embeddings per split |
| `train` | `--mode frozen`, `--model` (`logreg`, `xgboost`, `lightgbm`, `ffnn`) | Train a classifier on cached embeddings (use `tune` instead for best results) |
| `tune` | `--model`, `--top-k` (default 30), `--n-iter` (default 15) | Hyperparameter search + per-class threshold optimisation on the validation set |
| `evaluate` | `--model` | Evaluate a trained/tuned model on the test set |

All commands accept `--processed-dir` (default `data/processed`) and `--model-dir` (default `models`) for custom directory paths.

### Output artifacts

| File | Description |
|------|-------------|
| `data/processed/cleaned.csv` | All 3,061 cleaned cases |
| `data/processed/train.csv` | Training cases (size determined by `--test-size` and `--validate-size`) |
| `data/processed/test.csv` | Test cases (fraction set by `--test-size`, default 0.2) |
| `data/processed/validate.csv` | Validation cases (fraction set by `--validate-size`; not created when 0.0) |
| `data/processed/embeddings_train.npz` | ResNet50 features for training cases (N × 2048) |
| `data/processed/embeddings_test.npz` | ResNet50 features for test cases (N × 2048) |
| `data/processed/embeddings_validate.npz` | ResNet50 features for validation cases (N × 2048; only when validate split exists) |
| `models/baseline_logreg.joblib` | Logistic regression classifier + label binarizer (+ top-K indices, thresholds, and best params when tuned) |
| `models/xgboost_model.joblib` | XGBoost classifier + label binarizer (+ top-K indices, thresholds, and best params when tuned) |
| `models/lightgbm_model.joblib` | LightGBM classifier + label binarizer (+ top-K indices, thresholds, and best params when tuned) |
| `models/ffnn_mlp.joblib` | FFNN (sklearn MLP) classifier + label binarizer (+ top-K indices, thresholds, and best params when tuned) |

### Streamlit Dashboard

An interactive dashboard for exploring the data and model results. Requires the model to be trained first.

**Install dashboard dependencies:**
```bash
uv sync --group dashboard
```

**Launch the dashboard:**
```bash
uv run --group dashboard streamlit run app.py
```

Opens automatically at `http://localhost:8501`. Three pages are available via the sidebar:

| Page | Contents |
|------|----------|
| **Model Performance** | 8 summary metric cards, per-class F1 bar chart for top 20 conditions, metric interpretation |
| **Data Explorer** | Top 20 most common conditions, labels-per-case histogram, demographic breakdowns (age, Fitzpatrick skin type, race, sex) |
| **Prediction Explorer** | Select any test case (0–612) to see true labels vs model predictions with confidence scores, color-coded correct/incorrect |

## Results

All four models were tuned using the validation set with **top-K=30 class filtering** and **per-class threshold optimisation**, then evaluated on the held-out test set (**613 cases**, 1,047 label instances across 30 skin conditions).

| Metric | Logistic Regression | XGBoost | LightGBM | FFNN (sklearn MLP) |
|---|---:|---:|---:|---:|
| Hamming Loss | 0.1506 | 0.1035 | 0.1027 | 0.0909 |
| F1 (micro) | 0.2733 | 0.3052 | 0.3058 | 0.2757 |
| F1 (macro) | 0.1588 | 0.1567 | 0.1350 | 0.1424 |
| F1 (weighted) | 0.2772 | 0.2829 | 0.2604 | 0.2518 |
| Precision (micro) | 0.1884 | 0.2470 | 0.2485 | 0.2524 |
| Recall (micro) | 0.4976 | 0.3992 | 0.3973 | 0.3037 |
| Precision (macro) | 0.1378 | 0.1537 | 0.1316 | 0.1855 |
| Recall (macro) | 0.2720 | 0.2049 | 0.1849 | 0.1605 |

### Interpretation

- **LightGBM and XGBoost are near-tied for best micro F1** (0.3058 vs 0.3052), both substantially outperforming the linear baseline (0.2733) and FFNN (0.2757).
- **Logistic regression achieves the highest recall** (micro 0.4976, macro 0.2720) and best macro F1 (0.1588), catching more rare conditions at the cost of higher hamming loss (0.1506).
- **FFNN achieves the lowest hamming loss** (0.0909) and highest macro precision (0.1855), making it the most conservative predictor with the fewest per-label errors.
- **Clear precision-recall tradeoff across models:** logistic regression favors recall, FFNN favors precision, and the boosted-tree models (XGBoost/LightGBM) strike a middle ground.
- **Macro metrics remain low across all models** (macro F1 0.13–0.16), confirming that rare-class prediction is the primary challenge even after top-K filtering and threshold optimisation.

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

*Last Updated: March 2026*

