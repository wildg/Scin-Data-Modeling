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

### 4) Train — Logistic Regression Baseline
- Load cached ResNet50 embeddings (`embeddings_train.npz`)
- Binarize the multi-label targets: fit a `MultiLabelBinarizer` on training labels across all 370 unique skin conditions, producing a binary matrix of shape (2448 × 370)
- Train a `OneVsRestClassifier` wrapping `LogisticRegression (lbfgs solver)` — one binary classifier per condition, run in parallel via `n_jobs=-1`
- Save the classifier and binarizer together as `models/baseline_logreg.joblib`

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
│   └── baseline_logreg.joblib   # saved after training
└── scin_data_modeling/
    ├── cli.py
    ├── data/
    │   ├── download.py
    │   ├── preprocess.py
    │   ├── embed.py
    │   └── streaming.py
    ├── models/
    │   ├── backbone.py          # ResNet50 / EfficientNet-B0
    │   └── baseline.py          # logistic regression baseline
    └── evaluation/
        └── metrics.py           # multi-label evaluation metrics
```

## Technical Requirements

### Python Libraries
- `pandas`, `numpy`
- `google-cloud-storage`, `tqdm`
- `typer`, `rich`
- `scikit-learn`
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

### Run just the model (if data and embeddings already exist)

```bash
uv run scin_data_modeling train --mode frozen
uv run scin_data_modeling evaluate
```

### Options

```bash
# Use a different device for embedding generation (e.g. Apple Silicon)
uv run scin_data_modeling embed --device mps

# Custom data/model directories
uv run scin_data_modeling train --mode frozen --processed-dir data/processed --model-dir models
uv run scin_data_modeling evaluate --processed-dir data/processed --model-dir models
```

### Output artifacts

| File | Description |
|------|-------------|
| `data/processed/cleaned.csv` | All 3,061 cleaned cases |
| `data/processed/train.csv` | 2,448 training cases |
| `data/processed/test.csv` | 613 test cases |
| `data/processed/embeddings_train.npz` | ResNet50 features (2448 × 2048) |
| `data/processed/embeddings_test.npz` | ResNet50 features (613 × 2048) |
| `models/baseline_logreg.joblib` | Trained classifier + label binarizer |

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

