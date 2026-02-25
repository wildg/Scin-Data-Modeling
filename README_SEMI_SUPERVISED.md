# SCIN Semi-Supervised Learning README

This document explains the semi-supervised learning pipeline used in this repository to predict skin-condition labels from SCIN tabular features.

## 1. Problem Setup

- Goal: predict dermatologist skin-condition label using case metadata/symptoms (tabular features).
- Challenge: many rows are unlabeled (`weighted_skin_condition_label = {}`), often due to poor image quality.
- Strategy: train with both labeled and unlabeled rows using semi-supervised methods on tabular features only.

## 2. Data Used

Input CSV files:
- `data/raw/dataset/scin_cases.csv`
- `data/raw/dataset/scin_labels.csv`

Prepared outputs (from preprocessing):
- `data/processed/train_labeled.csv`
- `data/processed/train_unlabeled.csv`
- `data/processed/test_labeled.csv`
- `data/processed/manifest.json`

## 3. Preprocessing and Feature Engineering

Implemented in:
- `scin_data_modeling/data/preprocess.py`

Key steps:
1. Expand case-level rows to image-level rows while preserving `case_id`.
2. Join labels to each image row by `case_id`.
3. Prevent leakage with case-level split (a case appears in only one split).
4. Convert checkbox-style features (`textures_*`, `body_parts_*`, etc.) to binary.
5. Engineer aggregate features (`n_textures`, `n_body_parts`, `n_condition_symptoms`, `n_other_symptoms`).
6. Handle special categories explicitly:
   - `fitzpatrick_skin_type = NONE_IDENTIFIED`
   - `age_group = AGE_UNKNOWN`
   - `sex_at_birth = OTHER_OR_UNSPECIFIED`
7. Parse soft targets from `weighted_skin_condition_label` and mark unlabeled rows.
8. Scale non-binary numeric features (train-fitted scaling).

## 4. Semi-Supervised Modeling

Implemented in:
- `scin_data_modeling/models/semi_supervised.py`
- Runner script: `scripts/run_semi_supervised_models.py`

Methods benchmarked:
- `Supervised_Only_RF`
- `SelfTraining_RF`
- `LabelSpreading`
- `ClusterPseudoLabel_RF`
- `CoTraining_Consensus_RF`

Current corrections added for performance:
1. Collapse target to top-K classes + `OTHER` (default `top_k=15`).
2. Filter labeled rows by confidence (`target_max_prob >= 0.55` by default).
3. Remove aggressive rare-class weighting in RF.
4. Pseudo-label unlabeled data at case level (not per image row).
5. Use less conservative pseudo-label gates (`threshold=0.6`, `margin=0.1` defaults).

## 5. How to Run

### A) Generate processed data

```bash
uv run python -m scin_data_modeling.data.preprocess --output-dir data/processed
```

### B) Run semi-supervised benchmark and pseudo-labeling

```bash
uv run python scripts/run_semi_supervised_models.py \
  --processed-dir data/processed \
  --output-dir artifacts/semi_supervised
```

### C) Optional tuning example

```bash
uv run python scripts/run_semi_supervised_models.py \
  --processed-dir data/processed \
  --output-dir artifacts/semi_supervised_custom \
  --top-k-classes 20 \
  --min-label-confidence 0.55 \
  --self-training-threshold 0.60 \
  --pseudo-threshold 0.60 \
  --pseudo-margin 0.10
```

## 6. Outputs and How to Read Them

Main outputs:
- `artifacts/semi_supervised/semi_supervised_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_summary.json`
- `artifacts/semi_supervised/unlabeled_pseudo_labels_all.csv`
- `artifacts/semi_supervised/unlabeled_pseudo_labels_accepted.csv`

Interpretation:
- `semi_supervised_metrics.csv`: compare methods by `accuracy` and `f1_macro`.
- `semi_supervised_summary.json`: best method, grouped class count, pseudo-label acceptance rates.
- `unlabeled_pseudo_labels_accepted.csv`: pseudo-labels considered reliable enough to include.

## 7. Current Baseline (Latest Run)

- Best method: `SelfTraining_RF`
- Grouped classes: `16` (`top_k=15` + `OTHER`)
- Pseudo-label acceptance: about `10%` of unlabeled image rows (about `9.6%` of unlabeled cases)

## 8. Limitations

1. Diagnosis space remains highly imbalanced and noisy.
2. Hard labels are derived from soft dermatologist consensus.
3. Unlabeled rows are not random missingness (selection bias from poor images).
4. Metrics are sensitive to class grouping (`top_k`) and confidence thresholds.

## 9. Recommended Next Steps

1. Tune `top_k` and confidence thresholds per presentation objective (accuracy vs class granularity).
2. Compare case-level vs image-level evaluation explicitly.
3. Add probability calibration for pseudo-label confidence.
4. Add confusion-matrix reporting for top grouped classes.
