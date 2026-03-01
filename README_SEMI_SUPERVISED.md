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

Current evaluation/training flow:
1. Collapse target to top-K classes + `OTHER` (default `top_k=15`).
2. Filter labeled rows by confidence (`target_max_prob >= 0.55` by default).
3. Create grouped validation split from `train_labeled` by `case_id` (default `validation_size=0.2`).
4. Select best method on validation metrics only (`f1_macro`, then `accuracy`).
5. Run grouped cross-validation on labeled training cases (`cv_folds=5` default, stratified when feasible).
6. Retrain all methods on full labeled training data and report test metrics once.
7. Use the validation-selected method to pseudo-label unlabeled training rows.
8. Aggregate pseudo-label confidence at case level, then apply acceptance gates (`pseudo_threshold`, `pseudo_margin`).

## 5. How to Run

### A) Generate processed data

```bash
uv run python -m scin_data_modeling.data.preprocess --output-dir data/processed
```

### B) Run semi-supervised benchmark and pseudo-labeling

```bash
uv run python scripts/run_semi_supervised_models.py \
  --processed-dir data/processed \
  --output-dir artifacts/semi_supervised \
  --validation-size 0.2 \
  --cv-folds 5
```

### C) Optional tuning example

```bash
uv run python scripts/run_semi_supervised_models.py \
  --processed-dir data/processed \
  --output-dir artifacts/semi_supervised_custom \
  --top-k-classes 20 \
  --min-label-confidence 0.55 \
  --validation-size 0.2 \
  --cv-folds 5 \
  --self-training-threshold 0.60 \
  --pseudo-threshold 0.60 \
  --pseudo-margin 0.10
```

## 6. Outputs and How to Read Them

Main outputs:
- `artifacts/semi_supervised/semi_supervised_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_validation_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_cv_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_cv_fold_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_summary.json`
- `artifacts/semi_supervised/unlabeled_pseudo_labels_all.csv`
- `artifacts/semi_supervised/unlabeled_pseudo_labels_accepted.csv`
- `artifacts/semi_supervised/best_semi_supervised_model.joblib`

Interpretation:
- `semi_supervised_validation_metrics.csv`: model-selection table (validation split only).
- `semi_supervised_cv_metrics.csv`: grouped CV mean/std metrics by method.
- `semi_supervised_cv_fold_metrics.csv`: per-fold grouped CV metrics.
- `semi_supervised_metrics.csv`: final held-out test metrics (not used for selection).
- `semi_supervised_summary.json`: best method, split sizes, validation/CV/test summary metrics, pseudo-label acceptance rates.
- `unlabeled_pseudo_labels_accepted.csv`: pseudo-labels considered reliable enough to include.
- `best_semi_supervised_model.joblib`: serialized best method bundle containing:
  - trained model object
  - selected method name
  - label classes
  - feature columns
  - key run thresholds/parameters

## 7. Latest Full Run Results (March 1, 2026)

Run command:

```bash
uv run python -m scin_data_modeling.models.semi_supervised
```

Selection setup and data sizes:
- Best method was selected on **validation macro-F1** (with accuracy as tie-breaker).
- `top_k_classes=15`, `min_label_confidence=0.55`, `validation_size=0.2`, `cv_folds=5`.
- Grouped classes: `16` (`top_k=15` + `OTHER`).
- Labeled train rows: `2504` total (`1991` fit split + `513` validation split).
- Unlabeled train rows: `3889` (`1972` cases).
- Test labeled rows: `614`.

Model comparison (latest run):

| Method | Validation Accuracy | Validation F1-macro | CV Accuracy Mean +- Std | CV F1-macro Mean +- Std | Test Accuracy | Test F1-macro |
|---|---:|---:|---:|---:|---:|---:|
| LabelSpreading | 0.2710 | 0.0980 | 0.2476 +- 0.0387 | 0.0917 +- 0.0103 | 0.2443 | 0.0813 |
| SelfTraining_RF | 0.3411 | 0.0866 | 0.3300 +- 0.0089 | 0.0951 +- 0.0198 | 0.3160 | 0.0883 |
| Supervised_Only_RF | 0.3041 | 0.0291 | 0.2146 +- 0.1501 | 0.0208 +- 0.0136 | 0.0928 | 0.0106 |
| ClusterPseudoLabel_RF | 0.3041 | 0.0291 | 0.3225 +- 0.0016 | 0.0305 +- 0.0001 | 0.3583 | 0.0330 |
| CoTraining_Consensus_RF | 0.2008 | 0.0209 | 0.1934 +- 0.1315 | 0.0308 +- 0.0220 | 0.0928 | 0.0106 |

Pseudo-labeling outcomes (using selected method `LabelSpreading`):
- Accepted pseudo-labeled rows: `500 / 3889` (`12.86%`).
- Accepted pseudo-labeled cases: `211 / 1972` (`10.70%`).
- Acceptance gates: `pseudo_threshold=0.6`, `pseudo_margin=0.1`.

Interpretation:
1. `LabelSpreading` won model selection because it had the best **validation macro-F1**, which prioritizes balanced class performance over majority-class accuracy.
2. `SelfTraining_RF` achieved stronger accuracy (validation/CV/test), and also had the best CV and test macro-F1, indicating it is a strong alternative if you prioritize overall predictive strength.
3. `ClusterPseudoLabel_RF` had the highest test accuracy but low macro-F1, suggesting it is likely over-favoring frequent classes.
4. The gap between validation-selected method (`LabelSpreading`) and top test macro-F1 (`SelfTraining_RF`) suggests some selection noise at current split sizes; monitoring across multiple seeds is recommended.
5. Pseudo-label acceptance remains conservative (~13% rows), which helps precision but limits unlabeled-data expansion.

Assumptions:
1. `case_id` is a valid grouping key and no case should cross train/validation/test boundaries.
2. Grouping to top-K + `OTHER` preserves enough diagnostic signal for the intended use case.
3. `target_max_prob` from dermatologist consensus is a meaningful confidence proxy for filtering labels.
4. Unlabeled rows are informative enough to help decision boundaries in semi-supervised methods.
5. Macro-F1 is the primary model-selection objective because class balance is important.
6. Case-level pseudo-label acceptance (`threshold` + `margin`) is a reasonable proxy for pseudo-label reliability.

Understanding Model Errors:
1. Majority-class bias is still present in some methods; high accuracy with low macro-F1 (for example `ClusterPseudoLabel_RF`) indicates poor minority-class recall.
2. Label noise remains because the hard training label is derived from a soft weighted target; borderline and mixed-label cases are harder to learn.
3. Selection instability exists: validation chooses `LabelSpreading`, while CV/test macro-F1 favors `SelfTraining_RF`; this indicates split sensitivity.
4. Unlabeled-data bias is likely: unlabeled cases are often low-quality images, so pseudo-label distribution may not match labeled distribution.
5. `LabelSpreading` runtime warnings (`invalid value encountered in divide`) suggest graph neighborhoods can be sparse/degenerate for some points.
6. Current reporting is aggregate-only; without per-class confusion matrices, error modes by diagnosis family are partially hidden.

Usable vs Experimental Parts:
1. Usable now:
   - Leakage-safe preprocessing and case-level splits.
   - Validation-based model selection and grouped cross-validation outputs.
   - End-to-end artifact generation (`metrics`, `summary`, pseudo-label CSVs, `joblib` model bundle).
2. Use with caution:
   - Direct method choice between `LabelSpreading` and `SelfTraining_RF` for external reporting, because rankings vary by split/metric emphasis.
   - Pseudo-label acceptance thresholds (`0.6`/`0.1`) without calibration studies.
3. Not yet production-ready:
   - Automatic clinical decision support usage without per-class calibration, external validation, and fairness/bias evaluation.
   - Reliance on single-run metrics without seed sweeps and confidence intervals.

Artifacts generated in this run:
- `artifacts/semi_supervised/semi_supervised_validation_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_cv_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_cv_fold_metrics.csv`
- `artifacts/semi_supervised/semi_supervised_metrics.csv`
- `artifacts/semi_supervised/unlabeled_pseudo_labels_all.csv`
- `artifacts/semi_supervised/unlabeled_pseudo_labels_accepted.csv`
- `artifacts/semi_supervised/semi_supervised_summary.json`
- `artifacts/semi_supervised/best_semi_supervised_model.joblib`

## 8. Executed Next-Step Experiments (March 1, 2026)

Automation script:
- `scripts/run_semi_supervised_next_steps.py`

Executed scope:
1. Multi-seed parameter tuning (`top_k`, confidence filters, pseudo-label thresholds/margins).
2. Validation/CV setting sweep (`validation_size`, `cv_folds`).
3. Confidence-interval aggregation across seeds.
4. Case-level vs image-level evaluation on selected winning experiment.
5. Confusion matrices and per-class error-slice export.
6. Probability calibration check (temperature scaling) with ECE comparison.

Primary outcomes:
1. Best grid by mean test macro-F1: `k15_c055_t060_m010` (3 seeds).
2. `k15_c055_t060_m010` test F1-macro mean: `0.0886` (95% CI: `0.0783` to `0.0989`).
3. `k15_c055_t060_m010` test accuracy mean: `0.2948` (95% CI: `0.2451` to `0.3445`).
4. `k15_c055_t060_m010` mean pseudo acceptance rate: `0.1127`.
5. Best-method stability favored `SelfTraining_RF` across tuning grids: each of the three grids selected `SelfTraining_RF` in 2/3 seeds and `LabelSpreading` in 1/3 seeds.
6. Case-level aggregation improved performance on selected analysis run (`k15_c055_t060_m010_seed52`): image-level accuracy/F1 = `0.3241`/`0.0988`; case-level accuracy/F1 = `0.3423`/`0.1091`.
7. Calibration on selected run worsened ECE (`0.0729` uncalibrated -> `0.2953` calibrated) and sharply increased acceptance (`0.0570` -> `0.5423`), so uncalibrated gating remains preferred.

Artifacts generated:
- `artifacts/semi_supervised_next_steps/REPORT_INDEX.json`
- `artifacts/semi_supervised_next_steps/experiments_summary.csv`
- `artifacts/semi_supervised_next_steps/config_ci_summary.csv`
- `artifacts/semi_supervised_next_steps/best_method_frequency.csv`
- `artifacts/semi_supervised_next_steps/analysis/case_vs_image_metrics.json`
- `artifacts/semi_supervised_next_steps/analysis/confusion_matrix_image_level.csv`
- `artifacts/semi_supervised_next_steps/analysis/confusion_matrix_case_level.csv`
- `artifacts/semi_supervised_next_steps/analysis/error_slices_per_class.csv`
- `artifacts/semi_supervised_next_steps/analysis/test_image_predictions.csv`
- `artifacts/semi_supervised_next_steps/analysis/test_case_predictions.csv`

## 9. Limitations

1. Diagnosis space remains highly imbalanced and noisy.
2. Hard labels are derived from soft dermatologist consensus.
3. Unlabeled rows are not random missingness (selection bias from poor images).
4. Metrics remain sensitive to class grouping (`top_k`) and confidence thresholds.
5. `LabelSpreading` can emit runtime warnings on sparse/weak graph neighborhoods.

## 10. Recommended Next Steps

1. Increase seed count (for example 10+) on the current best grid to tighten confidence intervals.
2. Promote case-level metrics to primary reporting, since case aggregation performed better than image-level.
3. Inspect `error_slices_per_class.csv` and confusion matrices to define targeted class regrouping or data curation.
4. Keep `k15_c055_t060_m010` as the default baseline and compare any future variants against its CI band.
5. Investigate why temperature scaling degraded ECE before retrying calibration methods.
