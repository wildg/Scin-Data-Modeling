# BUSINESS_RESULTS

## Scope and Evidence

This analysis is based on the **main branch** snapshot at commit **`05688e3`** and these evaluation artifacts:

- `results/metrics_baseline_logreg.json`
- `results/metrics_xgboost_model.json`
- `results/metrics_lightgbm_model.json`
- `results/metrics_ffnn_mlp.json`

All four evaluations report performance on the same **test split (613 cases)** over **30 target classes**.

## Executive Business Summary

The project has reached a point where it can deliver **practical triage support** for common dermatology conditions, but it is **not yet suitable for autonomous diagnosis**.

- Best overall label-ranking performance comes from **LightGBM** (`F1 micro = 0.3058`) and **XGBoost** (`0.3052`).
- **Logistic Regression** gives the highest recall (`0.4976`), which is useful when missing a potential condition is costlier than raising extra false alarms.
- **FFNN** has the lowest hamming loss (`0.0909`) but lower recall (`0.3037`), indicating it predicts fewer labels and can miss relevant conditions.
- Macro F1 remains modest across models (`0.1350` to `0.1588`), which shows weaker performance on rarer conditions.

Business interpretation: this pipeline is best positioned as a **decision-support layer** that helps prioritize review queues, not replace clinician judgment.

## Business Value Delivered

1. Faster case prioritization
- The models provide ranked multi-label suggestions that can reduce manual review time for common conditions (for example Eczema and Allergic Contact Dermatitis).
- In operations terms, this can improve throughput for triage teams and research labeling workflows.

2. Better consistency in first-pass tagging
- Even with imperfect rare-class performance, model outputs create a consistent baseline that reduces reviewer-to-reviewer variance.

3. Scalable experimentation platform
- The pipeline (download -> preprocess -> embeddings -> train/tune -> evaluate) enables repeatable model iteration and objective comparison.
- This has direct business value because improvements can be validated before deployment decisions.

4. Practical risk control through model choice
- Different models support different risk postures:
- Recall-first mode (LogReg) for safety-oriented screening.
- Balanced mode (LightGBM/XGBoost) for precision-recall trade-off.

## Model Error Analysis

### 1) Error type by model family

- **Logistic Regression**
- Precision is low (`0.1884`) while recall is high (`0.4976`).
- This pattern implies many **false positives** but fewer missed labels.
- Good for broad screening; costly for reviewer workload if used without thresholds.

- **XGBoost**
- Best balance for micro metrics (`F1 micro 0.3052`, precision `0.2470`, recall `0.3992`).
- Still has **10 classes with zero recall**, so tail conditions are frequently missed.

- **LightGBM**
- Slightly best `F1 micro` (`0.3058`) and similar precision/recall to XGBoost.
- Has **9 classes with zero recall**; better than XGBoost by a small margin, but still weak on long-tail disease coverage.

- **FFNN**
- Lowest hamming loss (`0.0909`) but reduced recall and lower macro outcomes than LogReg.
- This indicates under-calling labels in many cases: fewer false alarms but more misses.

### 2) Class imbalance and long-tail failure

Frequent labels (for example Eczema and Allergic Contact Dermatitis) have moderate F1, while several low-support classes receive near-zero precision/recall in tree and neural models. This is the core reason macro metrics remain low despite acceptable micro metrics.

### 3) Multi-label boundary errors

Because each case can have multiple true conditions, the system must detect co-occurrence. Current behavior shows:

- Over-prediction tendency in LogReg (high recall, low precision).
- Under-prediction tendency in FFNN (lower recall, lower micro F1).
- Balanced but still tail-fragile behavior in boosted trees.

## What Is Usable Right Now

1. Usable immediately
- End-to-end data and modeling pipeline for repeatable experimentation.
- Top-30 class prediction for triage assistance.
- LightGBM/XGBoost as default candidate models for balanced performance.
- LogReg variant where recall-first screening is required.

2. Usable with guardrails
- Dashboard-level or workflow-level case suggestions to assist humans.
- Queue prioritization and weak-label generation for downstream review.

3. Not yet suitable
- Fully automated diagnosis or unsupervised clinical decisioning.
- Rare-condition decision support without human override.

## Assumptions Made in This Analysis

1. Metric files in `results/` represent the current intended benchmark outputs for `main` commit `05688e3`.
2. All models are compared on the same test split and label space (613 cases, 30 classes), as reported in each JSON.
3. Business cost of false negatives is generally higher in screening contexts, but false positives still create operational review cost.
4. Clinical use remains human-in-the-loop; this document does not assume regulatory clearance for autonomous use.
5. Hamming loss is treated as a secondary metric for business decisions; micro/macro precision-recall-F1 drive the primary interpretation.

## Lessons Learned

1. Model architecture alone is not the main bottleneck; **label imbalance and tail coverage** dominate performance limits.
2. Best micro F1 does not guarantee broad clinical usefulness; macro and per-class recall reveal significant blind spots.
3. Different deployment goals require different model choices:
- Screening safety -> favor recall (LogReg-like behavior).
- Reviewer efficiency -> favor balanced precision/recall (LightGBM/XGBoost).
4. Top-K class strategy is useful for near-term value, but long-term business impact requires improving rare-class handling.
5. The strongest asset today is the reproducible pipeline and evaluation discipline, which enables controlled improvement cycles.

## Closing

The project currently provides **real operational value for assisted triage on common conditions**. To move from useful prototype to high-confidence clinical support, the next phase should focus on rare-class robustness, calibrated thresholds by use case, and explicit human-in-the-loop governance.
