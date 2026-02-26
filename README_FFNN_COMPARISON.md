# Feedforward Neural Network Model Comparison

## Summary
This document adds a new sklearn feedforward neural network model (`MLPClassifier`) to compare against the existing logistic regression and XGBoost baselines for multi-label skin condition prediction on SCIN embeddings.

## New Model
- **Model name:** `ffnn`
- **Implementation:** `scin_data_modeling/models/ffnn_model.py`
- **Classifier:** `sklearn.neural_network.MLPClassifier`
- **Architecture:** hidden layers `(768, 256)`
- **Training setup:** `adam`, `learning_rate_init=5e-4`, `batch_size=64`, `max_iter=300`, `early_stopping=True`, `n_iter_no_change=20`, `random_state=42`
- **Artifact output:** `models/ffnn_mlp.joblib`

## How To Run
```bash
# Train the new feedforward neural network
uv run scin_data_modeling train --mode frozen --model ffnn

# Evaluate the new model
uv run scin_data_modeling evaluate --model ffnn
```

## Metrics Comparison
Test split size: **613** cases.

| Metric | Logistic Regression | XGBoost | FFNN (sklearn MLP) |
|---|---:|---:|---:|
| Hamming Loss | 0.0083 | 0.0070 | 0.0076 |
| F1 (micro) | 0.1857 | 0.0959 | 0.2056 |
| F1 (macro) | 0.0163 | 0.0051 | 0.0098 |
| F1 (weighted) | 0.1583 | 0.0777 | 0.1506 |
| Precision (micro) | 0.2973 | 0.5180 | 0.3802 |
| Recall (micro) | 0.1350 | 0.0528 | 0.1409 |
| Precision (macro) | 0.0290 | 0.0237 | 0.0228 |
| Recall (macro) | 0.0128 | 0.0036 | 0.0081 |

## Interpretation
- The new FFNN improves **micro F1** over logistic regression (**0.2056 vs 0.1857**, about +10.7%).
- The FFNN also improves **micro precision** and **micro recall** over logistic regression.
- Compared with XGBoost, FFNN has much better **micro F1** and **micro recall**, while XGBoost remains the most conservative/high-precision model.
- Macro metrics remain low across all models, indicating the rare-class challenge is still the main bottleneck.

## Notes
- Logistic regression and FFNN metrics were recomputed from local artifacts in the current sklearn 1.8 environment.
- XGBoost values are taken from the existing project baseline report in `README.md` to keep comparison aligned with prior project results.

## Files Changed For This Addition
- `scin_data_modeling/models/ffnn_model.py`
- `scin_data_modeling/cli.py`
- `README_FFNN_COMPARISON.md`

_Last updated: February 26, 2026._
