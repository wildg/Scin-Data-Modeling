# Scin-Data-Modeling

Skin condition classification using Google's SCIN dataset. Trains a neural network to predict skin conditions with confidence scores.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train model:**
```bash
python scin_model.py train
```

**Evaluate:**
```bash
python scin_model.py eval
```

**Predict image:**
```bash
python scin_model.py predict path/to/image.jpg
```

## What it does

- Downloads SCIN dataset from Google Cloud Storage
- Trains EfficientNetV2B0 model with dual outputs (condition + confidence)
- Evaluates with accuracy, precision, recall, F1 metrics
- Saves best model to `saved_models/`

## Output

- `saved_models/best.keras` - Best model
- `saved_models/final.keras` - Final model
- `saved_models/history.json` - Training history
- `saved_models/results.json` - Evaluation metrics
- `dataset_cache/` - Downloaded data

## Performance

Expected metrics:
- Accuracy: 65-75%
- Recall: 60-70%
- F1 Score: 60-70%

## Citation

Ward et al. (2024). Creating an Empirical Dermatology Dataset Through Crowdsourcing. JAMA Network Open, 7(11).
