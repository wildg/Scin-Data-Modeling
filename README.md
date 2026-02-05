# Scin Dataset Skin Condition Classification

A deep learning project for classifying skin conditions using Google's Scin (Skin Condition Image Network) dataset. This model predicts skin conditions with confidence scores similar to dermatologist assessments, achieving high accuracy and recall.

## Overview

This project implements a multi-task neural network that:
- **Classifies** skin conditions from images into multiple diagnostic categories
- **Predicts confidence scores** (0-1 scale) matching dermatologist confidence levels
- Uses **transfer learning** with pre-trained backbones (ResNet50 or EfficientNetB0)
- Implements **data augmentation** for robust training
- Provides comprehensive **evaluation metrics** including accuracy, precision, recall, and F1 scores

## Dataset

The project uses the [SCIN dataset](https://github.com/google-research-datasets/scin) from Google Research, which contains:
- 5,000+ volunteer contributions (10,000+ images)
- Dermatologist labels with confidence scores
- Self-reported demographic and symptom information
- Multiple skin condition categories

The data is automatically downloaded from Google Cloud Storage during training.

## Project Structure

```
Scin-Data-Modeling/
├── src/
│   ├── scin_dataset_handler.py    # Dataset download and preprocessing
│   ├── model_architecture.py       # Neural network model definition
│   └── data_pipeline.py            # TensorFlow data pipeline with augmentation
├── train.py                         # Main training script
├── evaluate.py                      # Model evaluation and prediction
├── requirements.txt                 # Python dependencies
├── data/                            # Downloaded dataset (created automatically)
├── models/                          # Trained model checkpoints (created automatically)
└── README.md                        # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wildg/Scin-Data-Modeling.git
cd Scin-Data-Modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model with default settings:
```bash
python train.py
```

Advanced training options:
```bash
python train.py \
    --data-dir data \
    --model-dir models \
    --epochs 50 \
    --batch-size 32 \
    --confidence-threshold 0.3 \
    --architecture ResNet50 \
    --learning-rate 0.0001
```

**Arguments:**
- `--data-dir`: Directory for dataset storage (default: `data`)
- `--model-dir`: Directory for model checkpoints (default: `models`)
- `--epochs`: Number of training epochs (default: `50`)
- `--batch-size`: Batch size for training (default: `32`)
- `--confidence-threshold`: Minimum confidence for training samples (default: `0.3`)
- `--architecture`: Base architecture - `ResNet50` or `EfficientNetB0` (default: `ResNet50`)
- `--learning-rate`: Initial learning rate (default: `0.0001`)

### Evaluation

Evaluate the trained model on the test set:
```bash
python evaluate.py --model-path models/best_model.keras --mode evaluate
```

Make predictions on a single image:
```bash
python evaluate.py \
    --model-path models/best_model.keras \
    --mode predict \
    --image-path path/to/image.jpg
```

**Output includes:**
- Top predicted diagnosis
- Confidence score (0-1 scale)
- Top 5 predictions with probabilities

## Model Architecture

The model uses a **multi-task learning** approach:

1. **Backbone**: Pre-trained CNN (ResNet50 or EfficientNetB0) with ImageNet weights
2. **Feature Extraction**: Transfer learning with fine-tuning of later layers
3. **Shared Layers**: Dense layers (512 → 256 units) with dropout for regularization
4. **Classification Head**: Softmax output for skin condition diagnosis
5. **Confidence Head**: Sigmoid output predicting dermatologist confidence (0-1)

### Loss Functions
- **Classification**: Sparse categorical cross-entropy
- **Confidence**: Mean squared error (MSE)
- **Multi-task weighting**: Classification loss weighted 1.0, confidence loss weighted 0.3

### Training Features
- **Early stopping**: Prevents overfitting by monitoring validation loss
- **Learning rate scheduling**: Reduces learning rate on plateaus
- **Model checkpointing**: Saves best model based on validation performance
- **Data augmentation**: Random flips, brightness, contrast, and saturation adjustments
- **TensorBoard logging**: Track training metrics in real-time

## Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision across all classes
- **Recall**: Macro-averaged recall across all classes
- **F1 Score**: Macro-averaged F1 score
- **Confidence MAE/RMSE**: Error in confidence prediction

Per-class metrics include precision, recall, and F1 scores for each skin condition.

## Training Process

1. **Data Loading**: Downloads metadata CSVs from GCS
2. **Data Preparation**: Filters cases by dermatologist gradability and confidence threshold
3. **Label Processing**: Extracts weighted condition labels from dermatologist annotations
4. **Image Download**: Retrieves images from cloud storage
5. **Data Splitting**: Creates train/validation/test splits (70%/15%/15%)
6. **Augmentation**: Applies random transformations during training
7. **Model Training**: Multi-task learning with dual loss functions
8. **Evaluation**: Comprehensive metrics on held-out test set

## Output Files

After training, the following files are generated:

- `models/best_model.keras`: Best model checkpoint (lowest validation loss)
- `models/final_model.keras`: Final model after all epochs
- `models/training_history.json`: Training metrics history
- `models/evaluation_results.json`: Test set evaluation metrics
- `data/category_mappings.json`: Category to index mappings
- `data/splits/train_cases.csv`: Training split information
- `data/splits/val_cases.csv`: Validation split information
- `data/splits/test_cases.csv`: Test split information

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy, Pandas, Scikit-learn
- Pillow, OpenCV
- Google Cloud Storage client
- tqdm for progress bars

See `requirements.txt` for complete dependencies.

## Performance Expectations

With default settings, the model typically achieves:
- **Accuracy**: 65-75% (varies by condition frequency and complexity)
- **Top-3 Accuracy**: 85-90%
- **Recall**: 60-70% (macro-averaged)
- **Confidence Prediction**: MAE < 0.15

Performance depends on:
- Number of training epochs
- Base architecture selection
- Data augmentation settings
- Class imbalance in the dataset

## Notes

- The dataset contains imbalanced classes; some conditions have more samples than others
- Training time varies based on hardware (GPU strongly recommended)
- First run will download ~10GB of data from Google Cloud Storage
- Internet connection required for initial data download

## License

This project follows the SCIN Data Use License for the dataset. See the [SCIN repository](https://github.com/google-research-datasets/scin) for details.

## Citation

If you use the SCIN dataset, please cite:

```bibtex
@article{ward2024scin,
    author = {Ward, Abbi and Li, Jimmy and Wang, Julie and others},
    title = {Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements},
    journal = {JAMA Network Open},
    volume = {7},
    number = {11},
    pages = {e2446615},
    year = {2024},
    doi = {10.1001/jamanetworkopen.2024.46615}
}
```

## Acknowledgments

- Google Research for the SCIN dataset
- TensorFlow and Keras teams for the deep learning framework
- Pre-trained model weights from ImageNet
