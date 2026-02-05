# Scin Data Modeling

Machine learning project for skin condition classification using the SCIN (Skin Condition Image Network) dataset from Google Research. This implementation trains a deep neural network to predict skin conditions and confidence scores matching dermatologist assessments.

## Project Goal

Train a model that:
- Classifies skin conditions from dermatology images
- Provides confidence scores similar to dermatologist ratings
- Achieves high accuracy and recall on diverse skin conditions
- Uses transfer learning for efficient training

## Features

- **Dual-Head Architecture**: Simultaneously predicts condition class and confidence score
- **Transfer Learning**: Leverages pre-trained ResNet50 or EfficientNetB0 backbones
- **Custom Augmentation**: Medical image-specific data augmentation (flips, rotations, color adjustments)
- **Stratified Splitting**: Ensures balanced train/validation/test splits
- **Comprehensive Metrics**: Accuracy, precision, recall, F1 scores, and confidence prediction error
- **Cloud Integration**: Downloads data directly from Google Cloud Storage

## Installation

```bash
git clone https://github.com/wildg/Scin-Data-Modeling.git
cd Scin-Data-Modeling
pip install -r requirements.txt
```

## Quick Start

Train the model:
```bash
python run_training.py
```

Evaluate on test set:
```bash
python run_inference.py --model-file checkpoints/top_model.keras --action test
```

Predict single image:
```bash
python run_inference.py --model-file checkpoints/top_model.keras --action predict --image path/to/image.jpg
```

## Training Options

```bash
python run_training.py \
  --work-dir data \
  --checkpoint-dir checkpoints \
  --n-epochs 50 \
  --batch-sz 32 \
  --min-confidence 0.3 \
  --backbone ResNet50 \
  --lr 0.0001
```

**Parameters:**
- `--work-dir`: Directory for downloaded data (default: data)
- `--checkpoint-dir`: Directory for model checkpoints (default: checkpoints)
- `--n-epochs`: Training epochs (default: 50)
- `--batch-sz`: Batch size (default: 32)
- `--min-confidence`: Minimum dermatologist confidence threshold (default: 0.3)
- `--backbone`: CNN backbone - ResNet50 or EfficientNetB0 (default: ResNet50)
- `--lr`: Learning rate (default: 0.0001)

## Architecture

### Network Structure
1. **Input**: 224x224x3 RGB images
2. **Backbone**: Pre-trained CNN (ResNet50/EfficientNetB0) with fine-tuning
3. **Shared Layers**: Dense layers (384→192 units) with dropout
4. **Condition Head**: Softmax classification output
5. **Score Head**: Sigmoid confidence prediction output (0-1 scale)

### Training Strategy
- Multi-task learning with weighted losses
- Early stopping and learning rate reduction
- Model checkpointing for best validation performance
- TensorBoard logging for monitoring

## Dataset

The [SCIN dataset](https://github.com/google-research-datasets/scin) contains:
- 5,000+ cases with 10,000+ images
- Dermatologist labels and confidence ratings
- Multiple dermatology condition categories
- Self-reported demographic and symptom data

Data is automatically downloaded from `dx-scin-public-data` GCS bucket.

## Project Structure

```
Scin-Data-Modeling/
├── modules/
│   ├── data_retrieval.py       # CSV parsing and GCS downloads
│   ├── image_processing.py     # Image preprocessing and TF datasets
│   └── network_builder.py      # Neural network architecture
├── run_training.py              # Training execution script
├── run_inference.py             # Evaluation and prediction script
├── requirements.txt             # Python dependencies
├── data/                        # Downloaded dataset (auto-created)
├── checkpoints/                 # Model saves (auto-created)
└── README.md
```

## Output Files

Training produces:
- `checkpoints/top_model.keras` - Best model (lowest validation loss)
- `checkpoints/final_model.keras` - Final epoch model
- `checkpoints/history.json` - Training metrics history
- `data/encoder.json` - Condition name encoder
- `data/train_split.csv` - Training set metadata
- `data/val_split.csv` - Validation set metadata
- `data/test_split.csv` - Test set metadata
- `data/test_results.json` - Test evaluation metrics

## Performance

Expected metrics with default configuration:
- **Accuracy**: 65-75%
- **Recall**: 60-70% (macro average)
- **Precision**: 60-70% (macro average)
- **F1 Score**: 60-70% (macro average)
- **Confidence MAE**: <0.15

*Performance varies based on condition frequency distribution and training settings.*

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- NumPy, Pandas, Scikit-learn
- Pillow for image processing
- Google Cloud Storage client
- GPU recommended for training

## Citation

If using the SCIN dataset, cite:

```bibtex
@article{ward2024scin,
  author = {Ward, Abbi and Li, Jimmy and Wang, Julie and others},
  title = {Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements},
  journal = {JAMA Network Open},
  volume = {7},
  number = {11},
  year = {2024},
  doi = {10.1001/jamanetworkopen.2024.46615}
}
```

## License

Dataset usage follows the SCIN Data Use License. See https://github.com/google-research-datasets/scin for details.
