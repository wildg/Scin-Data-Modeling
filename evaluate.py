"""
Evaluation and prediction module for trained models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import json
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scin_dataset_handler import DermatologyDatasetManager
from data_pipeline import ImageDataPipeline


class ModelEvaluator:
    """Evaluates trained model performance on test data"""
    
    def __init__(self, model_path, data_dir='data'):
        self.model = keras.models.load_model(model_path)
        self.data_dir = data_dir
        self.dataset_mgr = DermatologyDatasetManager(workspace_dir=data_dir)
        self.cat_to_idx, self.idx_to_cat = self.dataset_mgr.load_category_mappings()
        
    def evaluate_on_test_set(self, test_dataset, test_df):
        """Run comprehensive evaluation on test set"""
        
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        # Get predictions
        predictions = self.model.predict(test_dataset, verbose=1)
        pred_diagnoses = predictions[0]  # Classification probabilities
        pred_confidences = predictions[1]  # Confidence scores
        
        # Get predicted class indices
        pred_classes = np.argmax(pred_diagnoses, axis=1)
        
        # Get true labels
        true_classes = test_df['primary_diagnosis'].map(self.cat_to_idx).values
        true_confidences = test_df['diagnosis_confidence'].values
        
        # Calculate classification metrics
        accuracy = accuracy_score(true_classes, pred_classes)
        precision_macro = precision_score(true_classes, pred_classes, average='macro', zero_division=0)
        recall_macro = recall_score(true_classes, pred_classes, average='macro', zero_division=0)
        f1_macro = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
        
        print(f"\n=== Classification Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (macro): {precision_macro:.4f}")
        print(f"Recall (macro): {recall_macro:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        
        # Confidence prediction error
        confidence_mae = np.mean(np.abs(pred_confidences.flatten() - true_confidences))
        confidence_rmse = np.sqrt(np.mean((pred_confidences.flatten() - true_confidences)**2))
        
        print(f"\n=== Confidence Prediction ===")
        print(f"Mean Absolute Error: {confidence_mae:.4f}")
        print(f"Root Mean Squared Error: {confidence_rmse:.4f}")
        
        # Per-class metrics
        print(f"\n=== Per-Class Classification Report ===")
        target_names = [self.idx_to_cat[i] for i in range(len(self.idx_to_cat))]
        print(classification_report(true_classes, pred_classes, target_names=target_names, zero_division=0))
        
        # Save detailed results
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'confidence_mae': float(confidence_mae),
                'confidence_rmse': float(confidence_rmse)
            }
        }
        
        results_path = os.path.join(os.path.dirname(self.model.name if hasattr(self.model, 'name') else 'models'), 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")
        
        return results
    
    def predict_single_image(self, image_path):
        """Make prediction on a single image"""
        
        # Load and preprocess image
        img_array = self.dataset_mgr.process_image_file(image_path)
        if img_array is None:
            return None
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_batch, verbose=0)
        diagnosis_probs = predictions[0][0]  # First output, first sample
        confidence_score = predictions[1][0][0]  # Second output, first sample
        
        # Get top predictions
        top_indices = np.argsort(diagnosis_probs)[-5:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            top_predictions.append({
                'diagnosis': self.idx_to_cat[idx],
                'probability': float(diagnosis_probs[idx])
            })
        
        result = {
            'top_prediction': top_predictions[0]['diagnosis'],
            'confidence_score': float(confidence_score),
            'top_5_predictions': top_predictions
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model-path', required=True, help='Path to trained model file')
    parser.add_argument('--data-dir', default='data', help='Directory containing dataset')
    parser.add_argument('--mode', choices=['evaluate', 'predict'], default='evaluate', help='Evaluation mode')
    parser.add_argument('--image-path', help='Path to image for prediction (required if mode=predict)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.data_dir)
    
    if args.mode == 'evaluate':
        # Load test split
        test_split_path = os.path.join(args.data_dir, 'splits', 'test_cases.csv')
        if not os.path.exists(test_split_path):
            print(f"Error: Test split not found at {test_split_path}")
            print("Please run training first to generate splits.")
            return
        
        test_df = pd.read_csv(test_split_path)
        
        # Build test dataset
        pipeline = ImageDataPipeline(evaluator.dataset_mgr, augment=False)
        test_dataset = pipeline.build_tf_dataset(
            test_df, evaluator.cat_to_idx, batch_size=32, shuffle=False, is_training=False
        )
        
        # Run evaluation
        evaluator.evaluate_on_test_set(test_dataset, test_df)
        
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: --image-path required for prediction mode")
            return
        
        print(f"\nPredicting for image: {args.image_path}")
        result = evaluator.predict_single_image(args.image_path)
        
        if result:
            print(f"\n=== Prediction Results ===")
            print(f"Top Diagnosis: {result['top_prediction']}")
            print(f"Confidence Score: {result['confidence_score']:.3f}")
            print(f"\nTop 5 Predictions:")
            for i, pred in enumerate(result['top_5_predictions'], 1):
                print(f"  {i}. {pred['diagnosis']}: {pred['probability']:.3f}")
        else:
            print("Error: Could not process image")


if __name__ == '__main__':
    main()
