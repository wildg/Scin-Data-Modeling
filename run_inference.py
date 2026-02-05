#!/usr/bin/env python3
"""
Model evaluation and inference script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_retrieval import ScinMetadataProcessor, ConditionEncoder
from modules.image_processing import ImagePreprocessor, TFDatasetBuilder
from modules.network_builder import DualHeadNetwork
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse
import json


class ModelInference:
    """Handles model evaluation and single image prediction"""
    
    def __init__(self, model_file, work_dir="data"):
        self.model = tf.keras.models.load_model(model_file)
        self.work_dir = work_dir
        self.encoder = ConditionEncoder()
        self.encoder.load(os.path.join(work_dir, "encoder.json"))
        self.img_proc = ImagePreprocessor(target_size=(224, 224))
    
    def assess_test_performance(self, test_ds, test_df):
        """Evaluate model on test dataset"""
        print("\n" + "="*70)
        print("Model Performance Assessment")
        print("="*70)
        
        # Generate predictions
        outputs = self.model.predict(test_ds, verbose=1)
        condition_probs = outputs[0]
        score_preds = outputs[1]
        
        # Get predicted classes
        pred_classes = np.argmax(condition_probs, axis=1)
        
        # Get true labels
        true_classes = test_df["lead_condition"].map(self.encoder.encode).values
        true_scores = test_df["lead_score"].values
        
        # Calculate classification metrics
        acc = accuracy_score(true_classes, pred_classes)
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_classes, pred_classes, average="macro", zero_division=0
        )
        
        # Score prediction metrics
        score_mae = np.mean(np.abs(score_preds.flatten() - true_scores))
        score_rmse = np.sqrt(np.mean((score_preds.flatten() - true_scores)**2))
        
        print(f"\n=== Classification Performance ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (macro): {prec:.4f}")
        print(f"Recall (macro): {rec:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        
        print(f"\n=== Confidence Score Prediction ===")
        print(f"MAE: {score_mae:.4f}")
        print(f"RMSE: {score_rmse:.4f}")
        
        # Compute per-class metrics
        print(f"\n=== Per-Class Metrics ===")
        prec_per, rec_per, f1_per, support = precision_recall_fscore_support(
            true_classes, pred_classes, zero_division=0
        )
        
        for idx in range(min(10, len(prec_per))):
            condition_name = self.encoder.decode(idx)
            print(f"{condition_name[:30]:30s} | P: {prec_per[idx]:.3f} | R: {rec_per[idx]:.3f} | F1: {f1_per[idx]:.3f} | N: {support[idx]}")
        
        # Save results
        results_dict = {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "confidence_mae": float(score_mae),
            "confidence_rmse": float(score_rmse)
        }
        
        results_path = os.path.join(self.work_dir, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        return results_dict
    
    def predict_image(self, img_path):
        """Make prediction for single image"""
        img_arr = self.img_proc.load_and_transform(img_path)
        if img_arr is None:
            return None
        
        # Add batch dimension
        img_batch = np.expand_dims(img_arr, axis=0)
        
        # Predict
        outputs = self.model.predict(img_batch, verbose=0)
        cond_probs = outputs[0][0]
        score_val = outputs[1][0][0]
        
        # Get top 5 predictions
        top_idxs = np.argsort(cond_probs)[-5:][::-1]
        predictions_list = []
        
        for idx in top_idxs:
            predictions_list.append({
                "condition": self.encoder.decode(idx),
                "probability": float(cond_probs[idx])
            })
        
        return {
            "primary_diagnosis": predictions_list[0]["condition"],
            "confidence_score": float(score_val),
            "top_5": predictions_list
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate or run inference")
    parser.add_argument("--model-file", required=True, help="Path to model")
    parser.add_argument("--work-dir", default="data", help="Working directory")
    parser.add_argument("--action", choices=["test", "predict"], default="test")
    parser.add_argument("--image", help="Image path for prediction")
    
    args = parser.parse_args()
    
    # Load model
    inference = ModelInference(args.model_file, args.work_dir)
    
    if args.action == "test":
        # Load test split
        test_csv = os.path.join(args.work_dir, "test_split.csv")
        if not os.path.exists(test_csv):
            print(f"Error: {test_csv} not found. Run training first.")
            return
        
        test_df = pd.read_csv(test_csv)
        
        # Build test dataset
        meta_proc = ScinMetadataProcessor(work_dir=args.work_dir)
        ds_builder = TFDatasetBuilder(inference.img_proc, inference.encoder)
        test_ds = ds_builder.create_tf_dataset(test_df, batch_sz=32, shuffle_buf=0, is_train=False)
        
        # Evaluate
        inference.assess_test_performance(test_ds, test_df)
        
    elif args.action == "predict":
        if not args.image:
            print("Error: --image required for predict action")
            return
        
        print(f"\nAnalyzing: {args.image}")
        result = inference.predict_image(args.image)
        
        if result:
            print(f"\n=== Prediction ===")
            print(f"Diagnosis: {result['primary_diagnosis']}")
            print(f"Confidence: {result['confidence_score']:.3f}")
            print(f"\nTop 5 Differential:")
            for i, pred in enumerate(result['top_5'], 1):
                print(f"  {i}. {pred['condition']}: {pred['probability']:.3f}")
        else:
            print("Failed to process image")


if __name__ == "__main__":
    main()
