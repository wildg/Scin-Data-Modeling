"""
Main training script for skin condition classification model
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scin_dataset_handler import DermatologyDatasetManager
from data_pipeline import ImageDataPipeline
from model_architecture import ConfidenceAwareSkinClassifier, TrainingOrchestrator


def main():
    parser = argparse.ArgumentParser(description='Train skin condition classification model')
    parser.add_argument('--data-dir', default='data', help='Directory for dataset storage')
    parser.add_argument('--model-dir', default='models', help='Directory for model checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, help='Minimum confidence for samples')
    parser.add_argument('--architecture', default='ResNet50', choices=['ResNet50', 'EfficientNetB0'], help='Base architecture')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Initial learning rate')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SCIN Dataset Skin Condition Classification Training")
    print("="*60)
    
    # Initialize dataset manager
    print("\n[1/6] Initializing dataset manager...")
    dataset_mgr = DermatologyDatasetManager(
        workspace_dir=args.data_dir,
        target_dims=(224, 224)
    )
    
    # Download and load metadata
    print("\n[2/6] Fetching metadata from cloud storage...")
    case_data, label_data = dataset_mgr.fetch_csv_metadata()
    
    # Build unified dataset
    print("\n[3/6] Building training dataset...")
    unified_dataset = dataset_mgr.build_training_dataset(
        case_data,
        label_data,
        confidence_floor=args.confidence_threshold
    )
    
    # Create category mappings
    print("\n[4/6] Creating category mappings...")
    cat_to_idx, idx_to_cat = dataset_mgr.build_category_mappings(unified_dataset)
    dataset_mgr.persist_category_mappings(cat_to_idx, idx_to_cat)
    print(f"Number of categories: {len(cat_to_idx)}")
    
    # Split and prepare data pipeline
    print("\n[5/6] Preparing data pipeline...")
    pipeline = ImageDataPipeline(dataset_mgr, augment=True)
    train_df, val_df, test_df = pipeline.split_dataset(unified_dataset)
    
    # Download images for each split
    print("\nDownloading training images...")
    train_df = pipeline.download_images_batch(train_df)
    print("Downloading validation images...")
    val_df = pipeline.download_images_batch(val_df)
    print("Downloading test images...")
    test_df = pipeline.download_images_batch(test_df)
    
    # Save split information
    pipeline.save_split_info(train_df, val_df, test_df, args.data_dir)
    
    # Build TensorFlow datasets
    print("\nBuilding TensorFlow datasets...")
    train_dataset = pipeline.build_tf_dataset(
        train_df, cat_to_idx, batch_size=args.batch_size, shuffle=True, is_training=True
    )
    val_dataset = pipeline.build_tf_dataset(
        val_df, cat_to_idx, batch_size=args.batch_size, shuffle=False, is_training=False
    )
    
    # Construct model
    print(f"\n[6/6] Building model with {args.architecture} architecture...")
    classifier = ConfidenceAwareSkinClassifier(
        num_categories=len(cat_to_idx),
        input_shape=(224, 224, 3),
        base_arch=args.architecture
    )
    model = classifier.construct_model()
    model = classifier.compile_model(learning_rate=args.learning_rate)
    
    print("\nModel Summary:")
    classifier.summary()
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    orchestrator = TrainingOrchestrator(model, checkpoint_dir=args.model_dir)
    history = orchestrator.execute_training(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save final model and history
    print("\nSaving final model...")
    classifier.save_architecture(os.path.join(args.model_dir, 'final_model.keras'))
    orchestrator.save_training_history()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {args.model_dir}/best_model.keras")
    print(f"Final model saved to: {args.model_dir}/final_model.keras")
    print(f"Training history saved to: {args.model_dir}/training_history.json")


if __name__ == '__main__':
    main()
