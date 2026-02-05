#!/usr/bin/env python3
"""
Main execution script for training the skin condition classifier
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_retrieval import ScinMetadataProcessor, ConditionEncoder
from modules.image_processing import ImagePreprocessor, DatasetSplitter, TFDatasetBuilder
from modules.network_builder import DualHeadNetwork, TrainingController
import argparse


def execute_training():
    parser = argparse.ArgumentParser(description="Train skin condition model")
    parser.add_argument("--work-dir", default="data", help="Working directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Model save directory")
    parser.add_argument("--n-epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-sz", type=int, default=32, help="Batch size")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Min confidence threshold")
    parser.add_argument("--backbone", default="ResNet50", choices=["ResNet50", "EfficientNetB0"])
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    
    args = parser.parse_args()
    
    print("="*70)
    print("SCIN Dermatology Model Training Pipeline")
    print("="*70)
    
    # Step 1: Metadata processing
    print("\n[Step 1/7] Processing metadata...")
    meta_proc = ScinMetadataProcessor(work_dir=args.work_dir)
    cases_df, labels_df = meta_proc.get_csvs()
    
    # Step 2: Merge and filter
    print("\n[Step 2/7] Merging and filtering data...")
    combined_df = meta_proc.merge_and_filter(cases_df, labels_df, min_score=args.min_confidence)
    print(f"Top 10 conditions:\n{combined_df['lead_condition'].value_counts().head(10)}")
    
    # Step 3: Encode conditions
    print("\n[Step 3/7] Encoding conditions...")
    encoder = ConditionEncoder()
    encoder.fit(combined_df["lead_condition"])
    encoder.save(os.path.join(args.work_dir, "encoder.json"))
    print(f"Total classes: {len(encoder.str_to_int)}")
    
    # Step 4: Split dataset
    print("\n[Step 4/7] Splitting dataset...")
    splitter = DatasetSplitter(test_ratio=0.15, val_ratio=0.15, seed=42)
    train_df, val_df, test_df = splitter.split(combined_df, "lead_condition")
    
    # Save split info
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(os.path.join(args.work_dir, f"{name}_split.csv"), index=False)
    
    # Step 5: Download images and build datasets
    print("\n[Step 5/7] Building TensorFlow datasets...")
    img_proc = ImagePreprocessor(target_size=(224, 224))
    ds_builder = TFDatasetBuilder(img_proc, encoder)
    
    print("Downloading training images...")
    train_df = ds_builder.download_images_for_split(train_df, meta_proc)
    print("Downloading validation images...")
    val_df = ds_builder.download_images_for_split(val_df, meta_proc)
    print("Downloading test images...")
    test_df = ds_builder.download_images_for_split(test_df, meta_proc)
    
    # Create TF datasets
    train_ds = ds_builder.create_tf_dataset(train_df, args.batch_sz, shuffle_buf=1000, is_train=True)
    val_ds = ds_builder.create_tf_dataset(val_df, args.batch_sz, shuffle_buf=0, is_train=False)
    
    # Step 6: Build and compile network
    print(f"\n[Step 6/7] Building {args.backbone} network...")
    net = DualHeadNetwork(
        n_classes=len(encoder.str_to_int),
        img_shape=(224, 224, 3),
        backbone_type=args.backbone
    )
    model = net.build_network()
    model = net.compile_network(lr=args.lr)
    
    print("\nNetwork Architecture:")
    net.get_summary()
    
    # Step 7: Train
    print("\n[Step 7/7] Starting training...")
    print("="*70)
    
    controller = TrainingController(model, save_dir=args.checkpoint_dir)
    history = controller.run_training(
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=args.n_epochs,
        batch_sz=args.batch_sz
    )
    
    # Save final model
    print("\nSaving final model...")
    net.persist(os.path.join(args.checkpoint_dir, "final_model.keras"))
    controller.export_history()
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print(f"Best model: {args.checkpoint_dir}/top_model.keras")
    print(f"Final model: {args.checkpoint_dir}/final_model.keras")
    print(f"History: {args.checkpoint_dir}/history.json")


if __name__ == "__main__":
    execute_training()
