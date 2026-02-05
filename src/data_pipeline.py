"""
Data pipeline for creating TensorFlow datasets with custom augmentation
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm


class ImageDataPipeline:
    """Builds efficient TensorFlow data pipelines with augmentation"""
    
    def __init__(self, dataset_manager, augment=True):
        self.manager = dataset_manager
        self.augment = augment
        
    def split_dataset(self, dataset_df, test_fraction=0.15, val_fraction=0.15, random_seed=42):
        """Partition dataset into train/validation/test splits"""
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            dataset_df,
            test_size=test_fraction,
            random_state=random_seed,
            stratify=dataset_df['primary_diagnosis']
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_fraction / (1 - test_fraction)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=train_val_df['primary_diagnosis']
        )
        
        print(f"\n=== Data Splits ===")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def download_images_batch(self, df):
        """Download all images for a dataframe split"""
        local_paths = []
        print(f"Downloading {len(df)} images...")
        
        for gcs_path in tqdm(df['selected_image'].values):
            local_path = self.manager.retrieve_image_from_cloud(gcs_path)
            local_paths.append(local_path)
        
        df = df.copy()
        df['local_image_path'] = local_paths
        df = df[df['local_image_path'].notna()]
        
        return df
    
    def build_tf_dataset(self, df, category_to_idx, batch_size=32, shuffle=True, is_training=False):
        """Construct TensorFlow Dataset object"""
        
        # Prepare data arrays
        image_paths = df['local_image_path'].values
        diagnoses = df['primary_diagnosis'].map(category_to_idx).values
        confidences = df['diagnosis_confidence'].values
        
        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((
            image_paths,
            diagnoses,
            confidences
        ))
        
        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=42)
        
        # Map to load and preprocess images
        dataset = dataset.map(
            lambda path, diag, conf: self._process_single_sample(path, diag, conf, is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _process_single_sample(self, img_path, diagnosis_label, confidence_score, apply_augmentation):
        """Load and preprocess a single training sample"""
        
        # Load image file
        img_raw = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, self.manager.img_dimensions)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Apply augmentation if training
        if apply_augmentation and self.augment:
            img = self._augment_image(img)
        
        # Prepare outputs
        outputs = {
            'diagnosis_output': diagnosis_label,
            'confidence_output': confidence_score
        }
        
        return img, outputs
    
    def _augment_image(self, img):
        """Apply random augmentation transformations"""
        
        # Random horizontal flip
        img = tf.image.random_flip_left_right(img)
        
        # Random brightness adjustment
        img = tf.image.random_brightness(img, max_delta=0.15)
        
        # Random contrast adjustment
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        
        # Random saturation adjustment
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        
        # Clip to valid range
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img
    
    def save_split_info(self, train_df, val_df, test_df, output_dir):
        """Save split information for reproducibility"""
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            filepath = os.path.join(output_dir, 'splits', f'{name}_cases.csv')
            df[['case_id', 'primary_diagnosis', 'diagnosis_confidence', 'selected_image', 'local_image_path']].to_csv(
                filepath, index=False
            )
        print(f"Split information saved to {output_dir}/splits/")
