"""
Custom image preprocessing and TF Dataset generation
Unique augmentation strategy for dermatology images
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from tqdm import tqdm


class ImagePreprocessor:
    """Handles image loading and normalization"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_and_transform(self, img_path):
        """Load image file and apply standard preprocessing"""
        try:
            pil_image = Image.open(img_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize to target dimensions
            pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to array and scale to [0, 1]
            img_arr = np.asarray(pil_image, dtype=np.float32) / 255.0
            
            return img_arr
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None


class DatasetSplitter:
    """Creates stratified train/val/test splits"""
    
    def __init__(self, test_ratio=0.15, val_ratio=0.15, seed=42):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed
    
    def split(self, dataframe, target_col):
        """Perform stratified splitting"""
        # First split: test set
        splitter1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_ratio,
            random_state=self.seed
        )
        
        for train_val_idx, test_idx in splitter1.split(dataframe, dataframe[target_col]):
            train_val_set = dataframe.iloc[train_val_idx].copy()
            test_set = dataframe.iloc[test_idx].copy()
        
        # Second split: validation from train_val
        val_ratio_adj = self.val_ratio / (1.0 - self.test_ratio)
        splitter2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio_adj,
            random_state=self.seed
        )
        
        for train_idx, val_idx in splitter2.split(train_val_set, train_val_set[target_col]):
            train_set = train_val_set.iloc[train_idx].copy()
            val_set = train_val_set.iloc[val_idx].copy()
        
        print(f"Split sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        return train_set, val_set, test_set


class AugmentationPipeline:
    """Custom augmentation for medical images"""
    
    @staticmethod
    def apply_training_augment(image):
        """Apply random augmentations during training"""
        # Horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        
        # Vertical flip (useful for skin lesions)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
        
        # Random rotation (90 degree increments)
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        
        # Brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # Contrast adjustment
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        
        # Saturation adjustment
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        
        # Ensure values stay in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image


class TFDatasetBuilder:
    """Builds TensorFlow Dataset objects for training"""
    
    def __init__(self, preprocessor, encoder):
        self.preprocessor = preprocessor
        self.encoder = encoder
    
    def download_images_for_split(self, split_df, metadata_proc):
        """Download all images for a specific split"""
        paths = []
        print(f"Downloading {len(split_df)} images...")
        for cloud_path in tqdm(split_df["main_img"].values):
            local_path = metadata_proc.pull_image(cloud_path)
            paths.append(local_path)
        
        split_df = split_df.copy()
        split_df["local_path"] = paths
        split_df = split_df[split_df["local_path"].notna()]
        return split_df
    
    def create_tf_dataset(self, split_df, batch_sz, shuffle_buf, is_train):
        """Build TF Dataset from dataframe"""
        # Extract features
        paths = split_df["local_path"].values
        conditions = split_df["lead_condition"].map(self.encoder.encode).values
        scores = split_df["lead_score"].values
        
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((paths, conditions, scores))
        
        if shuffle_buf > 0:
            ds = ds.shuffle(buffer_size=shuffle_buf, seed=42)
        
        # Map loading function
        ds = ds.map(
            lambda p, c, s: self._load_sample(p, c, s, is_train),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        ds = ds.batch(batch_sz)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def _load_sample(self, path, condition, score, augment):
        """Load and process a single sample"""
        # Read and decode image
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, self.preprocessor.target_size)
        img = tf.cast(img, tf.float32) / 255.0
        
        # Apply augmentation if training
        if augment:
            img = AugmentationPipeline.apply_training_augment(img)
        
        # Return image and multi-output targets
        targets = {
            "condition_pred": condition,
            "score_pred": score
        }
        
        return img, targets
