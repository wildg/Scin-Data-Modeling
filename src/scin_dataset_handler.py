"""
Custom handler for Scin dermatology dataset with unique architecture
"""

import os
import ast
import pandas as pd
import numpy as np
from PIL import Image
from google.cloud import storage
from typing import Tuple
import json


class DermatologyDatasetManager:
    """Manages dermatology image dataset operations with custom processing pipeline"""
    
    def __init__(self, storage_bucket="dx-scin-public-data", workspace_dir="data", target_dims=(224, 224)):
        self.gcs_bucket = storage_bucket
        self.workspace = workspace_dir
        self.img_dimensions = target_dims
        self._ensure_workspace_structure()
        
    def _ensure_workspace_structure(self):
        """Build required directory hierarchy"""
        for subdir in ['', 'images', 'splits']:
            os.makedirs(os.path.join(self.workspace, subdir), exist_ok=True)
    
    def fetch_csv_metadata(self):
        """Retrieve CSV files containing case and label metadata"""
        gcs_client = storage.Client.create_anonymous_client()
        bucket_ref = gcs_client.bucket(self.gcs_bucket)
        
        metadata_files = {'scin_cases.csv': 'cases', 'scin_labels.csv': 'labels'}
        dataframes = {}
        
        for csv_name, df_key in metadata_files.items():
            local_csv = os.path.join(self.workspace, csv_name)
            if not os.path.exists(local_csv):
                print(f"Fetching {csv_name} from cloud storage...")
                blob_ref = bucket_ref.blob(csv_name)
                blob_ref.download_to_filename(local_csv)
            dataframes[df_key] = pd.read_csv(local_csv)
            print(f"Loaded {df_key}: {len(dataframes[df_key])} entries")
        
        return dataframes['cases'], dataframes['labels']
    
    def extract_condition_scores(self, label_string):
        """Parse condition confidence dictionary from string representation"""
        if pd.isna(label_string):
            return {}
        try:
            return ast.literal_eval(label_string)
        except:
            return {}
    
    def build_training_dataset(self, case_data, label_data, confidence_floor=0.3):
        """Construct unified dataset from separate case and label tables"""
        
        # Join tables on case identifier
        unified = case_data.merge(label_data, on='case_id', how='inner')
        
        # Keep only dermatologist-gradable samples
        gradable_cols = [c for c in unified.columns if 'dermatologist_gradable_for_skin_condition' in c]
        unified = unified[unified[gradable_cols].any(axis=1)]
        
        # Extract condition rankings with confidence values
        unified['condition_rankings'] = unified['weighted_skin_condition_label'].apply(
            self.extract_condition_scores
        )
        
        # Identify primary diagnosis and associated confidence
        unified['primary_diagnosis'] = unified['condition_rankings'].apply(
            lambda rankings: max(rankings, key=rankings.get) if rankings else None
        )
        unified['diagnosis_confidence'] = unified['condition_rankings'].apply(
            lambda rankings: max(rankings.values()) if rankings else 0.0
        )
        
        # Filter by confidence threshold and valid diagnoses
        unified = unified[unified['primary_diagnosis'].notna() & (unified['diagnosis_confidence'] >= confidence_floor)]
        
        # Select primary image from available options
        image_columns = [f'image_{i}_path' for i in range(1, 4)]
        unified['selected_image'] = unified[image_columns].bfill(axis=1).iloc[:, 0]
        unified = unified[unified['selected_image'].notna()]
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total samples: {len(unified)}")
        print(f"Unique diagnoses: {unified['primary_diagnosis'].nunique()}")
        print(f"\nDiagnosis distribution (top 10):")
        print(unified['primary_diagnosis'].value_counts().head(10))
        
        return unified
    
    def retrieve_image_from_cloud(self, gcs_path):
        """Download individual image file from cloud storage"""
        filename = os.path.basename(gcs_path)
        destination = os.path.join(self.workspace, 'images', filename)
        
        if os.path.exists(destination):
            return destination
        
        try:
            gcs_client = storage.Client.create_anonymous_client()
            bucket_ref = gcs_client.bucket(self.gcs_bucket)
            blob_ref = bucket_ref.blob(gcs_path)
            blob_ref.download_to_filename(destination)
            return destination
        except Exception as err:
            print(f"Failed to retrieve {gcs_path}: {err}")
            return None
    
    def process_image_file(self, filepath):
        """Load and transform image to standard format"""
        try:
            pil_img = Image.open(filepath).convert('RGB')
            resized = pil_img.resize(self.img_dimensions)
            normalized = np.array(resized, dtype=np.float32) / 255.0
            return normalized
        except Exception as err:
            print(f"Image processing error for {filepath}: {err}")
            return None
    
    def build_category_mappings(self, dataset_df):
        """Generate bidirectional category encoding dictionaries"""
        categories = sorted(dataset_df['primary_diagnosis'].unique())
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        idx_to_cat = {i: cat for cat, i in cat_to_idx.items()}
        return cat_to_idx, idx_to_cat
    
    def persist_category_mappings(self, cat_to_idx, idx_to_cat):
        """Save category mappings to JSON for later use"""
        output_file = os.path.join(self.workspace, 'category_mappings.json')
        mappings = {
            'category_to_index': cat_to_idx,
            'index_to_category': idx_to_cat
        }
        with open(output_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"Category mappings saved: {output_file}")
    
    def load_category_mappings(self):
        """Restore category mappings from JSON file"""
        input_file = os.path.join(self.workspace, 'category_mappings.json')
        with open(input_file, 'r') as f:
            mappings = json.load(f)
        cat_to_idx = mappings['category_to_index']
        idx_to_cat = {int(k): v for k, v in mappings['index_to_category'].items()}
        return cat_to_idx, idx_to_cat
