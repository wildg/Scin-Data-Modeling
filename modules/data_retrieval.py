"""
Scin Image Retrieval and Metadata Processor
Custom implementation for downloading and organizing dermatology images
"""

import pandas as pd
import numpy as np
import os
from google.cloud import storage
from pathlib import Path
import json
import ast


class ScinMetadataProcessor:
    """Handles metadata CSV parsing and case organization"""
    
    def __init__(self, bucket="dx-scin-public-data", work_dir="data"):
        self.gcs_bucket = bucket
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "img_cache").mkdir(exist_ok=True)
        
    def get_csvs(self):
        """Pull CSV files from GCS bucket"""
        try:
            anon_client = storage.Client.create_anonymous_client()
            bucket_handle = anon_client.bucket(self.gcs_bucket)
            
            csv_map = {"scin_cases.csv": "cases_data", "scin_labels.csv": "labels_data"}
            result = {}
            
            for csv_file, key in csv_map.items():
                target_path = self.work_dir / csv_file
                if not target_path.exists():
                    print(f"Downloading {csv_file}...")
                    blob = bucket_handle.blob(csv_file)
                    blob.download_to_filename(str(target_path))
                result[key] = pd.read_csv(target_path)
                print(f"Loaded {key}: {len(result[key])} rows")
            
            return result["cases_data"], result["labels_data"]
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def decode_confidence_dict(self, text_val):
        """Convert string representation to dictionary"""
        if pd.isna(text_val):
            return {}
        try:
            return ast.literal_eval(text_val)
        except:
            return {}
    
    def merge_and_filter(self, cases, labels, min_score=0.3):
        """Combine cases with labels and apply filters"""
        # Merge on case_id
        combined = pd.merge(cases, labels, on="case_id", how="inner")
        
        # Filter gradable cases only
        grad_cols = [c for c in combined.columns if "gradable_for_skin_condition" in c]
        combined = combined[combined[grad_cols].any(axis=1)]
        
        # Parse weighted labels
        combined["parsed_weights"] = combined["weighted_skin_condition_label"].apply(
            self.decode_confidence_dict
        )
        
        # Get top condition
        combined["lead_condition"] = combined["parsed_weights"].apply(
            lambda d: max(d, key=d.get) if len(d) > 0 else None
        )
        combined["lead_score"] = combined["parsed_weights"].apply(
            lambda d: max(d.values()) if len(d) > 0 else 0.0
        )
        
        # Apply filters
        combined = combined[
            (combined["lead_condition"].notna()) & 
            (combined["lead_score"] >= min_score)
        ]
        
        # Get primary image path
        img_cols = [f"image_{i}_path" for i in range(1, 4)]
        combined["main_img"] = combined[img_cols].bfill(axis=1).iloc[:, 0]
        combined = combined[combined["main_img"].notna()]
        
        print(f"\nFiltered dataset: {len(combined)} samples")
        print(f"Unique conditions: {combined['lead_condition'].nunique()}")
        return combined
    
    def pull_image(self, cloud_path):
        """Download single image from GCS"""
        local_name = Path(cloud_path).name
        local_file = self.work_dir / "img_cache" / local_name
        
        if local_file.exists():
            return str(local_file)
        
        try:
            anon_client = storage.Client.create_anonymous_client()
            bucket_handle = anon_client.bucket(self.gcs_bucket)
            blob = bucket_handle.blob(cloud_path)
            blob.download_to_filename(str(local_file))
            return str(local_file)
        except Exception as e:
            print(f"Download failed for {cloud_path}: {e}")
            return None


class ConditionEncoder:
    """Maps condition names to numeric indices"""
    
    def __init__(self):
        self.str_to_int = {}
        self.int_to_str = {}
    
    def fit(self, condition_series):
        """Build encoding from condition names"""
        unique_vals = sorted(condition_series.unique())
        self.str_to_int = {name: idx for idx, name in enumerate(unique_vals)}
        self.int_to_str = {idx: name for name, idx in self.str_to_int.items()}
        return self
    
    def encode(self, condition_name):
        """Convert condition name to index"""
        return self.str_to_int.get(condition_name, -1)
    
    def decode(self, index):
        """Convert index back to condition name"""
        return self.int_to_str.get(index, "Unknown")
    
    def save(self, filepath):
        """Persist encoder to JSON"""
        with open(filepath, 'w') as f:
            json.dump({
                "str_to_int": self.str_to_int,
                "int_to_str": self.int_to_str
            }, f, indent=2)
    
    def load(self, filepath):
        """Restore encoder from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.str_to_int = data["str_to_int"]
        self.int_to_str = {int(k): v for k, v in data["int_to_str"].items()}
        return self
