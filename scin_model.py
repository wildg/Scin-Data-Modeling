"""Scin Dataset Classifier - Main Training Script"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json
from google.cloud import storage
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ast


class ScinProject:
    """Complete skin condition classification project"""
    
    def __init__(self, workspace="dataset_cache", save_path="saved_models"):
        self.workspace = Path(workspace)
        self.save_path = Path(save_path)
        self.workspace.mkdir(exist_ok=True)
        self.save_path.mkdir(exist_ok=True)
        (self.workspace / "photos").mkdir(exist_ok=True)
        
        self.bucket_name = "dx-scin-public-data"
        self.img_size = 224
        self.label_map = {}
        self.reverse_map = {}
        
    def download_metadata_files(self):
        """Get CSV files from cloud"""
        print("Downloading metadata...")
        client = storage.Client.create_anonymous_client()
        bkt = client.bucket(self.bucket_name)
        
        files_to_get = ["scin_cases.csv", "scin_labels.csv"]
        dfs = []
        
        for fname in files_to_get:
            local = self.workspace / fname
            if not local.exists():
                bkt.blob(fname).download_to_filename(str(local))
            dfs.append(pd.read_csv(local))
        
        return dfs[0], dfs[1]
    
    def prepare_dataset_table(self, cases_df, labels_df, threshold=0.3):
        """Merge and clean data"""
        print("Preparing dataset...")
        
        # Join tables
        df = cases_df.merge(labels_df, on="case_id")
        
        # Keep gradable entries
        grad_filter = df.filter(regex="gradable_for_skin_condition").any(axis=1)
        df = df[grad_filter]
        
        # Parse condition weights
        def get_top_condition(weight_str):
            if pd.isna(weight_str):
                return None, 0.0
            try:
                weights = ast.literal_eval(weight_str)
                if not weights:
                    return None, 0.0
                top = max(weights.items(), key=lambda x: x[1])
                return top[0], top[1]
            except:
                return None, 0.0
        
        df[["condition", "confidence"]] = df["weighted_skin_condition_label"].apply(
            lambda x: pd.Series(get_top_condition(x))
        )
        
        # Filter valid entries
        df = df[(df["condition"].notna()) & (df["confidence"] >= threshold)]
        
        # Get image path
        for i in range(1, 4):
            col = f"image_{i}_path"
            if col in df.columns:
                df["img_path"] = df[col].fillna(method='bfill', axis=1) if i == 1 else df["img_path"].fillna(df[col])
        
        df = df[df["img_path"].notna()]
        
        print(f"Dataset size: {len(df)}")
        print(f"Conditions: {df['condition'].nunique()}")
        
        return df[["case_id", "condition", "confidence", "img_path"]].reset_index(drop=True)
    
    def build_label_encoder(self, conditions):
        """Create label mappings"""
        unique = sorted(conditions.unique())
        self.label_map = {c: i for i, c in enumerate(unique)}
        self.reverse_map = {i: c for c, i in self.label_map.items()}
        
        with open(self.workspace / "labels.json", 'w') as f:
            json.dump({"forward": self.label_map, "backward": self.reverse_map}, f)
        
        return len(unique)
    
    def fetch_photo(self, cloud_path):
        """Download image"""
        fname = Path(cloud_path).name
        local = self.workspace / "photos" / fname
        
        if local.exists():
            return str(local)
        
        try:
            client = storage.Client.create_anonymous_client()
            bkt = client.bucket(self.bucket_name)
            bkt.blob(cloud_path).download_to_filename(str(local))
            return str(local)
        except:
            return None
    
    def make_tf_dataset(self, df, batch_size, training=False):
        """Create TensorFlow dataset"""
        
        # Download images
        print(f"Downloading {len(df)} images...")
        paths = []
        for p in tqdm(df["img_path"], disable=False):
            paths.append(self.fetch_photo(p))
        
        df = df.copy()
        df["local"] = paths
        df = df[df["local"].notna()]
        
        # Prepare data
        image_paths = df["local"].values
        labels = df["condition"].map(self.label_map).values
        scores = df["confidence"].values
        
        def load_img(path, label, score):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.img_size, self.img_size])
            img = tf.cast(img, tf.float32) / 255.0
            
            if training:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.random_brightness(img, 0.2)
                img = tf.clip_by_value(img, 0, 1)
            
            return img, {"class": label, "conf": score}
        
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, scores))
        if training:
            ds = ds.shuffle(1000)
        ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return ds
    
    def create_model(self, num_classes):
        """Build neural network"""
        print("Building model...")
        
        inp = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        
        # Use EfficientNetV2 as backbone
        backbone = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_tensor=inp,
            pooling="avg"
        )
        
        # Fine-tune last layers
        for layer in backbone.layers[:-50]:
            layer.trainable = False
        
        x = backbone.output
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Outputs
        class_out = tf.keras.layers.Dense(num_classes, activation="softmax", name="class")(x)
        conf_out = tf.keras.layers.Dense(1, activation="sigmoid", name="conf")(x)
        
        model = tf.keras.Model(inp, [class_out, conf_out])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss={"class": "sparse_categorical_crossentropy", "conf": "mse"},
            loss_weights={"class": 1.0, "conf": 0.5},
            metrics={"class": ["accuracy"], "conf": ["mae"]}
        )
        
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Main training pipeline"""
        
        # Get data
        cases, labels = self.download_metadata_files()
        df = self.prepare_dataset_table(cases, labels)
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["condition"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["condition"], random_state=42)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Save splits
        train_df.to_csv(self.workspace / "train.csv", index=False)
        val_df.to_csv(self.workspace / "val.csv", index=False)
        test_df.to_csv(self.workspace / "test.csv", index=False)
        
        # Build encoder
        num_classes = self.build_label_encoder(df["condition"])
        
        # Create datasets
        train_ds = self.make_tf_dataset(train_df, batch_size, training=True)
        val_ds = self.make_tf_dataset(val_df, batch_size, training=False)
        
        # Build model
        model = self.create_model(num_classes)
        
        # Callbacks
        cbs = [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.save_path / "best.keras"),
                monitor="val_loss",
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5
            )
        ]
        
        # Train
        print("\nTraining...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=cbs
        )
        
        # Save
        model.save(self.save_path / "final.keras")
        
        with open(self.save_path / "history.json", 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
        
        print(f"\nSaved to {self.save_path}")
        
        return model
    
    def evaluate(self, model_path="saved_models/best.keras"):
        """Evaluate model"""
        print("\nEvaluating...")
        
        # Load model and data
        model = tf.keras.models.load_model(model_path)
        test_df = pd.read_csv(self.workspace / "test.csv")
        
        with open(self.workspace / "labels.json") as f:
            maps = json.load(f)
            self.label_map = maps["forward"]
            self.reverse_map = {int(k): v for k, v in maps["backward"].items()}
        
        # Create test dataset
        test_ds = self.make_tf_dataset(test_df, 32, training=False)
        
        # Predict
        preds = model.predict(test_ds)
        class_probs, conf_preds = preds
        
        # Get metrics
        pred_classes = np.argmax(class_probs, axis=1)
        true_classes = test_df["condition"].map(self.label_map).values
        
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        acc = accuracy_score(true_classes, pred_classes)
        prec, rec, f1, _ = precision_recall_fscore_support(true_classes, pred_classes, average="macro", zero_division=0)
        
        print(f"\nAccuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1: {f1:.3f}")
        
        results = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        with open(self.save_path / "results.json", 'w') as f:
            json.dump(results, f)
        
        return results
    
    def predict(self, image_path, model_path="saved_models/best.keras"):
        """Predict single image"""
        model = tf.keras.models.load_model(model_path)
        
        with open(self.workspace / "labels.json") as f:
            maps = json.load(f)
            self.reverse_map = {int(k): v for k, v in maps["backward"].items()}
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, 0)
        
        # Predict
        class_probs, conf = model.predict(img_batch, verbose=0)
        
        top_idx = np.argmax(class_probs[0])
        
        print(f"\nPrediction: {self.reverse_map[top_idx]}")
        print(f"Confidence: {conf[0][0]:.3f}")
        print(f"Probability: {class_probs[0][top_idx]:.3f}")
        
        return self.reverse_map[top_idx], float(conf[0][0])


if __name__ == "__main__":
    import sys
    
    project = ScinProject()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            project.train(epochs=50, batch_size=32)
            project.evaluate()
        
        elif command == "eval":
            project.evaluate()
        
        elif command == "predict" and len(sys.argv) > 2:
            project.predict(sys.argv[2])
        
        else:
            print("Usage: python scin_model.py [train|eval|predict <image_path>]")
    else:
        # Default: train and evaluate
        project.train(epochs=50, batch_size=32)
        project.evaluate()
