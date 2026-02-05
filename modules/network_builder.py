"""
Neural network architecture with dual prediction heads
Custom layer configuration for dermatology classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import os


class DualHeadNetwork:
    """Two-output neural network for condition and confidence prediction"""
    
    def __init__(self, n_classes, img_shape=(224, 224, 3), backbone_type="ResNet50"):
        self.n_classes = n_classes
        self.img_shape = img_shape
        self.backbone_type = backbone_type
        self.network = None
    
    def build_network(self):
        """Construct the complete network architecture"""
        # Input
        input_layer = layers.Input(shape=self.img_shape, name="image_input")
        
        # Backbone selection
        if self.backbone_type == "ResNet50":
            base_model = tf.keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                input_tensor=input_layer,
                pooling="avg"
            )
        elif self.backbone_type == "EfficientNetB0":
            base_model = tf.keras.applications.EfficientNetB0(
                weights="imagenet",
                include_top=False,
                input_tensor=input_layer,
                pooling="avg"
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_type}")
        
        # Freeze initial layers
        for layer in base_model.layers[:-40]:
            layer.trainable = False
        
        # Extract features
        features = base_model.output
        
        # Shared representation
        shared = layers.Dense(384, activation="relu", name="shared_repr")(features)
        shared = layers.Dropout(0.35)(shared)
        shared = layers.Dense(192, activation="relu", name="shared_repr2")(shared)
        shared = layers.Dropout(0.25)(shared)
        
        # Condition classification head
        condition_head = layers.Dense(self.n_classes, activation="softmax", name="condition_pred")(shared)
        
        # Confidence score head
        score_branch = layers.Dense(64, activation="relu", name="score_dense")(shared)
        score_branch = layers.Dropout(0.15)(score_branch)
        score_head = layers.Dense(1, activation="sigmoid", name="score_pred")(score_branch)
        
        # Assemble model
        self.network = models.Model(
            inputs=input_layer,
            outputs=[condition_head, score_head],
            name="DualHeadDermaNet"
        )
        
        return self.network
    
    def compile_network(self, lr=0.0001):
        """Configure optimizer, losses, and metrics"""
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        
        loss_dict = {
            "condition_pred": "sparse_categorical_crossentropy",
            "score_pred": "mean_squared_error"
        }
        
        weight_dict = {
            "condition_pred": 1.0,
            "score_pred": 0.4
        }
        
        metric_dict = {
            "condition_pred": ["accuracy"],
            "score_pred": ["mae"]
        }
        
        self.network.compile(
            optimizer=opt,
            loss=loss_dict,
            loss_weights=weight_dict,
            metrics=metric_dict
        )
        
        return self.network
    
    def get_summary(self):
        """Print architecture details"""
        if self.network:
            return self.network.summary()
        else:
            print("Network not built yet")
    
    def persist(self, save_path):
        """Save model to disk"""
        self.network.save(save_path)
        print(f"Saved to {save_path}")
    
    @staticmethod
    def restore(load_path):
        """Load model from disk"""
        return tf.keras.models.load_model(load_path)


class TrainingController:
    """Manages the training process with callbacks"""
    
    def __init__(self, network, save_dir="checkpoints"):
        self.network = network
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.train_history = None
    
    def setup_callbacks(self, early_stop_patience=12):
        """Configure training callbacks"""
        callback_list = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.save_dir, "top_model.keras"),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.4,
                patience=6,
                min_lr=1e-8,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.save_dir, "tensorboard_logs")
            )
        ]
        return callback_list
    
    def run_training(self, train_ds, val_ds, n_epochs, batch_sz):
        """Execute training loop"""
        cbs = self.setup_callbacks()
        
        self.train_history = self.network.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs,
            callbacks=cbs,
            verbose=1
        )
        
        return self.train_history
    
    def export_history(self):
        """Save training history to JSON"""
        if self.train_history:
            import json
            hist_file = os.path.join(self.save_dir, "history.json")
            hist_dict = {k: [float(v) for v in vals] 
                        for k, vals in self.train_history.history.items()}
            with open(hist_file, 'w') as f:
                json.dump(hist_dict, f, indent=2)
            print(f"History saved to {hist_file}")
