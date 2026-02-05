"""
Custom neural network architecture for dermatology classification with confidence scoring
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class ConfidenceAwareSkinClassifier:
    """Neural network model with confidence prediction for skin condition diagnosis"""
    
    def __init__(self, num_categories, input_shape=(224, 224, 3), base_arch='ResNet50'):
        self.num_categories = num_categories
        self.input_shape = input_shape
        self.base_arch = base_arch
        self.model = None
        
    def construct_model(self):
        """Build complete model architecture with transfer learning backbone"""
        
        # Input layer
        input_tensor = layers.Input(shape=self.input_shape, name='image_input')
        
        # Select and configure pre-trained backbone
        if self.base_arch == 'ResNet50':
            backbone = keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=input_tensor,
                pooling='avg'
            )
        elif self.base_arch == 'EfficientNetB0':
            backbone = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=input_tensor,
                pooling='avg'
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.base_arch}")
        
        # Freeze early layers, train later layers
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        
        # Feature extraction
        features = backbone.output
        
        # Shared dense layers
        x = layers.Dense(512, activation='relu', name='dense_shared_1')(features)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu', name='dense_shared_2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Classification head
        classification_output = layers.Dense(
            self.num_categories,
            activation='softmax',
            name='diagnosis_output'
        )(x)
        
        # Confidence prediction head (0-1 scale matching dermatologist confidence)
        confidence_branch = layers.Dense(128, activation='relu', name='confidence_dense_1')(x)
        confidence_branch = layers.Dropout(0.2)(confidence_branch)
        confidence_output = layers.Dense(
            1,
            activation='sigmoid',
            name='confidence_output'
        )(confidence_branch)
        
        # Build multi-output model
        self.model = Model(
            inputs=input_tensor,
            outputs=[classification_output, confidence_output],
            name='SkinConditionClassifier'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Configure model with custom loss functions and metrics"""
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Multiple losses for multi-task learning
        loss_config = {
            'diagnosis_output': 'sparse_categorical_crossentropy',
            'confidence_output': 'mse'
        }
        
        # Weight the losses
        loss_weights = {
            'diagnosis_output': 1.0,
            'confidence_output': 0.3
        }
        
        # Metrics for each output
        metrics_config = {
            'diagnosis_output': [
                'accuracy',
                keras.metrics.SparseCategoricalAccuracy(name='cat_accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ],
            'confidence_output': [
                'mae',
                'mse'
            ]
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_config,
            loss_weights=loss_weights,
            metrics=metrics_config
        )
        
        return self.model
    
    def summary(self):
        """Display model architecture summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not constructed yet. Call construct_model() first.")
    
    def save_architecture(self, filepath):
        """Persist model to disk"""
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_architecture(self, filepath):
        """Restore model from disk"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return self.model


class TrainingOrchestrator:
    """Manages model training with custom callbacks and monitoring"""
    
    def __init__(self, model, checkpoint_dir='models'):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.history = None
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def prepare_callbacks(self, patience=10):
        """Configure training callbacks for optimization"""
        
        callbacks = [
            # Save best model based on validation loss
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.checkpoint_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def execute_training(self, train_data, val_data, epochs=50, batch_size=32):
        """Run training loop with validation"""
        
        callbacks = self.prepare_callbacks()
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_training_history(self):
        """Export training metrics to JSON"""
        if self.history:
            import json
            history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
            history_dict = {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            print(f"Training history saved: {history_path}")
