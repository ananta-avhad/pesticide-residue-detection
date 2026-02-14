import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from model import PesticideDetectionModel

class Trainer:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_path = self.config['paths']['processed_data']
        self.models_dir = self.config['paths']['models_dir']
        self.batch_size = self.config['data']['batch_size']
        self.epochs = self.config['model']['epochs']
        self.image_size = tuple(self.config['data']['image_size'])
        
        os.makedirs(self.models_dir, exist_ok=True)
    
    def create_data_generators(self):
        """Create data generators for training and validation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.processed_path, 'train'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            color_mode='rgb',
            class_mode='sparse',  # For sparse_categorical_crossentropy
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(self.processed_path, 'validation'),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='sparse',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def get_callbacks(self):
        """Setup training callbacks"""
        callbacks = [
            # Save best model
            ModelCheckpoint(
                os.path.join(self.models_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'training_history.png'))
        plt.close()
        
        print(f"Training history plot saved to {self.models_dir}/training_history.png")
    
    def train(self):
        """Main training function"""
        print("Creating data generators...")
        train_gen, val_gen = self.create_data_generators()
        
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Classes: {train_gen.class_indices}")
        
        print("\nBuilding model...")
        model_builder = PesticideDetectionModel()
        model = model_builder.build_model()
        
        print("\nModel summary:")
        model.summary()
        
        print("\nStarting training...")
        history = model.fit(
            train_gen,
            epochs=self.epochs,
            validation_data=val_gen,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(self.models_dir, 'final_model.h5'))
        print(f"\nFinal model saved to {self.models_dir}/final_model.h5")
        
        # Plot training history
        self.plot_training_history(history)
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_loss, val_accuracy = model.evaluate(val_gen)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return model, history

if __name__ == "__main__":
    trainer = Trainer()
    model, history = trainer.train()