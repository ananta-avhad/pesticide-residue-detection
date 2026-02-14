import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
import yaml

class PesticideDetectionModel:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.image_size = tuple(self.config['data']['image_size'])
        self.num_classes = self.config['model']['num_classes']
        self.architecture = self.config['model']['architecture']
        
    def build_simple_cnn(self):
        """Simple CNN for binary classification"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.image_size, 3)),
            
            # Convolutional blocks
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet(self):
        """ResNet50 transfer learning model"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_efficientnet(self):
        """EfficientNet transfer learning model"""
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_model(self):
        """Build model based on configuration"""
        if self.architecture == 'cnn':
            model = self.build_simple_cnn()
        elif self.architecture == 'resnet':
            model = self.build_resnet()
        elif self.architecture == 'efficientnet':
            model = self.build_efficientnet()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['model']['learning_rate']
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

if __name__ == "__main__":
    # Test model building
    model_builder = PesticideDetectionModel()
    model = model_builder.build_model()
    model.summary()