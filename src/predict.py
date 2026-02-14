import os
import yaml
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from dicttoxml import dicttoxml
# Removed: from .model import PesticideDetectionModel (not needed here, model.py is only needed for training)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PesticidePredictor:
    """
    Loads the trained model and provides methods for making predictions
    and generating reports.
    """
    def __init__(self, config_path='config.yaml'):
        """
        Initializes the predictor by loading configuration and the model.
        """
        self.config_path = config_path
        
        # 1. Load Configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load configuration file {config_path}: {e}")

        # Get paths and model parameters
        self.models_dir = self.config['paths']['models_dir']
        # Ensure we look for the saved model
        self.model_filename = self.config['model'].get('model_filename', 'final_model.h5')
        self.image_size = tuple(self.config['data']['image_size'])
        self.confidence_threshold = self.config['data']['confidence_threshold']
        
        # Reverse the class mapping to get class names from indices (0, 1, ...)
        raw_classes = self.config['data']['class_mapping'].values()
        unique_classes = sorted(list(set(raw_classes)))
        # Assuming 'clean' is 0 and 'contaminated' is 1 due to alphabetical sort
        self.class_names = {i: name for i, name in enumerate(unique_classes)}
        
        # 2. Load Model
        self.model = self._load_model()
        if self.model is None:
            # Propagate error if model loading failed
            raise RuntimeError(f"Failed to load model from: {os.path.join(self.models_dir, self.model_filename)}")

    def _load_model(self):
        """Loads the Keras model from the specified path."""
        model_path = os.path.join(self.models_dir, self.model_filename)
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
        
        try:
            # NOTE: If using EfficientNet or ResNet, Keras often handles the loading without 
            # requiring the class definition, which is why we removed the custom import.
            model = load_model(model_path) 
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _preprocess_image(self, image_path):
        """Loads and preprocesses an image for prediction."""
        try:
            # Load the image, ensuring it is converted to RGB mode
            img = load_img(image_path, target_size=self.image_size, color_mode='rgb')
            img_array = img_to_array(img)
            
            # Sanity check: Ensure the image has 3 channels (R, G, B)
            if img_array.shape[-1] != 3:
                 raise ValueError(f"Image must have 3 channels (RGB). Found {img_array.shape[-1]} channels.")
            
            # Normalize pixel values
            img_array = img_array / 255.0  
            
            # Add batch dimension (1, H, W, C)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            # Re-raise the error with a more informative message
            raise IOError(f"Failed to preprocess image at {image_path}: {type(e).__name__}: {e}")

    def process_image(self, image_path):
        """
        Runs prediction on a single image and generates structured results.
        """
        # 1. Preprocess
        input_data = self._preprocess_image(image_path)
        
        # 2. Predict
        # The model output is a probability vector, e.g., [0.9, 0.1]
        prediction_vector = self.model.predict(input_data)[0]
        
        # 3. Interpret Results (Softmax output assumption)
        
        # Get probabilities for each class
        probabilities = {self.class_names[i]: float(prediction_vector[i]) for i in range(len(prediction_vector))}
        
        # Find the highest confidence class
        predicted_index = np.argmax(prediction_vector)
        predicted_class = self.class_names[predicted_index]
        confidence = prediction_vector[predicted_index]
        
        # Determine binary result (is_contaminated status)
        # We assume 'contaminated' is class 1 and 'clean' is class 0 (due to sorted keys in __init__)
        is_contaminated = (predicted_class == 'contaminated')
        
        # Check if confidence meets the defined threshold
        meets_threshold = (confidence >= self.confidence_threshold)
        
        # Prepare the final result dictionary
        result = {
            "timestamp": datetime.now().isoformat(),
            "image_path": os.path.basename(image_path),
            "predicted_class": predicted_class,
            "is_contaminated": is_contaminated,
            "confidence": confidence,
            "meets_confidence_threshold": meets_threshold,
            "probabilities": probabilities,
            "threshold_used": self.confidence_threshold
        }
        
        return result

    # --- Report Generation Functions (Used by app.py) ---
    # (These functions remain the same as they operate on the dictionary result)

    def generate_txt_report(self, result):
        """Generates a text report string."""
        report = "--- Pesticide Residue Detection Report ---\n"
        report += f"Timestamp: {result['timestamp']}\n"
        report += f"Image File: {result['image_path']}\n"
        report += f"Predicted Class: {result['predicted_class'].upper()}\n"
        report += f"Confidence: {result['confidence']:.2%}\n"
        report += f"Contaminated Status: {'YES' if result['is_contaminated'] else 'NO'}\n"
        report += f"Confidence Threshold: {result['threshold_used']:.2%}\n"
        report += "\n--- Probability Breakdown ---\n"
        for cls, prob in result['probabilities'].items():
            report += f"{cls.capitalize()}: {prob:.2%}\n"
        return report

    def generate_json_report(self, result):
        """Generates a JSON report string."""
        # Convert NumPy float to native Python float for JSON serialization
        json_friendly_result = {
            k: v if not isinstance(v, np.float32) else float(v) 
            for k, v in result.items()
        }
        return json.dumps(json_friendly_result, indent=4)

    def generate_xml_report(self, result):
        """Generates an XML report string."""
        # Use dicttoxml library for robust conversion
        xml_data = dicttoxml(result, custom_root='PesticideDetectionReport', attr_type=False)
        return xml_data.decode('utf-8')


if __name__ == "__main__":
    print("This file contains the prediction logic and should be used by app.py.")
