import os
import yaml
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from dicttoxml import dicttoxml

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PesticidePredictor:

    def __init__(self, config_path='config.yaml'):

        self.config_path = config_path

        # Load config
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load configuration file {config_path}: {e}")

        self.models_dir = self.config['paths']['models_dir']
        self.model_filename = self.config['model'].get('model_filename', 'final_model.h5')
        self.image_size = tuple(self.config['data']['image_size'])
        self.confidence_threshold = float(self.config['data']['confidence_threshold'])

        raw_classes = self.config['data']['class_mapping'].values()
        unique_classes = sorted(list(set(raw_classes)))
        self.class_names = {i: name for i, name in enumerate(unique_classes)}

        self.model = self._load_model()

        if self.model is None:
            raise RuntimeError(
                f"Failed to load model from: {os.path.join(self.models_dir, self.model_filename)}"
            )

    def _load_model(self):
        model_path = os.path.join(self.models_dir, self.model_filename)

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None

        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _preprocess_image(self, image_path):

        try:
            img = load_img(image_path, target_size=self.image_size, color_mode='rgb')
            img_array = img_to_array(img)

            if img_array.shape[-1] != 3:
                raise ValueError("Image must be RGB")

            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            raise IOError(f"Failed to preprocess image: {e}")

    def process_image(self, image_path):

        input_data = self._preprocess_image(image_path)

        prediction_vector = self.model.predict(input_data)[0]

        probabilities = {
            self.class_names[i]: float(prediction_vector[i])
            for i in range(len(prediction_vector))
        }

        predicted_index = int(np.argmax(prediction_vector))
        predicted_class = self.class_names[predicted_index]

        confidence = float(prediction_vector[predicted_index])

        is_contaminated = bool(predicted_class == 'contaminated')

        meets_threshold = bool(confidence >= self.confidence_threshold)

        result = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(os.path.basename(image_path)),
            "predicted_class": str(predicted_class),
            "is_contaminated": bool(is_contaminated),
            "confidence": float(confidence),
            "meets_confidence_threshold": bool(meets_threshold),
            "probabilities": probabilities,
            "threshold_used": float(self.confidence_threshold)
        }

        return result

    # ---------- REPORT GENERATION ----------

    def generate_txt_report(self, result):

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
        return json.dumps(result, indent=4)

    def generate_xml_report(self, result):
        xml_data = dicttoxml(
            result,
            custom_root='PesticideDetectionReport',
            attr_type=False
        )
        return xml_data.decode('utf-8')


if __name__ == "__main__":
    print("Prediction module loaded.")
