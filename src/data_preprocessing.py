import os
import numpy as np
import cv2
from PIL import Image
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import spectral
from tqdm import tqdm
import shutil



class HyperspectralDataProcessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_root = self.config['paths']['dataset_root']
        self.processed_path = self.config['paths']['processed_data']
        self.wavelength_range = self.config['data']['wavelength_range']
        self.image_size = tuple(self.config['data']['image_size'])
        
    def load_hyperspectral_image(self, image_path):
        """
        Load hyperspectral image and convert to usable format
        """
        try:
            # Try loading as ENVI format (common for hyperspectral)
            img = spectral.open_image(image_path)
            data = img.load()
            return data
        except:
            # Fallback to regular image loading
            img = cv2.imread(image_path)
            return img
    
    def reduce_spectral_bands(self, hyperspectral_img, method='pca', n_components=3):
        """
        Reduce hyperspectral bands to RGB-like representation
        Methods: 'pca', 'average', 'select'
        """
        if len(hyperspectral_img.shape) == 2:
            # Already 2D, convert to 3 channel
            return cv2.cvtColor(hyperspectral_img, cv2.COLOR_GRAY2RGB)
        
        if hyperspectral_img.shape[2] <= 3:
            return hyperspectral_img
        
        if method == 'pca':
            # Use PCA to reduce dimensions
            from sklearn.decomposition import PCA
            h, w, c = hyperspectral_img.shape
            reshaped = hyperspectral_img.reshape(-1, c)
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(reshaped)
            return reduced.reshape(h, w, n_components)
        
        elif method == 'select':
            # Select specific bands (R, G, B equivalents)
            # For visible spectrum, typically bands 50, 30, 20
            total_bands = hyperspectral_img.shape[2]
            r_band = min(int(total_bands * 0.7), total_bands - 1)
            g_band = min(int(total_bands * 0.5), total_bands - 1)
            b_band = min(int(total_bands * 0.3), total_bands - 1)
            return hyperspectral_img[:, :, [r_band, g_band, b_band]]
        
        elif method == 'average':
            # Average across all bands
            return np.mean(hyperspectral_img, axis=2, keepdims=True).repeat(3, axis=2)
    
    def preprocess_image(self, image_path, label=None):
        """
        Load and preprocess a single image
        """
        # Load image
        img = self.load_hyperspectral_image(image_path)
        
        # Handle hyperspectral data
        if len(img.shape) == 3 and img.shape[2] > 3:
            img = self.reduce_spectral_bands(img, method='select')
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_labels_from_filename(self, filename):
        """
        Create labels based on filename patterns
        You need to adjust this based on your dataset structure
        """
        # Example: if filename contains 'clean' or 'contaminated'
        filename_lower = filename.lower()
        
        if 'clean' in filename_lower or 'normal' in filename_lower:
            return 0  # Clean
        elif 'contaminated' in filename_lower or 'residue' in filename_lower:
            return 1  # Contaminated
        else:
            # Default or unknown
            return -1
    
    def organize_dataset(self):
        """
        Organize raw dataset into train/val/test splits
        """
        wavelength_folder = os.path.join(self.dataset_root, self.wavelength_range)
        
        if not os.path.exists(wavelength_folder):
            print(f"Error: Wavelength folder {wavelength_folder} not found!")
            return
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_files.extend(Path(wavelength_folder).rglob(ext))
        
        print(f"Found {len(image_files)} images")
        
        # Create labels (you need to implement this based on your data)
        # For now, we'll create a simple structure
        data = []
        for img_path in tqdm(image_files, desc="Processing images"):
            label = self.create_labels_from_filename(img_path.name)
            if label != -1:  # Skip unknown labels
                data.append({'path': str(img_path), 'label': label})
        
        if len(data) == 0:
            print("Warning: No labeled data found. Creating dummy structure.")
            print("You need to manually label your data!")
            # Create dummy structure for demonstration
            for i, img_path in enumerate(image_files[:min(100, len(image_files))]):
                # Alternate labels for demo
                data.append({'path': str(img_path), 'label': i % 2})
        
        # Split data
        train_data, temp_data = train_test_split(
            data, 
            test_size=(1 - self.config['data']['train_split']),
            random_state=42
        )
        
        val_size = self.config['data']['val_split'] / (1 - self.config['data']['train_split'])
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=42
        )
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create directory structure
        for split in ['train', 'validation', 'test']:
            for class_name in ['clean', 'contaminated']:
                os.makedirs(
                    os.path.join(self.processed_path, split, class_name),
                    exist_ok=True
                )
        
        # Copy and process files
        self._copy_split_data(train_data, 'train')
        self._copy_split_data(val_data, 'validation')
        self._copy_split_data(test_data, 'test')
        
        print("Dataset organization complete!")
    
    def _copy_split_data(self, data, split_name):
        """Helper to copy data to split folders"""
        for item in tqdm(data, desc=f"Processing {split_name}"):
            src_path = item['path']
            label = item['label']
            class_name = 'clean' if label == 0 else 'contaminated'
            
            dst_dir = os.path.join(self.processed_path, split_name, class_name)
            dst_path = os.path.join(dst_dir, Path(src_path).name)
            
            try:
                # Process and save image
                img = self.preprocess_image(src_path)
                img_uint8 = (img * 255).astype(np.uint8)
                cv2.imwrite(dst_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    processor = HyperspectralDataProcessor()
    processor.organize_dataset()