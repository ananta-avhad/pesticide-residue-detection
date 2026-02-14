# 🥗 Pesticide Residue Detection System

A machine learning-based system for detecting pesticide residues on fruits and vegetables using hyperspectral imaging.

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Important Notes](#important-notes)

## 🎯 Overview

This project uses deep learning to analyze hyperspectral images of vegetables and detect pesticide residues that may not be visible to the naked eye.

**Features:**
- Hyperspectral image processing
- CNN-based classification
- Real-time web interface
- Multiple report formats (TXT, XML, JSON)
- Visualization of results

## 🛠️ Installation

### Prerequisites
- Python 3.8-3.10
- 4GB+ RAM
- 2GB+ free disk space

### Step 1: Clone or Create Project

```bash
mkdir pesticide-detection
cd pesticide-detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 Dataset Setup

### Understanding Your Dataset

Your Kaggle dataset has three folders representing different wavelength ranges:
- **0.1-0.8**: Visible + Near-Infrared (400-800nm)
- **0.9-1.3**: Short-Wave Infrared (900-1300nm)
- **1.4-1.7**: Mid-Wave Infrared (1400-1700nm)

### Step 1: Prepare Dataset Structure

1. Place your downloaded dataset in `dataset/raw/`
2. Your structure should look like:
```
dataset/
└── raw/
    ├── 0.1-0.8/
    ├── 0.9-1.3/
    └── 1.4-1.7/
```

### Step 2: Label Your Data

**CRITICAL**: You need to label your images as either "clean" or "contaminated". 

You have two options:

**Option A: Manual Labeling** (Recommended)
1. Create two folders in each wavelength range: `clean/` and `contaminated/`
2. Manually sort images into these folders based on your knowledge
3. If you don't know the labels, you'll need to find labeled data

**Option B: Use Filenames**
- Rename files to include "clean" or "contaminated" in the filename
- Example: `image001_clean.png`, `image002_contaminated.png`

### Step 3: Preprocess Dataset

```bash
python src/data_preprocessing.py
```

This will organize your data into train/validation/test splits.

## 🚀 Usage

### 1. Train the Model

```bash
python src/train.py
```

This will:
- Load preprocessed data
- Train the model
- Save best model to `models/saved_models/`
- Generate training plots

**Training time:** 30-60 minutes (depending on dataset size)

### 2. Test Single Image

```bash
python src/predict.py path/to/test/image.jpg
```

### 3. Run Web Interface

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
pesticide-detection/
├── dataset/                    # Dataset files
│   ├── raw/                    # Original Kaggle data
│   └── processed/              # Preprocessed data
├── models/                     # Trained models
│   └── saved_models/
├── src/                        # Source code
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── web_app/                    # Web interface
│   └── app.py
├── outputs/                    # Predictions and reports
│   ├── predictions/
│   └── reports/
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
└── README.md
```

## ⚠️ Important Notes

### About Testing with Regular Images

**You CANNOT use regular phone/camera images for testing if you train on hyperspectral data.**

Here's why:
- Hyperspectral images have 100+ spectral bands
- Regular RGB images have only 3 bands (Red, Green, Blue)
- Pesticide residues are often invisible to naked eye but visible in infrared wavelengths

**Solution:**
1. **For testing, use images from the same hyperspectral dataset**
2. **OR** modify the project to work with RGB images only (less accurate)

### Dataset Recommendations

**Better Alternative Datasets for Beginners:**

1. **Plant Disease Datasets** (RGB images):
   - PlantVillage Dataset
   - Kaggle Plant Disease Recognition
   
2. **Fruit Quality Datasets** (RGB images):
   - Fresh and Rotten Fruits Dataset
   - Fruit Recognition Dataset

These use regular RGB images that work with phone cameras!

### Troubleshooting

**Problem:** Model accuracy is very low
- **Solution:** You likely need more labeled data or better labels

**Problem:** "No module named X"
- **Solution:** Make sure virtual environment is activated and run `pip install -r requirements.txt`

**Problem:** Out of memory during training
- **Solution:** Reduce batch_size in config.yaml (try 16 or 8)

**Problem:** Web app won't start
- **Solution:** Make sure you've trained the model first using `python src/train.py`

## 📊 Expected Results

With proper labeled data (500+ images):
- Training accuracy: 85-95%
- Validation accuracy: 75-85%
- Real-world accuracy: Depends on data quality

## 🎓 Learning Resources

- [Hyperspectral Imaging Basics](https://en.wikipedia.org/wiki/Hyperspectral_imaging)
- [CNN for Image Classification](https://www.tensorflow.org/tutorials/images/cnn)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📝 TODO

- [ ] Label your dataset properly
- [ ] Run preprocessing
- [ ] Train initial model
- [ ] Test predictions
- [ ] Improve model accuracy
- [ ] Deploy web interface

## 🤝 Contributing

This is a learning project. Feel free to modify and experiment!

## 📄 License

Educational use only

---

**Good Luck with Your Project! 🚀**