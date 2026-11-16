# Pneumonia Detection from Chest X-Rays using DenseNet121

## Overview

This project implements a Deep Learning-based Pneumonia Detection system using chest X-ray images and **DenseNet121 Transfer Learning**.

The model classifies chest X-ray images into two categories:

- **NORMAL** – Normal chest X-ray.
- **PNEUMONIA** – Chest X-ray classified as pneumonia.

The project uses ImageNet-pretrained DenseNet121, a two-phase training strategy, data augmentation, class weighting, threshold optimization, and Grad-CAM explainability.

## Dataset

**Chest X-Ray Images (Pneumonia) – Kaggle**

🔗 Dataset Download: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains two classes:
- NORMAL
- PNEUMONIA

### Dataset Structure

After downloading and extracting the dataset, the folder structure should look like this:

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Dataset Preparation:** The original validation folder contains only 16 images. Therefore, the notebook combines the original training and validation data and creates a new stratified 85:15 training-validation split. The official test set is kept separate for final evaluation.

## Technologies Used

- Python
- TensorFlow
- Keras
- DenseNet121
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyze class distribution.
- Visualize sample NORMAL and PNEUMONIA X-ray images.
- Examine training, validation, and test datasets.

### 2. Image Preprocessing
- Resize images to 224 × 224 pixels.
- Convert grayscale images into three channels.
- Apply DenseNet121 preprocessing.
- Remove images that cannot be decoded.

### 3. Data Augmentation

The following augmentation techniques are applied to training images:

- Random Rotation
- Random Zoom
- Random Translation
- Random Contrast

Horizontal flipping is intentionally avoided.

### 4. Model Architecture – DenseNet121

DenseNet121 is used as the pretrained feature extraction model with ImageNet weights.

A custom classification head is added:

- Global Average Pooling
- Dropout (0.4)
- Dense Layer (128 neurons, ReLU)
- Dropout (0.2)
- Output Layer (1 neuron, Sigmoid)

### 5. Two-Phase Model Training

**Phase 1 – Feature Extraction**
- Freeze the pretrained DenseNet121 base.
- Train the custom classification head.
- Adam optimizer with learning rate 0.001.
- Maximum 8 epochs.

**Phase 2 – Fine-Tuning**
- Unfreeze approximately the last 40 DenseNet121 layers.
- Keep Batch Normalization layers frozen.
- Fine-tune using Adam with learning rate 0.00001.
- Maximum 15 epochs.

Additional training techniques:
- Class weighting for class imbalance.
- Early stopping based on validation AUC.
- Model checkpointing.
- Reduce learning rate on validation loss plateau.

### 6. Model Evaluation

The model is evaluated on the official test dataset using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Sensitivity
- Specificity

### 7. Threshold Optimization

Instead of relying only on the default classification threshold of 0.50, the notebook searches thresholds from 0.05 to 0.95.

The threshold producing the highest F1-score on the validation dataset is selected and then applied to the official test dataset.

### 8. Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize the image regions influencing the model's predictions.

It generates heatmap overlays to help inspect which regions of the chest X-ray contribute to the classification.

### 9. Error Analysis

The notebook identifies incorrectly classified test images and visualizes misclassified examples along with their predicted pneumonia probabilities.

### 10. Model Saving and Prediction

The trained model and optimized threshold are saved for future use.

A custom `predict_image()` function allows prediction on an individual chest X-ray image and displays its Grad-CAM visualization.

## Installation

Install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/chest-xray-pneumonia-detection.git
```

2. Download the dataset from Kaggle using the link provided above.

3. Extract the dataset into the project environment.

4. Open the notebook:

`pneumonia_detection_chest_xray.ipynb`

5. Configure the dataset path using the `DATA_DIR` variable if necessary.

6. Select a GPU-enabled environment (recommended).

7. Run all notebook cells sequentially.

## Repository Structure

```text
chest-xray-pneumonia-detection/
│
├── pneumonia_detection_chest_xray.ipynb
│
├── outputs/
│   ├── phase1.keras
│   ├── best_model.keras
│   ├── pneumonia_densenet121.keras
│   ├── threshold.txt
│   │
│   └── images/
│       ├── 1_training_curves.png
│       ├── 2_confusion_matrix.png
│       ├── 3_roc_curve.png
│       ├── 4_precision_recall_curve.png
│       ├── 5_gradcam_*.png
│       └── 6_misclassified_examples.png
│
└── README.md
```

## Future Improvements

- Compare DenseNet121 with EfficientNetB0 and ResNet50.
- Experiment with ensemble learning.
- Implement k-fold cross-validation with patient-level splitting where patient IDs are available.
- Experiment with focal loss for class imbalance.
- Explore bacterial and viral pneumonia classification.

## Disclaimer

This project is developed strictly for educational and research purposes.

The model is not a certified medical diagnostic system and must not be used as a replacement for professional medical advice or clinical diagnosis.

## Author

**Akshay Kumar**

GitHub: https://github.com/YOUR-USERNAME# chest-xray-pneumonia-detection
