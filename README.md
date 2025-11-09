# ML Model for Satellite Imagery - Climate Change Analysis

**Author:** Rahul Chaudhari  
**Project:** Summer Research 2025  
**Goal:** Classifying satellite images into meaningful land-use categories to support climate change research using efficient deep learning models.

---

## Overview

This project builds and compares multiple deep learning models using satellite imagery to classify four key categories:

- cloudy
- desert
- green_area
- water

We experiment with EfficientNet-based architectures, including:
- Custom lightweight heads
- Squeeze-and-Excitation attention
- EfficientNet-Lite variants

---

## Folder Structure

```
MLModel_forSatelliteImagery_climate_change_analysis/
│
├── processed_data/              # Final dataset split into:
│   ├── train/                   # Training set with class folders
│   ├── val/                     # Validation set with class folders
│   └── test/                    # Test set with class folders
│
├── unprocessed_data/           # Raw dataset with folders:
│   ├── cloudy/
│   ├── desert/
│   ├── green_area/
│   └── water/
│
├── pth_files/                  # Trained model weights (.pth)
├── results/                    # Evaluation results (confusion matrices, metrics)
├── visuals/                    # Image distribution plots and samples
│
├── data_download_prep.ipynb           # Kaggle download and unzip logic
├── efficientnetb0_SELayer.ipynb       # Model 3: EfficientNetB0 with SE-enhanced head
├── efficientnetb0_with2dconv.ipynb    # Model 1: EfficientNetB0 with minimal 2D head
├── efficientnetlite.ipynb             # Model 2: EfficientNetLite with SE module
│
├── pyfiles_ipynb/                     # Converted .py versions of notebooks (optional)
├── scratchfiles.ipynb                 # Sandbox for temporary testing
└── README.md                          # Project documentation
```

---

## Models and Architectures

| File                          | Description                                       |
|------------------------------|---------------------------------------------------|
| `efficientnetb0_with2dconv`  | EfficientNetB0 + minimal 2D head (lightweight)   |
| `efficientnetlite`           | EfficientNet-Lite + Squeeze-and-Excitation head  |
| `efficientnetb0_SELayer`     | EfficientNetB0 with deeper SE-enhanced structure |

Each model is trained using the same preprocessed and augmented dataset for fair evaluation.

---

## Preprocessing Pipeline

1. **Downloading** from Kaggle using `kagglehub`
2. **Unpacking** and placing into `unprocessed_data`
3. **Augmenting** underrepresented classes using `albumentations`
4. **Resizing** images to `224x224`
5. **Splitting** into train/val/test using a controlled script
6. **Saving** final data in `processed_data/`

All preprocessing is done **once**, so training can begin immediately.

---

## Evaluation

Each model is evaluated using:
- Accuracy
- Confusion Matrix (saved in `results/`)
- Classification Report (Precision, Recall, F1-Score)

---

## Running Predictions

You can test new images using a prompt interface:

```bash
Enter model type: [minimal | lite | se]
Enter image path (or Q to quit): path/to/image.jpg
Predicted Class: green_area
```

Models will automatically load class mappings from the training dataset.

---

## Dependencies

Install core libraries using:

```bash
pip install torch torchvision timm albumentations matplotlib seaborn scikit-learn
```

Use a virtual environment for reproducibility (see `projectvenv/`).

---

## Future Scope

* Live satellite stream classification
* Grad-CAM visualization for interpretability
* Deployment to edge devices (Raspberry Pi, Jetson Nano)
* Extending to pixel-level segmentation

---

## Acknowledgments

* Dataset: [Kaggle - Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)
* Architecture Base: EfficientNet, SE-Net
* Toolkits: PyTorch, Albumentations, Timm, Matplotlib

---

This project serves as an academic initiative under Summer Research 2025 and may be extended for real-world satellite analytics applications.