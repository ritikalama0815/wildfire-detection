# Burning Signals, Forecasting Wildfire

**Author:** Ritika Lama
**Project:** Summer Research 2025  
**Objective:** To be able to detect wildfire in the certain region based on its satellite images.

---

## Overview

This project proposes a deep learning model to detect wildfire using satellite imagery to classify two key categories:

- fire
- No Fire

This uses EfficientNet-B0 architecture as its backbone and also includes Squeeze-and-Excitation attention and proxy normalization instead of batch normalization.

---

## Folder Structure

```
wildfire-detection/
│
├── Wildfire/              
│   ├── train/                   # Training set with class folders
│   ├── val/                     # Validation set with class folders
│
├── pth_files/                  # Trained model weights (.pth) with best results
├── results/                    # Evaluation results (confusion matrices, metrics)
├── diagrams/                    # confusion matrix and other plits
│
├── proxy_se_eff
│
└── README.md                          # documentation
```

---

## Models and Architectures

| File                          | Description                                       |
|------------------------------|---------------------------------------------------|
| `proxy_se_eff`  | EfficientNetB0 + SENet + Proxy Normalization                   |

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


## Dependencies

Install libraries:

```bash
pip install torch torchvision timm  matplotlib seaborn scikit-learn
```

Use a virtual environment for reproducibility (see `projectvenv/`).

---


## References

* Dataset: https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
* Architecture Base: EfficientNet, SE-Net
* Libraries: PyTorch, Timm, Matplotlib

---
