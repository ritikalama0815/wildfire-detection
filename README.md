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

## Preprocessing 

1. **Transform both train and val dataset**:
      - Resize
      - randomly flip,
      - randomly rotate,
      - apply color jitter, and
      - normalize to [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

Training begins after transformation and preprocessings. 

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
