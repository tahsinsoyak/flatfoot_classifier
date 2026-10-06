# Deep Learning-Based Automated Flatfoot (Pes Planus) Classification from Weight-Bearing Lateral Radiographs

This repository contains the end-to-end deep learning classification pipeline for automated diagnosis of **Flatfoot (Pes Planus)** vs. **Normal Foot** using weight-bearing lateral foot radiographs.

## 1. Project Background & Rationale
Previous geometric landmark-based approaches (e.g. measuring Meary's angle and calcaneal pitch via multi-bone segmentation) face severe practical bottlenecks:
- Extreme sensitivity to millimeter-level landmark errors
- Reliance on clinician-drawn support/ground lines
- Lack of dense multi-bone landmark annotations
- High heuristic complexity

This project directly classifies clinical lateral radiographs using deep convolutional neural networks (CNNs) and vision models, combined with medical image preprocessing (contrast enhancement, ROI cropping, standardized canonical orientation) and Explainable AI (Grad-CAM/Grad-CAM++) to validate anatomical focus against clinical standards.

## 2. Dataset Overview
- **Pes Planus (Flatfoot):** 908 lateral radiographs
- **Normal Foot:** 621 lateral radiographs
- **Total:** 1,529 clinical weight-bearing lateral radiographs
- **Image Type:** 16-bit Grayscale (uint16) radiographs (~3000x2400 to ~3000x3026)

## 3. Pipeline Architecture
1. **Raw Ingestion & Verification:** Integrity checks and metadata cataloging.
2. **Medical Preprocessing & ROI Extraction:**
   - 16-bit to 8-bit dynamic range windowing with Contrast Limited Adaptive Histogram Equalization (CLAHE).
   - Automated foot ROI extraction (removing standing platform artifacts and upper tibia/fibula).
   - Canonical orientation normalization (toes pointing in a consistent direction).
3. **Stratified Splitting:** 70% Train, 15% Validation, 15% Test.
4. **Model Benchmarking:**
   - ResNet-50 (Baseline standard)
   - EfficientNet-B2 / B3 (Parameter-efficient feature extraction)
   - ConvNeXt-Tiny (Modern pure-convolutional architecture)
5. **Explainability & Validation:**
   - Grad-CAM / Grad-CAM++ anatomical saliency maps
   - Confusion matrices, ROC-AUC curves, sensitivity/specificity analysis.

## 4. Directory Structure
```
flatfoot_classifier/
├── configs/           # Experiment and training configurations
├── data/
│   ├── raw/           # Original extracted radiographs
│   ├── processed/     # ROI-cropped and standardized images
│   └── splits/        # Train/val/test CSV manifests
├── experiments/       # Checkpoints, logs, figures, reports
├── scripts/           # Standalone execution scripts
├── src/
│   ├── preprocessing/ # Image windowing, CLAHE, ROI detection
│   ├── models/        # Model definitions, factory, Grad-CAM
│   ├── training/      # Trainer loops, losses, schedulers
│   ├── evaluation/    # Metric calculators, ROC curves
│   └── dataset.py     # PyTorch Dataset and augmentations
├── requirements.txt
└── README.md
```
