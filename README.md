---
title: Histomancer
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# Lung Tissue Analysis Tool

AI-powered lung histopathology classifier matching the original Flask application design.

## Features

- **Image Classification**: Upload histopathology images for instant classification
- **Grad-CAM Visualization**: See which regions the model focuses on
- **Saliency Maps**: Understand pixel-level feature importance
- **Confidence Scores**: Get probability scores for all three classes

## Model Details

- **Architecture**: EfficientNetB0 with transfer learning
- **Training Data**: LC25000 + LungHist700 datasets
- **Test Accuracy**: 98.25%

## Disclaimer

This is an educational demonstration tool and is **NOT** intended for clinical diagnostic use.
