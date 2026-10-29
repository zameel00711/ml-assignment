# Machine Learning Assignment Project Plan

**Author:** Zameel  
**Status:** Completed  

---

## Overview

A Streamlit project containing two applications:
1. **Face Detection Application (`modules/face_detection.py`)**: Uses OpenCV Haar Cascades to detect faces, eyes, and smiles, with camera input and optional privacy blurring.
2. **Bike Sharing Demand Prediction (`modules/bike_sharing_ml.py`)**: End-to-end regression assignment using the UCI Bike Sharing Dataset, 5-fold cross-validation, plot exports, A4 PDF report generation, and an interactive prediction simulator.

---

## Implementation Details

### Application 1: Face Detection
- Model: OpenCV Haar Cascades (`haarcascade_frontalface_default`, `haarcascade_frontalface_alt2`, `haarcascade_eye`, `haarcascade_smile`).
- Features: Scale factor and neighbor tuning, eye/smile tracking, privacy blur, face crop gallery with ZIP download, bounding box telemetry table.
- Inputs: Sample images, user upload, camera input.

### Application 2: Bike Sharing Demand Prediction
- Dataset: UCI Bike Sharing hourly data (17,379 rows, 12 explanatory features).
- Models Evaluated: Random Forest Regressor, Gradient Boosting Regressor, Extra Trees Regressor, Ridge Linear Regression, K-Nearest Neighbors Regressor.
- Validation: Stratified 5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- Outputs:
  - Six evaluation plots saved to `evaluation_plots/`: model comparison, actual vs predicted, residuals, feature importance, hourly trend curve, and correlation heatmap.
  - Three-page A4 PDF report compiled with `fpdf2`.
  - Interactive prediction simulator with preset scenarios and commuter/casual demand breakdown.
