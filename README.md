# Machine Learning Assignment Project

**Author:** Zameel  
**Technologies:** Python, Streamlit, Scikit-Learn, OpenCV (Headless), FPDF2  

A machine learning project featuring two applications selectable from the home interface: a computer vision face detection tool and a bike sharing demand regression pipeline with 5-fold cross-validation, evaluation plots, an A4 PDF report generator, and an interactive prediction simulator.

---

## Applications

### 1. Face Detection Application (`modules/face_detection.py`)
- Uses OpenCV Haar Cascades (`haarcascade_frontalface_default` and `haarcascade_frontalface_alt2`) for multi-scale face detection.
- Detects eye positions and smiles within identified face regions.
- Supports input from sample images, image uploads (JPG, PNG), and device camera input (`st.camera_input`).
- Includes an optional privacy blur mode to mask detected faces.
- Displays bounding box telemetry (coordinates, dimensions, frame area percentage) and allows downloading cropped faces as a ZIP file.

### 2. Bike Sharing Demand Prediction (`modules/bike_sharing_ml.py`)
- Based on the UCI Machine Learning Repository Bike Sharing Dataset (17,379 hourly records).
- Implements an end-to-end regression workflow predicting total hourly rentals (`cnt`) from 12 calendar and weather features.
- Evaluates 5 regression models with stratified 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`):
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Extra Trees Regressor
  - Ridge Linear Regression
  - K-Nearest Neighbors Regressor
- Automatically selects the best model based on mean cross-validation R-squared and RMSE.
- Generates and saves 6 evaluation plots to the `evaluation_plots/` directory:
  - `model_comparison.png`: 5-fold CV R-squared and RMSE comparison bar chart.
  - `actual_vs_predicted.png`: Scatter plot with identity line and error metrics.
  - `residuals_distribution.png`: Residual error vs. predictions and error histogram.
  - `feature_importance.png`: Feature importance ranking.
  - `hourly_demand_trend.png`: Average actual vs. predicted hourly demand profile.
  - `correlation_matrix.png`: Environmental and temporal feature correlation heatmap.
- Compiles an academic 3-page A4 PDF evaluation report using `fpdf2` with a 1-click download button in the app.
- Provides an interactive simulator to test custom weather and calendar scenarios with instant predictions and commuter vs. casual breakdown.

---

## Local Setup and Running

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Installation
Clone or navigate to the project directory:
```bash
cd ML_Assignment
```

Install required dependencies:
```bash
pip install -r requirements.txt
```
*(Or if using `uv`: `uv pip install -r requirements.txt`)*

### 2. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Automated Tests

Run the test suite to verify the face detection pipeline, cross-validation workflow, plot generation, and PDF compilation:

```bash
python -m unittest tests/test_suite.py
```

---

## Project Structure

```text
ML_Assignment/
├── .streamlit/
│   └── config.toml           # Streamlit server and theme configuration
├── data/
│   ├── hour.csv              # UCI Bike Sharing hourly dataset
│   ├── day.csv               # UCI Bike Sharing daily dataset
│   └── Readme.txt            # Dataset documentation
├── evaluation_plots/         # Directory for saved evaluation charts and reports
├── modules/
│   ├── __init__.py
│   ├── face_detection.py     # Application 1: Face detection
│   ├── bike_sharing_ml.py    # Application 2: Bike sharing regression
│   └── pdf_generator.py      # A4 PDF report compiler
├── utils/
│   ├── __init__.py
│   ├── sample_generator.py   # Synthetic test portrait fixtures
│   └── ui_theme.py           # Clean plot styling configuration
├── tests/
│   └── test_suite.py         # Automated test cases
├── app.py                    # Main Streamlit router and home page
├── requirements.txt          # Python dependencies
├── packages.txt              # Linux system dependencies for deployment
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```
