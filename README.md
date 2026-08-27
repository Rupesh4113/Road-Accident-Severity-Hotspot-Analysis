# Road Accident Severity & Hotspot Analysis

This project implements an end-to-end machine learning and geospatial analysis pipeline to classify the severity of traffic incidents and identify spatial risk factors (such as weather, road geometry, and lighting) using traffic accident datasets from the UK and the US.

## Key Features & Techniques
- **Imbalanced Classification**: Class balancing using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure minority (fatal/high-severity) classes are correctly classified.
- **Predictive Modeling**: Comparison between **Random Forest Classifier** and **CatBoost Classifier**.
- **Feature Importance**: Analysis of key risk factors (weather conditions, lighting, and road geometry).
- **Geospatial Hotspot Analysis**: Using **Kernel Density Estimation (KDE)** to identify high-density accident hotspots.
- **Interactive Visualizations**: Zoomable map outputs with Folium displaying density heatmaps and highlighted hotspot markers.

---

## Datasets Supported

1. **UK Road Safety Data (STATS19 2022)**:
   - Programmatically fetched from the UK Department for Transport (DfT) portal.
   - Includes ~106k collision records with fields covering severity, vehicle counts, weather conditions, lighting, road type, and GPS coordinates.
   
2. **US Accidents Dataset (Kaggle)**:
   - Built to handle the massive US Accidents countrywide dataset.
   - **Out-of-the-box support**: If no US Accidents CSV is present in `data/`, the pipeline automatically generates a representative synthetic sample (~15k clustered coordinates around New York, Los Angeles, Chicago, Houston, and Miami) so that the entire repository is fully executable immediately.
   - **Using the real dataset**: Simply download the US Accidents Kaggle dataset (e.g., `US_Accidents_Dec23.csv`) and place it inside the `data/` directory. The preprocessing module will automatically detect and run the pipeline on the real data.

---

## Folder Structure

```text
├── data/                    # Downloaded and generated CSV datasets
├── models/                  # Trained classifier models (.joblib)
├── outputs/                 # Evaluation plots and interactive Folium HTML maps
├── src/
│   ├── __init__.py          # Declares src package
│   ├── downloader.py        # Manages directory setups and data downloading/sampling
│   ├── data_preprocessing.py# Data cleaning, scaling, cyclical feature encoding
│   ├── model_training.py    # SMOTE balancing, classification, evaluations, importances
│   └── spatial_analysis.py  # 2D KDE computations and Folium HeatMap generation
├── main.py                  # Orchestrator entrypoint script
├── requirements.txt         # Package dependencies
└── README.md                # Repository documentation
```

---

## Setup & Installation

Ensure you have Python (version 3.10+ recommended, tested up to 3.14) and `pip` installed.

1. Clone or download the repository into your workspace.
2. Install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

To run the entire pipeline—including data download/generation, preprocessing, modeling, evaluation, and interactive map generation:

```bash
python main.py
```

### Outputs Generated:
- **Trained Models**: Saved to the `models/` directory:
  - `uk_random_forest.joblib` & `uk_catboost.joblib`
  - `us_random_forest.joblib` & `us_catboost.joblib`
- **Evaluation Plots**: Saved to `outputs/` showing confusion matrices and feature importance charts:
  - `uk_random_forest_cm.png` & `uk_random_forest_fi.png`
  - `uk_catboost_cm.png` & `uk_catboost_fi.png`
  - `us_random_forest_cm.png` & `us_random_forest_fi.png`
  - `us_catboost_cm.png` & `us_catboost_fi.png`
- **Interactive Maps**: Saved to `outputs/`. Double-click these files to open them in any web browser:
  - `uk_hotspots.html`
  - `us_hotspots.html`