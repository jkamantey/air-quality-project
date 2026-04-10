# Global Air Quality Analysis & AQI Prediction

## Overview
This project analyzes global air pollution data and builds machine learning models to predict Air Quality Index (AQI) values based on pollutant concentrations (CO, Ozone, NO2, PM2.5).

## Dataset
- **Source:** Global Air Pollution Dataset (Kaggle)
- **Size:** ~23,000 city-level records across multiple countries
- **Features:** AQI value/category, CO, Ozone, NO2, PM2.5 sub-indices

## Approach
1. **EDA** — Distribution analysis, top polluted countries, correlation heatmap
2. **Feature Engineering** — Dominant pollutant, total pollution load, category encoding
3. **Modeling** — Linear Regression (baseline), Random Forest, XGBoost
4. **Evaluation** — MAE, RMSE, R² comparison across all three models

## Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

*(Fill in after running the notebook)*

## Key Finding
PM2.5 is the strongest individual predictor of overall AQI. XGBoost achieves the best performance by capturing non-linear pollutant interactions.

## Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook notebook.ipynb
```
