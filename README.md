# EnergyPred — Hourly Energy Demand Forecasting

EnergyPred explores hourly electricity demand in the AEP region and compares machine learning models for forecasting consumption. It uses 121,273 hourly readings from December 2004 through January 2018, with demand measured in megawatts (MW).

## What this project does

- **Exploratory analysis** — examines demand over time, its distribution, and average hourly patterns.
- **Time-based features** — adds hour, day of week, and month, along with lagged demand (`lag1`, `lag24`, `lag168`) and rolling statistics over 24 and 168 hours.
- **Chronological evaluation** — trains on the first 80% of the data and evaluates on the most recent 20% to preserve time order.
- **Model comparison** — evaluates Linear Regression, Random Forest, and XGBoost.
- **Saved model artifacts** — exports the trained estimators as `.joblib` files and the input feature order as `feature_columns.json`.

## Test-set results

| Model | RMSE (MW) | MAE (MW) | R² |
|---|---:|---:|---:|
| Linear Regression | 532.48 | 396.56 | 0.9526 |
| Random Forest | 272.31 | — | 0.9876 |
| XGBoost | 258.54 | — | 0.9888 |

The notebook reports MSE and R² for Random Forest and XGBoost. Their RMSE values above are calculated as the square root of the reported MSE. MAE was not reported for those two models.

## Project files

```text
EnergyPred/
├── AEP_hourly.csv          # Hourly AEP demand data
├── EnergyPred.ipynb        # Data exploration, feature engineering, training, and evaluation
├── feature_columns.json    # Feature names and order used by the saved models
├── LinearRegression.joblib # Trained Linear Regression model
├── RandomForest.joblib     # Trained Random Forest model (Git LFS)
└── XGBoost.joblib           # Trained XGBoost model
```

## Run the notebook

1. Clone the repository and download the Git LFS model file:

   ```bash
   git lfs install
   git clone https://github.com/saisubhransu/EnergyPredX.git
   cd EnergyPredX
   git lfs pull
   ```

2. Install the notebook dependencies:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
   ```

3. Start Jupyter and open `EnergyPred.ipynb`:

   ```bash
   jupyter notebook
   ```

## Dataset

The AEP hourly consumption data is from the [Hourly Energy Consumption dataset on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption). The notebook uses `AEP_hourly.csv` and its `Datetime` and `AEP_MW` columns.

## Notes

- The trained models expect the feature names and order in `feature_columns.json`.
- Lag and rolling features depend on prior observations. For future multi-hour forecasts, update these features from the latest observed values and/or prior predictions at each step.
- `RandomForest.joblib` is stored with Git LFS because its size exceeds GitHub's regular per-file limit.
