# Airline Flight Price Analysis and Prediction

An end-to-end Python data science project exploring airline ticket prices and building machine learning models to predict prices from flight characteristics.

## Project overview

This project examines a historical dataset of airline flights, explores relationships between flight details and ticket prices, and compares regression models. It also evaluates whether a model trained on one set of flight identifiers can generalise to unseen flight identifiers.

**This is an educational project using historical data, not a live airfare pricing service.**

## Dataset

- Original dataset: **300,153 rows and 12 columns**
- Records retained after removing repeated feature/price combinations and invalid targets: **297,940**
- Target variable: `price`
- Prediction features: `airline`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`, `duration`, `days_left`
- The original `index` and `flight` identifier are excluded from prediction inputs.

The dataset contains airline flights between Indian cities. Prices are presented in the dataset's currency (Indian rupees, ₹). Add the original dataset source and license here before publishing if you know them.

## Tools and libraries

Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Workflow

1. **Explore the data:** inspect columns, data types, missing values, distributions, and relationships.
2. **Prepare the data:** remove repeated feature/price combinations, exclude invalid prices, impute missing predictors, one-hot encode categorical features, and scale numerical features.
3. **Train models:** compare a mean-price baseline, Ridge Regression, and Random Forest Regression.
4. **Evaluate models:** report Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² on a held-out 20% random test split.
5. **Validate:** compare training and testing performance and separately test on flight identifiers not used for training.
6. **Interpret:** examine permutation feature importance and prediction errors by subgroup.

## Results

### Random test split

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Mean baseline | 19,779.74 | 22,663.84 | ~0.0000 |
| Ridge Regression | 4,576.93 | 6,766.27 | 0.9109 |
| **Random Forest** | **1,467.12** | **2,928.98** | **0.9833** |

### Additional Random Forest validation

| Evaluation | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Training set | 1,290.00 | 2,552.15 | 0.9874 |
| Random test set | 1,467.12 | 2,928.98 | 0.9833 |
| Unseen flight IDs | 2,577.35 | 4,789.84 | 0.9494 |

The higher error on unseen flight IDs suggests that predicting unfamiliar flight patterns is more challenging than predicting records drawn from a random split of the dataset.

### Feature importance

Permutation importance identified **ticket class** as the most influential feature for this model, followed by **flight duration**. This describes the model's reliance on these variables, **not** a causal relationship.

![Random Forest feature importance](ml_results/feature_importance.png)

![Actual versus predicted ticket prices](ml_results/actual_vs_predicted.png)

![Model comparison](ml_results/model_comparison.png)

## Project files

```text
.
├── airlines_flights_data.csv
├── temp.py
├── price_prediction.py
├── validate_airline_model.py
├── README.md
├── requirements.txt
├── images/
└── ml_results/
```

- `temp.py`: exploratory analysis and charts saved to `images/`.
- `price_prediction.py`: baseline and regression model comparison, with results saved to `ml_results/`.
- `validate_airline_model.py`: overfitting, unseen-flight, subgroup, and feature-importance checks.
- `ml_results/`: CSV, JSON, and PNG outputs from the experiments.

## How to run

1. Install **Python 3.10+**.
2. Download or clone this repository.
3. Place `airlines_flights_data.csv` in the same folder as the Python scripts.
4. Open a terminal in the project folder.
5. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. Run the exploratory analysis:

   ```bash
   python temp.py
   ```

7. Train and compare models:

   ```bash
   python price_prediction.py
   ```

8. Run additional validation:

   ```bash
   python validate_airline_model.py
   ```

The scripts save their results in `images/` and `ml_results/`. Model training may take several minutes.

## Limitations and next steps

- A random row split can contain similar routes and flight patterns in training and testing.
- The unseen-flight-ID evaluation is a more demanding check, but is not equivalent to a future-date or external-dataset test.
- Historical prices may not represent today's airline market.
- Feature importance measures predictive usefulness, not causality.
- Future work could include time-based validation (if dates become available), hyperparameter tuning with cross-validation, and a small application for entering flight details.

## Author

Portfolio project by [R-Murangi](https://github.com/R-Murangi).
