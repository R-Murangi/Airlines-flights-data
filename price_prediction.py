"""Airline ticket price prediction using reproducible scikit-learn pipelines.

Place this file beside airlines_flights_data.csv, then run:
    python price_prediction.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'airlines_flights_data.csv'
OUT = ROOT / 'ml_results'
OUT.mkdir(exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError(f'CSV not found: {CSV}\nPut airlines_flights_data.csv beside price_prediction.py')

df = pd.read_csv(CSV)
print(f'Loaded {len(df):,} rows and {len(df.columns)} columns')
print('Missing values:', int(df.isna().sum().sum()))
print('Exact duplicate rows:', int(df.duplicated().sum()))

features = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
            'destination_city', 'class', 'duration', 'days_left']
missing = [c for c in features + ['price'] if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

# Exclude the arbitrary row index and flight identifier to keep a clear, interpretable baseline.
df = df.drop_duplicates(subset=features + ['price']).copy()
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])
df = df[df['price'] > 0]
print(f'Usable records after duplicate and target checks: {len(df):,}')

X = df[features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f'Train rows: {len(X_train):,}; test rows: {len(X_test):,}')

categorical = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
               'destination_city', 'class']
numeric = ['duration', 'days_left']
preprocess = ColumnTransformer([
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ]), categorical),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ]), numeric),
])

models = {
    'Mean baseline': DummyRegressor(strategy='mean'),
    'Ridge regression': Ridge(alpha=1.0),
    # Bounded complexity to run on a typical laptop; increase after checking runtime.
    'Random forest': RandomForestRegressor(n_estimators=80, max_depth=18,
                                           min_samples_leaf=3, max_features=0.8,
                                           n_jobs=-1, random_state=42),
}
results = []
predictions = {}
for name, estimator in models.items():
    print(f'\nTraining {name} ...', flush=True)
    model = Pipeline([('preprocessing', preprocess), ('model', estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    result = {
        'model': name,
        'MAE': float(mean_absolute_error(y_test, pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
        'R2': float(r2_score(y_test, pred)),
    }
    results.append(result)
    print(f"MAE: {result['MAE']:,.2f} | RMSE: {result['RMSE']:,.2f} | R²: {result['R2']:.4f}")

scores = pd.DataFrame(results).sort_values('MAE')
scores.to_csv(OUT / 'model_comparison.csv', index=False)

best_name = scores.iloc[0]['model']
print(f'\nBest test MAE: {best_name}')
# For visualisation only, sample a fixed subset of the test predictions.
rng = np.random.default_rng(42)
indices = rng.choice(len(y_test), size=min(1500, len(y_test)), replace=False)
actual = y_test.to_numpy()[indices]
predicted = predictions[best_name][indices]
plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, s=9, alpha=0.35)
lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
plt.plot([lo, hi], [lo, hi], '--', color='black', label='Perfect prediction')
plt.xlabel('Actual ticket price (dataset currency)')
plt.ylabel('Predicted ticket price (dataset currency)')
plt.title(f'Actual vs Predicted — {best_name}')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'actual_vs_predicted.png', dpi=180)
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(scores['model'], scores['MAE'])
plt.xlabel('Mean Absolute Error (lower is better)')
plt.title('Model comparison — Test MAE')
plt.tight_layout()
plt.savefig(OUT / 'model_comparison.png', dpi=180)
plt.close()

metadata = {
    'train_rows': len(X_train), 'test_rows': len(X_test),
    'features': features, 'random_state': 42,
    'note': 'Random train/test split. Results are dataset-specific, not live airfare guarantees.'
}
(OUT / 'experiment_details.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(f'\nSaved results in: {OUT}')
print('Files: model_comparison.csv, model_comparison.png, actual_vs_predicted.png, experiment_details.json')
