from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'airlines_flights_data.csv'
OUT = ROOT / 'ml_results'
OUT.mkdir(exist_ok=True)
FEATURES = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']
CATEGORICAL = FEATURES[:7]
NUMERIC = FEATURES[7:]

def make_model():
    preprocessor = ColumnTransformer([
        ('categories', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
        ('numbers', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC)
    ])
    forest = RandomForestRegressor(n_estimators=80, max_depth=18, min_samples_leaf=3,
                                   max_features=0.8, n_jobs=-1, random_state=42)
    return Pipeline([('preprocessing', preprocessor), ('model', forest)])

def metrics(y, pred):
    return {'MAE': round(float(mean_absolute_error(y, pred)), 2),
            'RMSE': round(float(np.sqrt(mean_squared_error(y, pred))), 2),
            'R2': round(float(r2_score(y, pred)), 4)}

if not CSV.exists():
    raise FileNotFoundError(f'Put airlines_flights_data.csv beside this script: {CSV}')
df = pd.read_csv(CSV)
missing = [c for c in FEATURES + ['price', 'flight'] if c not in df.columns]
if missing:
    raise ValueError(f'Missing CSV columns: {missing}')
df = df.drop_duplicates(subset=FEATURES + ['price']).copy()
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])
df = df[df['price'] > 0].reset_index(drop=True)
X, y = df[FEATURES], df['price']
print(f'Usable records: {len(df):,}', flush=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('\nTraining Random Forest for overfitting check ...', flush=True)
model = make_model()
model.fit(X_train, y_train)
train_scores = metrics(y_train, model.predict(X_train))
test_predictions = model.predict(X_test)
test_scores = metrics(y_test, test_predictions)
print('Training:', train_scores, flush=True)
print('Testing: ', test_scores, flush=True)
pd.DataFrame([{'dataset': 'Training', **train_scores}, {'dataset': 'Testing', **test_scores}]).to_csv(OUT / 'overfitting_check.csv', index=False)

print('\nChecking errors by ticket class ...', flush=True)
errors = X_test[['class', 'days_left']].copy()
errors['absolute_error'] = np.abs(y_test.to_numpy() - test_predictions)
errors['booking_window'] = pd.cut(errors['days_left'], [0, 7, 14, 30, np.inf], labels=['1-7 days', '8-14 days', '15-30 days', '31+ days'])
rows = []
for column in ['class', 'booking_window']:
    for label, group in errors.groupby(column, observed=True):
        rows.append({'category': column, 'value': str(label), 'count': len(group), 'MAE': round(group['absolute_error'].mean(), 2)})
pd.DataFrame(rows).to_csv(OUT / 'subgroup_errors.csv', index=False)

print('\nCalculating feature importance ...', flush=True)
X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
importance = permutation_importance(model, X_sample, y_test.loc[X_sample.index], scoring='neg_mean_absolute_error', n_repeats=2, random_state=42)
features = pd.DataFrame({'feature': FEATURES, 'MAE_increase': importance.importances_mean}).sort_values('MAE_increase', ascending=False)
features.to_csv(OUT / 'feature_importance.csv', index=False)
plt.figure(figsize=(9, 5))
plt.barh(features['feature'][::-1], features['MAE_increase'][::-1])
plt.xlabel('Increase in prediction error when feature is shuffled')
plt.title('Random Forest feature importance')
plt.tight_layout()
plt.savefig(OUT / 'feature_importance.png', dpi=180)
plt.close()

print('\nTraining second model to test unseen flight IDs ...', flush=True)
groups = df['flight'].fillna('UNKNOWN').astype(str)
if groups.nunique() < 10:
    raise ValueError('Not enough distinct flight IDs for group validation')
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
group_model = make_model()
group_model.fit(X.iloc[train_idx], y.iloc[train_idx])
group_scores = metrics(y.iloc[test_idx], group_model.predict(X.iloc[test_idx]))
print('Unseen-flight test:', group_scores, flush=True)

report = {'random_split_training': train_scores, 'random_split_testing': test_scores,
          'unseen_flight_testing': group_scores, 'unseen_flight_train_rows': len(train_idx),
          'unseen_flight_test_rows': len(test_idx),
          'note': 'Historical dataset results. Random splits may share flight patterns. Feature importance is not causation.'}
(OUT / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('\nFinished! Open ml_results/overfitting_check.csv, subgroup_errors.csv, feature_importance.png and validation_report.json')
