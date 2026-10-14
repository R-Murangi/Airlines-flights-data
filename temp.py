"""Airline flight data analysis and chart export."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / 'airlines_flights_data.csv'
IMAGES_DIR = BASE_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid', palette='husl')


def save_graph(filename):
    path = IMAGES_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved graph: {path}')
    plt.close()


if not CSV_FILE.exists():
    raise FileNotFoundError(f'CSV file not found: {CSV_FILE}\nPut airlines_flights_data.csv next to temp.py.')

df = pd.read_csv(CSV_FILE)
required = ['airline', 'source_city', 'destination_city', 'departure_time',
            'arrival_time', 'class', 'duration', 'days_left', 'price']
missing_columns = [c for c in required if c not in df.columns]
if missing_columns:
    raise ValueError(f'Missing columns: {missing_columns}')

print('DATASET OVERVIEW')
print('Shape:', df.shape)
print(df.head().to_string(index=False))
print('\nMissing values:\n', df.isna().sum().to_string())
print('\nDuplicate rows (excluding index):', df.drop(columns=['index'], errors='ignore').duplicated().sum())
print('\nSummary:\n', df.describe(include='all').to_string())

# 1. Airline distribution
counts = df['airline'].value_counts()
plt.figure(figsize=(10, 8))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Airline Records', fontsize=16, fontweight='bold')
plt.axis('equal')
save_graph('01_airline_distribution.png')

# 2. Departure and arrival times
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, column, title in zip(axes, ['departure_time', 'arrival_time'],
                             ['Departure Time Distribution', 'Arrival Time Distribution']):
    df[column].value_counts().plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Number of Records')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_graph('02_departure_arrival_times.png')

# 3. Source and destination cities
fig, axes = plt.subplots(2, 1, figsize=(13, 11))
for ax, column, title in zip(axes, ['source_city', 'destination_city'],
                             ['Source City Distribution', 'Destination City Distribution']):
    df[column].value_counts().plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Number of Records')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
save_graph('03_city_distributions.png')

# 4. Price by airline
print('\nPRICE BY AIRLINE\n', df.groupby('airline')['price'].agg(['mean', 'median', 'std', 'min', 'max']).round(2))
plt.figure(figsize=(13, 7))
sns.boxplot(data=df, x='airline', y='price', hue='airline', palette='Set2', legend=False, showfliers=False)
plt.title('Ticket Price by Airline (Outliers Hidden for Readability)')
plt.xlabel('Airline')
plt.ylabel('Price (INR)')
plt.xticks(rotation=30)
plt.tight_layout()
save_graph('04_prices_by_airline.png')

# 5. Average price by time of day
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, column, title in zip(axes, ['departure_time', 'arrival_time'],
                             ['Average Price by Departure Time', 'Average Price by Arrival Time']):
    df.groupby(column)['price'].mean().sort_values().plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Average Price (INR)')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
save_graph('05_price_by_time.png')

# 6. Average price by source/destination
fig, axes = plt.subplots(2, 1, figsize=(13, 11))
for ax, column, title in zip(axes, ['source_city', 'destination_city'],
                             ['Average Price by Source City', 'Average Price by Destination City']):
    df.groupby(column)['price'].mean().sort_values().plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Average Price (INR)')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
save_graph('06_price_by_city.png')

# 7. Last-minute vs other bookings
last_minute = df.loc[df['days_left'].isin([1, 2]), 'price']
regular = df.loc[df['days_left'] > 2, 'price']
print('\nLAST MINUTE MEAN:', last_minute.mean(), 'OTHER MEAN:', regular.mean())
plt.figure(figsize=(9, 6))
plt.bar(['1–2 days left', 'More than 2 days left'], [last_minute.mean(), regular.mean()])
plt.title('Average Price by Days Until Departure')
plt.ylabel('Average Price (INR)')
plt.tight_layout()
save_graph('07_last_minute_vs_regular.png')

# 8. Travel class
print('\nPRICE BY CLASS\n', df.groupby('class')['price'].agg(['mean', 'median', 'std', 'count']).round(2))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df, x='class', y='price', hue='class', palette='Set2', legend=False, showfliers=False, ax=axes[0])
axes[0].set_title('Price Distribution by Class')
axes[0].set_ylabel('Price (INR)')
df.groupby('class')['price'].mean().plot.bar(ax=axes[1])
axes[1].set_title('Average Price by Class')
axes[1].set_ylabel('Average Price (INR)')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
save_graph('08_price_by_class.png')

# 9. Duration vs price: sample for readability
sample = df.sample(n=min(12000, len(df)), random_state=42)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample, x='duration', y='price', hue='class', alpha=0.35, s=14)
plt.title('Flight Duration vs Ticket Price (Sample)')
plt.xlabel('Duration (Hours)')
plt.ylabel('Price (INR)')
plt.tight_layout()
save_graph('09_duration_vs_price.png')

# 10. Numeric correlation (exclude artificial index)
numeric = df.select_dtypes(include=np.number).drop(columns=['index'], errors='ignore')
plt.figure(figsize=(9, 7))
sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation of Numerical Variables')
plt.tight_layout()
save_graph('10_numeric_correlations.png')

# Specific route example from original project
route = df.loc[(df['airline'] == 'Vistara') & (df['source_city'] == 'Delhi') &
               (df['destination_city'] == 'Hyderabad') & (df['class'] == 'Business')]
print('\nVISTARA DELHI TO HYDERABAD (BUSINESS)')
print('Matching records:', len(route))
if not route.empty:
    print('Mean price (INR):', round(route['price'].mean(), 2))
    print('Min/max price (INR):', route['price'].min(), route['price'].max())

summary = f'''AIRLINE FLIGHT DATA ANALYSIS — SUMMARY
======================================
Records: {len(df):,}
Columns: {len(df.columns)}
Airlines: {df['airline'].nunique()}
Missing cells: {int(df.isna().sum().sum()):,}
Average ticket price (INR): {df['price'].mean():,.2f}
Median ticket price (INR): {df['price'].median():,.2f}
Minimum ticket price (INR): {df['price'].min():,}
Maximum ticket price (INR): {df['price'].max():,}
Most represented airline: {df['airline'].value_counts().idxmax()}
Most represented source city: {df['source_city'].value_counts().idxmax()}
Most represented destination city: {df['destination_city'].value_counts().idxmax()}
Note: Counts describe this dataset, not real-world market shares.
Charts saved: 10
'''
(BASE_DIR / 'analysis_summary.txt').write_text(summary, encoding='utf-8')
print('\n' + summary)
print('Done! Open the images folder to view all graphs.')
