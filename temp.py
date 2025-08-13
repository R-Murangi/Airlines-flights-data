#Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

#Load the Dataset
df = pd.read_csv('/kaggle/input/airlines-flights-data/airlines_flights_data.csv')
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

#Explore the Data
print("First 5 rows:")
print(df.head())
print("\nDataset Description:")
print(df.describe(include='all'))

# Check the Data Quality
print("Missing values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

#Data Overview
print("Unique values in categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    print(f"Sample: {df[col].unique()[:5]}")
    print("-" * 30)
    
#Airlines and Frequencies
airline_freq = df['airline'].value_counts()
print("Airlines and frequencies:")
print(airline_freq)

plt.figure(figsize=(10, 8))
plt.pie(airline_freq.values, labels=airline_freq.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Airlines', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.show()

#Departure and Arrival Time Bar Graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

departure_counts = df['departure_time'].value_counts().sort_index()
ax1.bar(departure_counts.index, departure_counts.values, color='skyblue', alpha=0.7)
ax1.set_title('Departure Times Distribution')
ax1.set_xlabel('Departure Time')
ax1.set_ylabel('Number of Flights')
ax1.tick_params(axis='x', rotation=45)

arrival_counts = df['arrival_time'].value_counts().sort_index()
ax2.bar(arrival_counts.index, arrival_counts.values, color='lightcoral', alpha=0.7)
ax2.set_title('Arrival Times Distribution')
ax2.set_xlabel('Arrival Time')
ax2.set_ylabel('Number of Flights')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#Source and Destination City Bar Graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

source_counts = df['source_city'].value_counts()
ax1.bar(source_counts.index, source_counts.values, color='lightgreen', alpha=0.8)
ax1.set_title('Source Cities Distribution')
ax1.set_xlabel('Source City')
ax1.set_ylabel('Number of Flights')
ax1.tick_params(axis='x', rotation=45)

dest_counts = df['destination_city'].value_counts()
ax2.bar(dest_counts.index, dest_counts.values, color='gold', alpha=0.8)
ax2.set_title('Destination Cities Distribution')
ax2.set_xlabel('Destination City')
ax2.set_ylabel('Number of Flights')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#Price Variation with Airlines
price_by_airline = df.groupby('airline')['price'].agg(['mean', 'median', 'std', 'min', 'max'])
print("Price statistics by airline:")
print(price_by_airline.round(2))

plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='airline', y='price', palette='Set2')
plt.title('Price Variation Across Airlines', fontsize=16)
plt.xlabel('Airline')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price vs Departure and Arrival Time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

df.groupby('departure_time')['price'].mean().plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Average Price by Departure Time')
ax1.set_xlabel('Departure Time')
ax1.set_ylabel('Average Price')
ax1.tick_params(axis='x', rotation=45)

df.groupby('arrival_time')['price'].mean().plot(kind='bar', ax=ax2, color='darkorange')
ax2.set_title('Average Price by Arrival Time')
ax2.set_xlabel('Arrival Time')
ax2.set_ylabel('Average Price')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Average price by departure time:")
print(df.groupby('departure_time')['price'].mean().round(2))

# Price Changes with Source and Destination
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
source_price = df.groupby('source_city')['price'].mean().sort_values(ascending=False)
plt.bar(source_price.index, source_price.values, color='lightblue', alpha=0.8)
plt.title('Average Price by Source City')
plt.xlabel('Source City')
plt.ylabel('Average Price')
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
dest_price = df.groupby('destination_city')['price'].mean().sort_values(ascending=False)
plt.bar(dest_price.index, dest_price.values, color='lightcoral', alpha=0.8)
plt.title('Average Price by Destination City')
plt.xlabel('Destination City')
plt.ylabel('Average Price')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("Average price by source city:")
print(source_price.round(2))

# Price Effect for Last-Minute Bookings
if 'days_left' in df.columns:
    last_minute = df[df['days_left'].isin([1, 2])]
    regular = df[df['days_left'] > 2]
    
    print(f"Last-minute avg price: {last_minute['price'].mean():.2f}")
    print(f"Regular booking avg price: {regular['price'].mean():.2f}")
    
    plt.figure(figsize=(10, 6))
    categories = ['Last Minute\n(1-2 days)', 'Regular\n(>2 days)']
    prices = [last_minute['price'].mean(), regular['price'].mean()]
    
    plt.bar(categories, prices, color=['red', 'green'], alpha=0.7)
    plt.title('Price: Last Minute vs Regular Booking')
    plt.ylabel('Average Price')
    plt.show()
else:
    print("'days_left' column not found. Available columns:")
    print(list(df.columns))
    
# Price Variation by Class
if 'class' in df.columns:
    class_price = df.groupby('class')['price'].agg(['mean', 'median', 'std', 'count'])
    print("Price statistics by class:")
    print(class_price.round(2))
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='class', y='price', palette='viridis')
    plt.title('Price Distribution by Class')
    
    plt.subplot(1, 2, 2)
    avg_prices = df.groupby('class')['price'].mean()
    plt.bar(avg_prices.index, avg_prices.values, color=['skyblue', 'gold'], alpha=0.8)
    plt.title('Average Price by Class')
    plt.ylabel('Average Price')
    
    plt.tight_layout()
    plt.show()
else:
    print("'class' column not found. Available columns:")
    print(list(df.columns))
    
#Vistara Delhi-Hyderabad Business Class Price
vistara_del_hyd = df[
    (df['airline'].str.contains('Vistara', case=False, na=False)) &
    (df['source_city'].str.contains('Delhi', case=False, na=False)) &
    (df['destination_city'].str.contains('Hyderabad', case=False, na=False))
]

if 'class' in df.columns:
    vistara_del_hyd_business = vistara_del_hyd[
        vistara_del_hyd['class'].str.contains('Business', case=False, na=False)
    ]
else:
    vistara_del_hyd_business = vistara_del_hyd

print(f"Matching flights found: {len(vistara_del_hyd_business)}")

if len(vistara_del_hyd_business) > 0:
    avg_price = vistara_del_hyd_business['price'].mean()
    print(f"Average Price: ₹{avg_price:.2f}")
    print(f"Price range: ₹{vistara_del_hyd_business['price'].min():.2f} - ₹{vistara_del_hyd_business['price'].max():.2f}")
else:
    print("No matching flights found.")
    print("Available Vistara routes:")
    vistara_flights = df[df['airline'].str.contains('Vistara', case=False, na=False)]
    if len(vistara_flights) > 0:
        print(vistara_flights.groupby(['source_city', 'destination_city']).size().head())
        
#Correlation Analysis
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))

# CELL 16 - Summary Insights
print("="*60)
print("SUMMARY OF KEY INSIGHTS")
print("="*60)
print(f"Total flights: {len(df):,}")
print(f"Airlines: {df['airline'].nunique()}")
print(f"Average price: ₹{df['price'].mean():.2f}")
print(f"Price range: ₹{df['price'].min():.2f} - ₹{df['price'].max():.2f}")
print(f"Most popular airline: {df['airline'].mode()[0]}")
print(f"Most popular source: {df['source_city'].mode()[0]}")
print(f"Most popular destination: {df['destination_city'].mode()[0]}")
print("Analysis Complete! ✈️")