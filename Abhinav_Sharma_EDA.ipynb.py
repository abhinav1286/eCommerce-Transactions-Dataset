import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')

# Check for missing values
missing_values = merged_df.isnull().sum()
print("Missing values in the merged data:")
print(missing_values)

# Convert TransactionDate and SignupDate to datetime
merged_df['TransactionDate'] = pd.to_datetime(merged_df['TransactionDate'])
merged_df['SignupDate'] = pd.to_datetime(merged_df['SignupDate'])

# Calculate 'DaysSinceSignup' and filter transactions after signup
merged_df['DaysSinceSignup'] = (merged_df['TransactionDate'] - merged_df['SignupDate']).dt.days
filtered_df = merged_df[merged_df['DaysSinceSignup'] >= 0]

# Calculate total spend per customer
total_spend_per_customer = filtered_df.groupby('CustomerID')['TotalValue'].sum().reset_index()
total_spend_per_customer.rename(columns={'TotalValue': 'TotalSpend'}, inplace=True)

# Merge total spend back into filtered data
filtered_df = filtered_df.merge(total_spend_per_customer, on='CustomerID', how='left')

# Add 'HighValueCustomer' column
average_spend = filtered_df['TotalSpend'].mean()
filtered_df['HighValueCustomer'] = filtered_df['TotalSpend'] > average_spend

# Add 'FirstPurchase' feature (whether it's the first purchase)
filtered_df['FirstPurchase'] = filtered_df['DaysSinceSignup'] > 0

# Insights generation
print("\nINSIGHTS:")

# Insight 1: Top 5 High-Value Customers
high_value_customers = filtered_df[filtered_df['HighValueCustomer']].groupby('CustomerID')['TotalSpend'].sum().sort_values(ascending=False).head(5)
print("\n1. Top 5 High-Value Customers:")
print(high_value_customers)

# Insight 2: Revenue by Region
region_revenue = filtered_df.groupby('Region')['TotalValue'].sum().reset_index().sort_values(by='TotalValue', ascending=False)
print("\n2. Revenue by Region:")
print(region_revenue)

# Insight 3: Revenue by Product Category
category_revenue = filtered_df.groupby('Category')['TotalValue'].sum().reset_index().sort_values(by='TotalValue', ascending=False)
print("\n3. Revenue by Product Category:")
print(category_revenue)

# Insight 4: Average Spending by Signup Date
signup_trends = filtered_df.groupby('SignupDate')['TotalSpend'].mean().reset_index().sort_values(by='SignupDate')
print("\n4. Average Spending by Signup Date:")
print(signup_trends.head())

# Insight 5: Spending Trend by Days Since Signup
days_spend_trend = filtered_df.groupby('DaysSinceSignup')['TotalValue'].mean().reset_index()
print("\n5. Spending Trend by Days Since Signup:")
print(days_spend_trend.head())

# Feature selection for predictive modeling
feature_columns = ['DaysSinceSignup', 'FirstPurchase', 'HighValueCustomer']
X = filtered_df[feature_columns]
y = filtered_df['HighValueCustomer']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nMODEL EVALUATION:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Prepare data for clustering
clustering_features = ['DaysSinceSignup', 'TotalSpend']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[clustering_features])

# Display scaled features
print("\nScaled Features (for clustering):")
print(X_scaled[:5])

# Calculate the similarity between customers based on their spend and signup days
similarity_matrix = cosine_similarity(X_scaled)

# Convert similarity matrix into a DataFrame for easier interpretation
similarity_df = pd.DataFrame(similarity_matrix, index=filtered_df['CustomerID'], columns=filtered_df['CustomerID'])

# Print the similarity matrix (optional for debugging)
print("\nCustomer Similarity Matrix:")
print(similarity_df.head())
