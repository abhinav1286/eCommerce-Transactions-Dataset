import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge customer, product, and transaction data
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')

# Check for missing values
missing_values = merged_df.isnull().sum()
print("Missing values in the merged data:")
print(missing_values)

# Feature engineering: creating aggregate features like total spend and product preferences
customer_data = merged_df.groupby('CustomerID').agg(
    TotalSpend=('TotalValue', 'sum'),
    ProductCount=('ProductID', 'nunique'),
    MostBoughtCategory=('Category', lambda x: x.mode()[0])
).reset_index()

# Feature selection (using total spend and product count for simplicity)
X = customer_data[['TotalSpend', 'ProductCount']]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute similarity matrix
similarity_matrix = cosine_similarity(X_scaled)

# Display customer similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=customer_data['CustomerID'], columns=customer_data['CustomerID'])
print("\nCustomer Similarity Matrix:")
print(similarity_df.head())

# Prepare Lookalike Recommendations (Top 3 similar customers for each of the first 20 customers)
lookalike_list = []

for customer_id in customer_data['CustomerID'][:20]:  # First 20 customers (C0001 - C0020)
    similarity_scores = similarity_df[customer_id].drop(customer_id)
    top_3_similar_customers = similarity_scores.nlargest(3)  # Get top 3 most similar customers
    
    for similar_customer, score in zip(top_3_similar_customers.index, top_3_similar_customers):
        lookalike_list.append([customer_id, similar_customer, score])

# Convert the lookalike_list into a DataFrame and save to CSV
lookalike_df = pd.DataFrame(lookalike_list, columns=['CustomerID', 'Lookalike_CustomerID', 'Similarity_Score'])
lookalike_df.to_csv('Lookalike.csv', index=False)

print("\nLookalike recommendations saved to Lookalike.csv")
