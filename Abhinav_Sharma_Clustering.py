import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers_df = pd.read_csv("Customers.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Merge datasets based on CustomerID
merged_df = pd.merge(customers_df, transactions_df, on="CustomerID", how="inner")

# Check for missing values and fill them
merged_df = merged_df.fillna(0)

# Feature Engineering
customer_data = merged_df.groupby("CustomerID").agg({
    "TotalValue": "sum",
    "Quantity": "sum",
    "Region": "first"  # Assuming 'Region' is categorical
}).reset_index()

# Normalize the features
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[["TotalValue", "Quantity"]])

# Elbow Method for optimal number of clusters
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customer_data_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 11), inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Perform KMeans clustering with chosen k (e.g., k=3)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(customer_data_scaled, customer_data['Cluster'])
print(f'Davies-Bouldin Index: {db_index}')

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(customer_data_scaled)

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=customer_data['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title(f'Customer Segmentation (k={k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Save results to CSV
customer_data.to_csv("Customer_Segmentation.csv", index=False)
# Cluster Summary
cluster_summary = customer_data.groupby('Cluster').agg({
    'TotalValue': 'mean',
    'Quantity': 'mean'
}).reset_index()

# Print the summary to the console
print("\nCluster Summary (Average Total Spend and Quantity for each Cluster):")
print(cluster_summary)
