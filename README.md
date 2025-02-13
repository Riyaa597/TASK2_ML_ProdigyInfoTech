import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

np.random.seed(42)
data = {
    'customer_id': np.arange(1, 101),
    'total_spent': np.random.randint(100, 1000, 100), 
    'frequency_of_purchases': np.random.randint(1, 20, 100), 
    'product_category': np.random.randint(1, 10, 100)  
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

X = df[['total_spent', 'frequency_of_purchases', 'product_category']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("Clustered Customers:")
print(df.head())

sil_score = silhouette_score(X_scaled, df['cluster'])
print(f"Silhouette Score: {sil_score}")

plt.figure(figsize=(8, 6))
plt.scatter(df['total_spent'], df['frequency_of_purchases'], c=df['cluster'], cmap='viridis')
plt.title('Customer Segmentation Using K-Means')
plt.xlabel('Total Spent')
plt.ylabel('Frequency of Purchases')
plt.colorbar(label='Cluster')
plt.show()
