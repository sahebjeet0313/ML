import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load Iris dataset and select two features for clustering and plotting
iris = load_iris()
X = iris.data[:, :2]  # Using sepal length and sepal width
y_true = iris.target

# Initialize and fit KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the clustered data points
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("K-Means Clustering on Iris Dataset (Sepal length vs Sepal width)")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()
plt.show()

# Manual input for prediction
print("Enter the features for cluster prediction:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))

manual_input = np.array([[sepal_length, sepal_width]])
manual_pred = kmeans.predict(manual_input)
print(f"Predicted cluster: {manual_pred[0]}")

# Plotting the dataset and manual input point
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.scatter(manual_input[0, 0], manual_input[0, 1], color='black', marker='X', s=100, label='Manual Input')
plt.title("K-Means Clustering on Iris Dataset with Manual Input")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()
plt.show()
