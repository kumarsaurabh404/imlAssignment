import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the data
data = pd.read_csv('your_data.csv')

# Step 2: Preprocess the data
# Remove any unnecessary columns
data = data.drop(columns=['id'])

# Perform standard scaling on the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: EM Algorithm
# Initialize the Gaussian Mixture model with the desired number of clusters
em_model = GaussianMixture(n_components=3, random_state=0)

# Fit the model to the scaled data
em_model.fit(scaled_data)

# Get the predicted cluster labels
em_labels = em_model.predict(scaled_data)

# Step 4: K-means Algorithm
# Initialize the K-means model with the desired number of clusters
kmeans_model = KMeans(n_clusters=3, random_state=0)

# Fit the model to the scaled data
kmeans_model.fit(scaled_data)

# Get the predicted cluster labels
kmeans_labels = kmeans_model.predict(scaled_data)

# Step 5: Compare the results
# Evaluate the EM algorithm using silhouette score
em_silhouette = silhouette_score(scaled_data, em_labels)

# Evaluate the K-means algorithm using silhouette score
kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)

# Print the silhouette scores
print("EM Algorithm Silhouette Score:", em_silhouette)
print("K-means Algorithm Silhouette Score:", kmeans_silhouette)
