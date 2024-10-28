import cv2
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model (excluding the final classification layers)
model = VGG16(weights='imagenet', include_top=False)

# Set directory path to your images
image_directory = "C:\\Users\\naren\\Downloads\\archive (3)\\Mars Surface and Curiosity Image\\sharpened images"

# Feature extraction function
def extract_features(image_path, model):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# Prepare dataset
image_paths = [os.path.join(image_directory, img) for img in os.listdir(image_directory)]
features = [extract_features(path, model) for path in image_paths]
features = np.array(features)

# Optional: Reduce dimensionality using PCA
pca = PCA(n_components=2)  # 2D for visualization
features_pca = pca.fit_transform(features)

# Split dataset into 50% train and 50% test sets
X_train, X_test = train_test_split(features_pca, test_size=0.5, random_state=42)

# Number of clusters
n_clusters = 25

# Step 1: GMM Clustering on the first half (X_train)
gmm_train = GaussianMixture(n_components=n_clusters, random_state=42)
train_labels = gmm_train.fit_predict(X_train)
silhouette_train = silhouette_score(X_train, train_labels)
print(f"Silhouette Score for First 50% of Data (GMM): {silhouette_train}")

# Step 2: GMM Clustering on the full dataset
gmm_full = GaussianMixture(n_components=n_clusters, random_state=42)
full_labels = gmm_full.fit_predict(features_pca)
silhouette_full = silhouette_score(features_pca, full_labels)
print(f"Silhouette Score for Full Dataset (GMM): {silhouette_full}")

# Visualization - Original Data, 50% Dataset Clustering, Full Dataset Clustering
plt.figure(figsize=(20, 5))

# Plot for Original Data (Unclustered)
plt.subplot(1, 3, 1)
plt.scatter(features_pca[:, 0], features_pca[:, 1], s=20, c='gray')
plt.title("Original Data (Unclustered)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()

# Plot for 50% Dataset Clustering with GMM
plt.subplot(1, 3, 2)
for i in range(n_clusters):
    cluster_points = X_train[train_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20, label=f"Cluster {i}")
plt.title(f'50% Dataset Clustering (GMM, Silhouette: {silhouette_train:.2f})')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()

# Plot for Full Dataset Clustering with GMM
plt.subplot(1, 3, 3)
for i in range(n_clusters):
    cluster_points = features_pca[full_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=20, label=f"Cluster {i}")
plt.title(f'Full Dataset Clustering (GMM, Silhouette: {silhouette_full:.2f})')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()

plt.tight_layout()
plt.show()
