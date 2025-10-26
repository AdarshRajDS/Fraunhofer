"""
Basic usage example for sklearn_gmm.GaussianMixture
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn_gmm import GaussianMixture

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
print("Generating sample data...")
n_samples = 300
centers = np.array([[0, 0], [5, 5], [0, 5]])
X = np.vstack([
    np.random.multivariate_normal(centers[0], [[1, 0.5], [0.5, 1]], n_samples),
    np.random.multivariate_normal(centers[1], [[1, -0.3], [-0.3, 1]], n_samples),
    np.random.multivariate_normal(centers[2], [[0.8, 0.2], [0.2, 0.8]], n_samples),
])

# Create and fit the model
print("Fitting GMM with 3 components...")
gmm = GaussianMixture(n_components=3, random_state=42, max_iter=100)
gmm.fit(X)

print(f"\nConverged: {gmm.converged_}")
print(f"Iterations: {gmm.n_iter_}")
print(f"Log-likelihood: {gmm.score(X):.4f}")
print(f"Components weights: {gmm.weights_}")
print(f"\nMeans:\n{gmm.means_}")

# Make predictions
labels = gmm.predict(X)
proba = gmm.predict_proba(X)

print(f"\nPredictions shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Evaluate
aic = gmm.aic(X)
bic = gmm.bic(X)
print(f"\nAIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: True clusters (if we knew them)
for i, center in enumerate(centers):
    mask = labels == i
    axes[0].scatter(X[mask, 0], X[mask, 1], label=f'Component {i}', alpha=0.6)
axes[0].scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='True centers')
axes[0].set_title('Predicted Clusters')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: GMM means
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.3)
axes[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=200, linewidths=3, label='GMM centers')
axes[1].set_title('Fitted GMM')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('basic_usage_result.png', dpi=150)
print("\nVisualization saved as 'basic_usage_result.png'")
plt.show()

