"""
Example of sampling from a fitted GMM
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn_gmm import GaussianMixture

# Set random seed
np.random.seed(42)

# Generate some training data
print("Generating training data...")
centers = np.array([[0, 0], [3, 3], [-1, 2]])
X_train = np.vstack([
    np.random.multivariate_normal(centers[0], [[1, 0.5], [0.5, 1]], 100),
    np.random.multivariate_normal(centers[1], [[0.8, -0.2], [-0.2, 0.8]], 100),
    np.random.multivariate_normal(centers[2], [[1, 0.1], [0.1, 1]], 100),
])

# Fit the model
print("\nFitting GMM...")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_train)

print(f"Weights: {gmm.weights_}")
print(f"Means:\n{gmm.means_}")

# Generate samples from the model
print("\nSampling from the model...")
n_samples = 200
samples, component_labels = gmm.sample(n_samples)

print(f"Generated {len(samples)} samples")
print(f"Component distribution: {np.bincount(component_labels)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training data
for k in range(3):
    mask = gmm.predict(X_train) == k
    axes[0].scatter(X_train[mask, 0], X_train[mask, 1], alpha=0.4, label=f'Component {k}')
axes[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', 
               s=200, linewidths=3, label='Means')
axes[0].set_title('Training Data')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Generated samples
for k in range(3):
    mask = component_labels == k
    axes[1].scatter(samples[mask, 0], samples[mask, 1], alpha=0.5, label=f'Component {k}')
axes[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', 
               s=200, linewidths=3, label='Means')
axes[1].set_title(f'Generated Samples (n={n_samples})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sampling_result.png', dpi=150)
print("\nVisualization saved as 'sampling_result.png'")
plt.show()

