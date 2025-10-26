"""
Comparing different covariance types
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn_gmm import GaussianMixture

# Set random seed
np.random.seed(42)

# Generate sample data with correlation
print("Generating sample data with correlated clusters...")
centers = np.array([[0, 0], [4, 3]])
X = np.vstack([
    np.random.multivariate_normal(centers[0], [[2, 1], [1, 1]], 150),
    np.random.multivariate_normal(centers[1], [[1.5, -0.8], [-0.8, 1.5]], 150),
])

# Fit models with different covariance types
covariance_types = ['full', 'diag', 'spherical', 'tied']

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, cov_type in enumerate(covariance_types):
    print(f"\nFitting with covariance_type='{cov_type}'...")
    gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=42)
    gmm.fit(X)
    
    labels = gmm.predict(X)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    
    print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
    print(f"  Weights: {gmm.weights_}")
    
    # Plot
    ax = axes[idx]
    for k in range(2):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], label=f'Component {k}', alpha=0.6)
    
    ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', 
               s=200, linewidths=3, label='Means')
    ax.set_title(f'covariance_type="{cov_type}"\nAIC={aic:.0f}, BIC={bic:.0f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('covariance_types_result.png', dpi=150)
print("\nVisualization saved as 'covariance_types_result.png'")
plt.show()

