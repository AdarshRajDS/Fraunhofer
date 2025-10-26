"""
Model selection example: finding the optimal number of components
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn_gmm import GaussianMixture

# Set random seed
np.random.seed(42)

# Generate sample data
print("Generating sample data...")
centers = np.array([[0, 0], [3, 3], [-2, 2], [4, 0]])
X = np.vstack([
    np.random.multivariate_normal(centers[0], [[1, 0.5], [0.5, 1]], 200),
    np.random.multivariate_normal(centers[1], [[0.8, -0.2], [-0.2, 0.8]], 150),
    np.random.multivariate_normal(centers[2], [[1, 0.1], [0.1, 1]], 180),
    np.random.multivariate_normal(centers[3], [[0.6, 0.3], [0.3, 0.6]], 170),
])

# Test different numbers of components
n_components_range = range(1, 11)
aics = []
bics = []
log_likelihoods = []

print("\nTesting different numbers of components...")
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
    gmm.fit(X)
    aics.append(gmm.aic(X))
    bics.append(gmm.bic(X))
    log_likelihoods.append(gmm.score(X))
    print(f"  n_components={n:2d}: AIC={aics[-1]:8.2f}, BIC={bics[-1]:8.2f}, LL={log_likelihoods[-1]:6.2f}")

# Find best model
best_aic_n = n_components_range[np.argmin(aics)]
best_bic_n = n_components_range[np.argmin(bics)]

print(f"\nBest according to AIC: {best_aic_n} components")
print(f"Best according to BIC: {best_bic_n} components")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(n_components_range, aics, 'o-', label='AIC')
axes[0].plot(n_components_range, bics, 's-', label='BIC')
axes[0].axvline(best_aic_n, color='red', linestyle='--', alpha=0.5, label=f'Best AIC (n={best_aic_n})')
axes[0].axvline(best_bic_n, color='blue', linestyle='--', alpha=0.5, label=f'Best BIC (n={best_bic_n})')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Information Criterion')
axes[0].set_title('Model Selection: AIC and BIC')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_components_range, log_likelihoods, 'o-', color='green')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Log-Likelihood')
axes[1].set_title('Log-Likelihood vs Number of Components')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_selection_result.png', dpi=150)
print("\nVisualization saved as 'model_selection_result.png'")
plt.show()

