# Sklearn GMM - A Scikit-Learn Compatible Gaussian Mixture Model

A fully sklearn-compatible implementation of Gaussian Mixture Models (GMM) using the Expectation-Maximization (EM) algorithm. This package provides a production-ready implementation with support for multiple covariance types and sklearn-standard API.

## Features

- **Sklearn-compatible API**: Works seamlessly with sklearn pipelines, cross-validation, and model selection
- **Multiple covariance types**: Support for full, tied, diagonal, and spherical covariances
- **Robust initialization**: K-means++ and random initialization strategies
- **Production-ready**: Numerical stability, proper regularization, and comprehensive error handling
- **Complete sklearn interface**: Implements fit, predict, predict_proba, score, sample, AIC/BIC, and more
- **Well-documented**: Comprehensive docstrings following sklearn conventions

## Installation

```bash
pip install sklearn-gmm
```

Or install from source:

```bash
git clone https://github.com/yourusername/sklearn-gmm.git
cd sklearn-gmm
pip install -e .
```

## Quick Start

```python
import numpy as np
from sklearn_gmm import GaussianMixture

# Generate some sample data
X = np.random.randn(100, 2)

# Create and fit the model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Make predictions
labels = gmm.predict(X)
proba = gmm.predict_proba(X)

# Evaluate the model
log_likelihood = gmm.score(X)
aic = gmm.aic(X)
bic = gmm.bic(X)

# Sample from the model
new_samples, component_labels = gmm.sample(n_samples=10)
```

## Usage Examples

### Basic Classification

```python
import numpy as np
from sklearn_gmm import GaussianMixture

# Fit a 3-component GMM
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Predict component assignments
labels = gmm.predict(X)
```

### Model Selection with AIC/BIC

```python
n_components_range = range(1, 10)
bics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X)
    bics.append(gmm.bic(X))

best_n = n_components_range[np.argmin(bics)]
print(f"Best number of components: {best_n}")
```

### Using Different Covariance Types

```python
# Full covariance (default)
gmm_full = GaussianMixture(n_components=3, covariance_type='full')

# Diagonal covariance
gmm_diag = GaussianMixture(n_components=3, covariance_type='diag')

# Spherical covariance
gmm_spherical = GaussianMixture(n_components=3, covariance_type='spherical')

# Tied covariance (shared across components)
gmm_tied = GaussianMixture(n_components=3, covariance_type='tied')
```

### Using with sklearn Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_gmm import GaussianMixture

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gmm', GaussianMixture(n_components=3))
])

pipeline.fit(X)
labels = pipeline.named_steps['gmm'].predict(X)
```

## API Reference

### GaussianMixture

```python
GaussianMixture(
    n_components=1,
    covariance_type='full',
    tol=1e-3,
    max_iter=100,
    n_init=1,
    init_params='kmeans++',
    reg_covar=1e-6,
    random_state=None
)
```

#### Parameters

- **n_components** (int, default=1): The number of mixture components
- **covariance_type** ({'full', 'tied', 'diag', 'spherical'}, default='full'): Type of covariance parameters
- **tol** (float, default=1e-3): Convergence threshold for EM algorithm
- **max_iter** (int, default=100): Maximum number of EM iterations
- **n_init** (int, default=1): Number of random initializations
- **init_params** ({'kmeans++', 'random'}, default='kmeans++'): Initialization method
- **reg_covar** (float, default=1e-6): Regularization added to diagonal of covariance
- **random_state** (int or None, default=None): Random seed for reproducibility

#### Methods

- **fit(X)**: Fit the GMM to data X
- **predict(X)**: Predict component labels for samples in X
- **predict_proba(X)**: Predict posterior probabilities for each component
- **score(X)**: Compute average log-likelihood
- **score_samples(X)**: Compute log-likelihood for each sample
- **aic(X)**: Compute Akaike Information Criterion
- **bic(X)**: Compute Bayesian Information Criterion
- **sample(n_samples)**: Generate random samples from the model
- **get_params()**: Get parameters for this estimator
- **set_params()**: Set parameters for this estimator

#### Attributes

- **weights_**: Weights of each mixture component (shape: n_components,)
- **means_**: Mean of each mixture component (shape: n_components, n_features)
- **covariances_**: Covariances (shape depends on covariance_type)
- **converged_**: Whether convergence was reached
- **n_iter_**: Number of EM iterations
- **lower_bound_**: Lower bound of log-likelihood

## Algorithm Details

This implementation uses the Expectation-Maximization (EM) algorithm as described in standard machine learning textbooks:

1. **E-step**: Compute responsibilities (posterior probabilities) using current parameters
2. **M-step**: Update parameters (means, covariances, weights) using responsibilities
3. **Convergence check**: Stop when the change in log-likelihood is below `tol`

The implementation includes:
- Numerically stable log-sum-exp computations
- Cholesky-based precision calculations for numerical stability
- Regularization to ensure positive-definite covariance matrices
- Support for multiple covariance structures

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this implementation in your research, please cite it as:

```bibtex
@software{sklearn-gmm,
  author = {Your Name},
  title = {sklearn-gmm: A Scikit-Learn Compatible Gaussian Mixture Model},
  year = {2024},
  url = {https://github.com/yourusername/sklearn-gmm}
}
```

## Acknowledgments

This implementation follows the standard EM algorithm for Gaussian Mixture Models and is designed to be compatible with scikit-learn's API conventions.

