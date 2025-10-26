# Quick Start Guide

## Installation

```bash
pip install sklearn-gmm
# or from source:
pip install -e .
```

## Basic Usage

### 1. Simple Clustering

```python
from sklearn_gmm import GaussianMixture
import numpy as np

# Generate or load your data
X = np.random.randn(100, 2)

# Create and fit the model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Get cluster assignments
labels = gmm.predict(X)

# Get probabilities
probabilities = gmm.predict_proba(X)

# Evaluate
score = gmm.score(X)  # Average log-likelihood
```

### 2. Model Selection

```python
# Find optimal number of components
n_components_range = range(1, 10)
bics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    bics.append(gmm.bic(X))

best_n = n_components_range[np.argmin(bics)]
print(f"Optimal components: {best_n}")
```

### 3. Different Covariance Types

```python
# Full covariance (default) - most flexible
gmm_full = GaussianMixture(n_components=3, covariance_type='full')

# Diagonal covariance - faster, less parameters
gmm_diag = GaussianMixture(n_components=3, covariance_type='diag')

# Spherical covariance - equal variances in all directions
gmm_spherical = GaussianMixture(n_components=3, covariance_type='spherical')

# Tied covariance - shared across all components
gmm_tied = GaussianMixture(n_components=3, covariance_type='tied')
```

### 4. Sampling from the Model

```python
# After fitting
gmm.fit(X)

# Generate new samples
samples, component_labels = gmm.sample(n_samples=100)

# samples.shape: (100, n_features)
# component_labels.shape: (100,)
```

### 5. Using with sklearn Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gmm', GaussianMixture(n_components=3))
])

pipeline.fit(X)
labels = pipeline.named_steps['gmm'].predict(X)
```

### 6. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_components': [2, 3, 4, 5],
    'covariance_type': ['full', 'diag'],
}

gmm = GaussianMixture(random_state=42)
grid = GridSearchCV(gmm, param_grid, cv=5)
grid.fit(X)

print(f"Best parameters: {grid.best_params_}")
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 1 | Number of mixture components |
| `covariance_type` | 'full' | Type of covariance ('full', 'diag', 'spherical', 'tied') |
| `tol` | 1e-3 | Convergence threshold |
| `max_iter` | 100 | Maximum EM iterations |
| `n_init` | 1 | Number of random initializations |
| `init_params` | 'kmeans++' | Initialization method ('kmeans++' or 'random') |
| `reg_covar` | 1e-6 | Regularization for covariance matrices |
| `random_state` | None | Random seed |

## Main Methods

- `fit(X)`: Fit the model to data
- `predict(X)`: Predict component assignments
- `predict_proba(X)`: Predict posterior probabilities
- `score(X)`: Compute average log-likelihood
- `aic(X)`: Akaike Information Criterion
- `bic(X)`: Bayesian Information Criterion
- `sample(n_samples)`: Generate random samples

## Common Tasks

### Density Estimation
```python
# Get likelihood for new data
log_prob = gmm.score_samples(new_data)
prob = np.exp(log_prob)  # Convert to probability
```

### Anomaly Detection
```python
# Use low likelihood as indicator
gmm.fit(normal_data)
log_probs = gmm.score_samples(test_data)
threshold = np.percentile(log_probs, 5)
anomalies = test_data[log_probs < threshold]
```

### Data Compression
```python
# Represent data by component assignments
gmm.fit(X)
labels = gmm.predict(X)
# Now you can store only labels + model parameters
```

## Examples

See the `examples/` directory for complete working examples:
- `basic_usage.py`: Basic clustering
- `model_selection.py`: Finding optimal components
- `covariance_types.py`: Comparing covariance types
- `sampling.py`: Generating new samples

