# Installation Guide

## Installing from Source

1. **Clone or download the repository**

```bash
cd sklearn_gmm
```

2. **Install in development mode**

```bash
pip install -e .
```

Or for a regular installation:

```bash
pip install .
```

## Installing Required Dependencies

Install the required dependencies:

```bash
pip install numpy>=1.21.0
```

Or install with additional features:

```bash
# With dev dependencies
pip install -e ".[dev]"

# With example dependencies
pip install -e ".[examples]"

# Both
pip install -e ".[dev,examples]"
```

## Verifying Installation

Run this simple test to verify the installation:

```python
import numpy as np
from sklearn_gmm import GaussianMixture

# Create some data
X = np.random.randn(100, 2)

# Create and fit the model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Check it works
labels = gmm.predict(X)
print(f"Fitted GMM with {gmm.n_components} components")
print(f"Predictions shape: {labels.shape}")
print("Installation successful!")
```

## Running Tests

After installation, you can run the test suite:

```bash
pytest sklearn_gmm/tests/
```

## Building for Distribution

To build source and wheel distributions:

```bash
python setup.py sdist bdist_wheel
```

To build with modern build system:

```bash
pip install build
python -m build
```

The distributions will be in the `dist/` directory.

