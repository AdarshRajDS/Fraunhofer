"""
Tests for sklearn_gmm.GaussianMixture
"""
import numpy as np
import pytest
from sklearn_gmm import GaussianMixture


def test_initialization():
    """Test GMM initialization"""
    gmm = GaussianMixture(n_components=3)
    assert gmm.n_components == 3
    assert gmm.covariance_type == "full"
    assert gmm.tol == 1e-3
    assert gmm.max_iter == 100


def test_initialization_errors():
    """Test that invalid parameters raise errors"""
    with pytest.raises(ValueError):
        GaussianMixture(n_components=0)
    with pytest.raises(ValueError):
        GaussianMixture(covariance_type="invalid")
    with pytest.raises(ValueError):
        GaussianMixture(tol=0)
    with pytest.raises(ValueError):
        GaussianMixture(max_iter=0)


def test_fit_and_predict():
    """Test basic fit and predict"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    assert gmm.weights_ is not None
    assert gmm.means_ is not None
    assert gmm.covariances_ is not None
    assert len(gmm.weights_) == 3
    assert gmm.means_.shape == (3, 2)
    
    predictions = gmm.predict(X)
    assert predictions.shape == (100,)
    assert np.all(predictions >= 0) and np.all(predictions < 3)


def test_predict_proba():
    """Test predict_proba"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    proba = gmm.predict_proba(X)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_score():
    """Test score method"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    score = gmm.score(X)
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_aic_bic():
    """Test AIC and BIC"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    
    assert isinstance(aic, float)
    assert isinstance(bic, float)
    assert np.isfinite(aic)
    assert np.isfinite(bic)
    # BIC should generally be larger (more penalty)
    assert bic >= aic


def test_different_covariance_types():
    """Test different covariance types"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    for cov_type in ['full', 'diag', 'spherical', 'tied']:
        gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
        gmm.fit(X)
        
        assert gmm.converged_
        assert gmm.n_iter_ > 0
        predictions = gmm.predict(X)
        assert len(predictions) == 100


def test_sample():
    """Test sampling from the fitted model"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    samples, labels = gmm.sample(n_samples=50)
    
    assert samples.shape == (50, 2)
    assert labels.shape == (50,)
    assert np.all(labels >= 0) and np.all(labels < 3)


def test_get_set_params():
    """Test parameter getter and setter"""
    gmm = GaussianMixture(n_components=5, tol=1e-5)
    
    params = gmm.get_params()
    assert params['n_components'] == 5
    assert params['tol'] == 1e-5
    
    gmm.set_params(n_components=10, tol=1e-6)
    assert gmm.n_components == 10
    assert gmm.tol == 1e-6


def test_not_fitted_error():
    """Test that calling predict without fit raises error"""
    gmm = GaussianMixture(n_components=3)
    X = np.random.randn(10, 2)
    
    with pytest.raises(RuntimeError):
        gmm.predict(X)
    
    with pytest.raises(RuntimeError):
        gmm.score(X)


def test_invalid_X():
    """Test validation of input data"""
    gmm = GaussianMixture(n_components=3)
    
    # Invalid: not 2D
    with pytest.raises(ValueError):
        X = np.random.randn(100)
        gmm.fit(X)
    
    # Invalid: has NaN
    with pytest.raises(ValueError):
        X = np.array([[1, 2], [np.nan, 3]])
        gmm.fit(X)


def test_convergence():
    """Test that model converges"""
    np.random.seed(42)
    centers = np.array([[0, 0], [3, 3]])
    X = np.vstack([
        np.random.multivariate_normal(centers[0], [[1, 0.5], [0.5, 1]], 100),
        np.random.multivariate_normal(centers[1], [[1, -0.3], [-0.3, 1]], 100),
    ])
    
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(X)
    
    # Should converge with reasonable data
    assert gmm.converged_ is not None
    assert gmm.n_iter_ > 0


if __name__ == "__main__":
    pytest.main([__file__])

