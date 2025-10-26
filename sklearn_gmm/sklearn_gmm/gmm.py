"""
Gaussian Mixture Model implementation compatible with scikit-learn.
"""
import math
from typing import Optional, Tuple, Literal, Dict, Any

import numpy as np

CovarianceType = Literal["full", "tied", "diag", "spherical"]


class GaussianMixture:
    """
    Gaussian Mixture Model estimated via Expectation-Maximization (EM).

    This implementation follows standard textbook equations for GMMs with
    support for different covariance types and sklearn-compatible API.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of:
        - 'full': each component has its own general covariance matrix
        - 'tied': all components share the same general covariance matrix
        - 'diag': each component has its own diagonal covariance matrix
        - 'spherical': each component has its own single variance

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans++', 'random'}, default='kmeans++'
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of:
        - 'kmeans++' : use K-Means++ initialization for means
        - 'random' : use random initialization

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    random_state : int or None, default=None
        Controls the random seed used for initialization.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on the covariance_type:
        - 'full': shape (n_components, n_features, n_features)
        - 'tied': shape (n_features, n_features)
        - 'diag': shape (n_components, n_features)
        - 'spherical': shape (n_components,)

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Log-likelihood (actually the lower bound) of the model.

    precisions_cholesky_ : array-like
        The Cholesky decomposition of the precision matrices. The shape depends
        on the covariance_type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn_gmm import GaussianMixture
    >>> X = np.random.randn(100, 2)
    >>> gmm = GaussianMixture(n_components=3)
    >>> gmm.fit(X)
    GaussianMixture(n_components=3)
    >>> gmm.predict(X)
    array([0, 1, 2, ...])
    >>> gmm.score(X)
    -2.345...
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: CovarianceType = "full",
        tol: float = 1e-3,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal["kmeans++", "random"] = "kmeans++",
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.init_params = init_params
        self.reg_covar = float(reg_covar)
        self.random_state = random_state

        # Learned parameters (set after fit)
        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.precisions_cholesky_: Optional[np.ndarray] = None

        # Training information
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = -np.inf

        self._validate_hyperparams()

    # -------------------------- Public API --------------------------
    def fit(self, X: np.ndarray, y: None = None) -> "GaussianMixture":
        """
        Fit the GMM to data X using the EM algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        self : GaussianMixture
            Returns self for method chaining.
        """
        X = self._validate_X(X)
        rng = np.random.default_rng(self.random_state)

        best_lower_bound = -np.inf
        best_params = None

        for run in range(self.n_init):
            # Initialize weights, means, covariances
            weights, means, covariances = self._initialize_parameters(X, rng)

            # Compute precision Cholesky
            precisions_chol = self._compute_precisions_cholesky(covariances)

            lower_bound = -np.inf
            converged = False
            for n_iter in range(1, self.max_iter + 1):
                # E-step: compute responsibilities
                log_prob_norm, responsibilities = self._e_step(X, weights, means, precisions_chol)

                # M-step: update parameters
                weights, means, covariances = self._m_step(X, responsibilities)

                # Enforce constraints
                weights = np.maximum(weights, np.finfo(float).eps)
                weights /= weights.sum()

                covariances = self._ensure_valid_covariances(covariances, X.shape[1])

                # Recompute precision cholesky
                precisions_chol = self._compute_precisions_cholesky(covariances)

                # Check convergence
                new_lower_bound = float(log_prob_norm.mean())
                change = new_lower_bound - lower_bound
                lower_bound = new_lower_bound

                if abs(change) < self.tol:
                    converged = True
                    self.n_iter_ = n_iter
                    break
            else:
                self.n_iter_ = self.max_iter

            if lower_bound > best_lower_bound or best_params is None:
                best_lower_bound = lower_bound
                best_params = (weights.copy(), means.copy(), covariances.copy(), precisions_chol.copy())

        # Store the best run
        self.weights_, self.means_, self.covariances_, self.precisions_cholesky_ = best_params  # type: ignore[assignment]
        self.lower_bound_ = float(best_lower_bound)
        self.converged_ = True if best_lower_bound > -np.inf else False
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        labels : array of shape (n_samples,)
            Index of the component each sample belongs to.
        """
        X = self._ensure_fitted_validate_X(X)
        # Choose component with maximum posterior
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : array of shape (n_samples, n_components)
            Posterior probability of each component.
        """
        X = self._ensure_fitted_validate_X(X)
        _, resp = self._e_step(X, self.weights_, self.means_, self.precisions_cholesky_)  # type: ignore[arg-type]
        return resp

    def responsibilities(self, X: np.ndarray) -> np.ndarray:
        """
        Return responsibilities (posterior probabilities) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        responsibilities : array of shape (n_samples, n_components)
            Responsibilities of each component for each sample.
        """
        X = self._ensure_fitted_validate_X(X)
        _, resp = self._e_step(X, self.weights_, self.means_, self.precisions_cholesky_)  # type: ignore[arg-type]
        return resp

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        log_prob : array of shape (n_samples,)
            Log probabilities of each data point.
        """
        X = self._ensure_fitted_validate_X(X)
        return self._compute_log_prob_norm(X)

    def score(self, X: np.ndarray, y: None = None) -> float:
        """
        Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        y : None
            Ignored. Present for sklearn compatibility.

        Returns
        -------
        score : float
            Log-likelihood of the data under the model.
        """
        X = self._ensure_fitted_validate_X(X)
        return float(self._compute_log_prob_norm(X).mean())

    def bic(self, X: np.ndarray) -> float:
        """
        Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        bic : float
            The lower the better.
        """
        X = self._ensure_fitted_validate_X(X)
        n_params = self._num_free_parameters(X.shape[1])
        log_likelihood = self._compute_log_prob_norm(X).sum()
        return float(n_params * math.log(X.shape[0]) - 2.0 * log_likelihood)

    def aic(self, X: np.ndarray) -> float:
        """
        Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        aic : float
            The lower the better.
        """
        X = self._ensure_fitted_validate_X(X)
        n_params = self._num_free_parameters(X.shape[1])
        log_likelihood = self._compute_log_prob_norm(X).sum()
        return float(2.0 * n_params - 2.0 * log_likelihood)

    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        random_state : int or None, default=None
            Controls the random seed.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Randomly generated sample.

        y : array of shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        n_components = self.n_components
        n_features = self.means_.shape[1]  # type: ignore[union-attr]
        
        # Sample component indices according to mixture weights
        component_choices = rng.choice(n_components, size=n_samples, p=self.weights_)
        samples = np.empty((n_samples, n_features), dtype=float)
        
        for k in range(n_components):
            mask = component_choices == k
            if not np.any(mask):
                continue
            mean = self.means_[k]  # type: ignore
            # Construct covariance matrix according to covariance_type
            if self.covariance_type == "full":
                cov = self.covariances_[k]  # type: ignore
            elif self.covariance_type == "tied":
                cov = self.covariances_  # type: ignore
            elif self.covariance_type == "diag":
                cov = np.diag(self.covariances_[k])  # type: ignore
            else:  # spherical
                cov = np.eye(n_features) * self.covariances_[k]  # type: ignore
            samples[mask] = rng.multivariate_normal(mean, cov, size=mask.sum())
        return samples, component_choices

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Not used, present for sklearn compatibility.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "init_params": self.init_params,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "GaussianMixture":
        """
        Set the parameters of this estimator.

        Returns
        -------
        self : GaussianMixture
            Returns self for method chaining.
        """
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        self._validate_hyperparams()
        return self

    # ----------------------- Internal utilities ---------------------
    def _validate_hyperparams(self) -> None:
        if self.n_components <= 0:
            raise ValueError("n_components must be > 0")
        if self.covariance_type not in ("full", "tied", "diag", "spherical"):
            raise ValueError("invalid covariance_type")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self.n_init <= 0:
            raise ValueError("n_init must be > 0")
        if self.init_params not in ("kmeans++", "random"):
            raise ValueError("init_params must be 'kmeans++' or 'random'")
        if self.reg_covar < 0:
            raise ValueError("reg_covar must be >= 0")

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        if X.shape[0] < self.n_components:
            raise ValueError("n_samples must be >= n_components")
        return X.astype(float, copy=False)

    def _ensure_fitted_validate_X(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self._validate_X(X)

    def _check_is_fitted(self) -> None:
        if self.means_ is None or self.weights_ is None or self.covariances_ is None:
            raise RuntimeError("Model is not fitted. Call fit(X) first.")

    # ---------------- Initialization helpers ----------------
    def _random_means_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        return X[indices]

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """KMeans++ initialization for better convergence."""
        n_samples, n_features = X.shape
        centers = np.empty((self.n_components, n_features), dtype=float)
        first_idx = rng.integers(0, n_samples)
        centers[0] = X[first_idx]
        closest_dist_sq = np.full(n_samples, np.inf, dtype=float)
        
        for c in range(1, self.n_components):
            diff = X - centers[c - 1]
            dist_sq_new_center = np.einsum("ij,ij->i", diff, diff)
            np.minimum(closest_dist_sq, dist_sq_new_center, out=closest_dist_sq)
            probs = closest_dist_sq / closest_dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centers[c] = X[next_idx]
        return centers

    def _initialize_parameters(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape
        
        # Initialize means
        if self.init_params == "kmeans++":
            means = self._kmeanspp_init(X, rng)
        else:
            means = self._random_means_init(X, rng)

        # Uniform initial weights
        weights = np.full(self.n_components, 1.0 / self.n_components, dtype=float)

        # Initialize covariances
        if self.covariance_type == "full":
            base_cov = np.cov(X, rowvar=False)
            covariances = np.array([base_cov + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == "tied":
            covariances = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
        elif self.covariance_type == "diag":
            variances = X.var(axis=0) + self.reg_covar
            covariances = np.tile(variances, (self.n_components, 1))
        else:  # spherical
            scalar = float(X.var() + self.reg_covar)
            covariances = np.full(self.n_components, scalar, dtype=float)

        return weights, means, covariances

    # ---------------- Numerical helpers ----------------
    def _ensure_valid_covariances(self, covariances: np.ndarray, n_features: int) -> np.ndarray:
        """Add regularization to ensure positive-definiteness."""
        if self.covariance_type == "full":
            covs = covariances.copy()
            for k in range(self.n_components):
                covs[k] = covs[k].copy()
                covs[k].flat[:: n_features + 1] += self.reg_covar
            return covs
        elif self.covariance_type == "tied":
            cov = covariances.copy()
            cov = cov.copy()
            cov.flat[:: n_features + 1] += self.reg_covar
            return cov
        elif self.covariance_type == "diag":
            return np.maximum(covariances, self.reg_covar)
        else:  # spherical
            return np.maximum(covariances, self.reg_covar)

    def _compute_precisions_cholesky(self, covariances: np.ndarray) -> np.ndarray:
        """Compute Cholesky factor of precision for numerical stability."""
        if self.covariance_type == "full":
            n_components, n_features, _ = covariances.shape
            precisions_chol = np.empty_like(covariances)
            for k in range(n_components):
                cov = covariances[k].copy()
                cov.flat[:: n_features + 1] += self.reg_covar
                chol = np.linalg.cholesky(cov)
                precisions_chol[k] = np.linalg.solve(chol, np.eye(n_features)).T
            return precisions_chol
        elif self.covariance_type == "tied":
            cov = covariances.copy()
            n_features = cov.shape[0]
            cov.flat[:: n_features + 1] += self.reg_covar
            chol = np.linalg.cholesky(cov)
            return np.linalg.solve(chol, np.eye(n_features)).T
        elif self.covariance_type == "diag":
            return 1.0 / np.sqrt(covariances)
        else:  # spherical
            return 1.0 / np.sqrt(covariances)

    def _estimate_log_gaussian_prob(
        self, X: np.ndarray, means: np.ndarray, precisions_cholesky: np.ndarray
    ) -> np.ndarray:
        """Compute log N(x | mu_k, Sigma_k) for all (n,k)."""
        n_samples, n_features = X.shape
        K = self.n_components
        
        if self.covariance_type == "full":
            log_prob = np.empty((n_samples, K), dtype=float)
            log_det = np.empty(K, dtype=float)
            for k in range(K):
                pc = precisions_cholesky[k]
                y = (X - means[k]) @ pc
                log_det[k] = np.log(np.abs(np.diag(pc))).sum()
                log_prob[:, k] = -0.5 * (np.sum(y * y, axis=1) + n_features * np.log(2.0 * np.pi)) + log_det[k]
            return log_prob
        elif self.covariance_type == "tied":
            pc = precisions_cholesky
            y = (X[:, None, :] - means[None, :, :]) @ pc
            log_det = np.log(np.abs(np.diag(pc))).sum()
            return -0.5 * (np.sum(y * y, axis=2) + n_features * np.log(2.0 * np.pi)) + log_det
        elif self.covariance_type == "diag":
            precisions = precisions_cholesky ** 2
            log_det = 0.5 * np.sum(np.log(precisions), axis=1)
            diff = X[:, None, :] - means[None, :, :]
            maha = np.sum(diff * diff * precisions[None, :, :], axis=2)
            return -0.5 * (maha + n_features * np.log(2.0 * np.pi)) + log_det
        else:  # spherical
            precisions = (precisions_cholesky ** 2).astype(float)
            log_det = 0.5 * n_features * np.log(precisions)
            diff = X[:, None, :] - means[None, :, :]
            maha = np.sum(diff * diff, axis=2) * precisions[None, :]
            return -0.5 * (maha + n_features * np.log(2.0 * np.pi)) + log_det

    def _estimate_weighted_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Return log [π_k * N(x | μ_k, Σ_k)]."""
        log_gauss = self._estimate_log_gaussian_prob(X, self.means_, self.precisions_cholesky_)  # type: ignore[arg-type]
        return log_gauss + np.log(self.weights_)[None, :]

    def _compute_log_prob_norm(self, X: np.ndarray) -> np.ndarray:
        """Compute log p(x) = logsumexp_k [log π_k + log N_k(x)]."""
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        max_log = weighted_log_prob.max(axis=1, keepdims=True)
        logsumexp = max_log + np.log(np.exp(weighted_log_prob - max_log).sum(axis=1, keepdims=True))
        return logsumexp.ravel()

    def _e_step(
        self,
        X: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        precisions_cholesky: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """E-step: compute responsibilities."""
        log_gauss = self._estimate_log_gaussian_prob(X, means, precisions_cholesky)
        weighted_log_prob = log_gauss + np.log(weights)[None, :]
        max_log = weighted_log_prob.max(axis=1, keepdims=True)
        exp_log_prob = np.exp(weighted_log_prob - max_log)
        sum_exp = exp_log_prob.sum(axis=1, keepdims=True)
        responsibilities = exp_log_prob / sum_exp
        log_prob_norm = max_log + np.log(sum_exp)
        return log_prob_norm.ravel(), responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: update parameters using responsibilities."""
        n_samples, n_features = X.shape
        nk = responsibilities.sum(axis=0) + 10.0 * np.finfo(float).eps

        # Update weights
        weights = nk / n_samples

        # Update means
        means = (responsibilities.T @ X) / nk[:, None]

        # Update covariances
        if self.covariance_type == "full":
            covariances = np.empty((self.n_components, n_features, n_features), dtype=float)
            for k in range(self.n_components):
                diff = X - means[k]
                cov = (responsibilities[:, k][:, None] * diff).T @ diff / nk[k]
                cov.flat[:: n_features + 1] += self.reg_covar
                covariances[k] = cov
        elif self.covariance_type == "tied":
            cov = np.zeros((n_features, n_features), dtype=float)
            for k in range(self.n_components):
                diff = X - means[k]
                cov += (responsibilities[:, k][:, None] * diff).T @ diff
            cov /= nk.sum()
            cov.flat[:: n_features + 1] += self.reg_covar
            covariances = cov
        elif self.covariance_type == "diag":
            covariances = np.empty((self.n_components, n_features), dtype=float)
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = (responsibilities[:, k][:, None] * (diff * diff)).sum(axis=0) / nk[k] + self.reg_covar
        else:  # spherical
            covariances = np.empty(self.n_components, dtype=float)
            for k in range(self.n_components):
                diff = X - means[k]
                covariances[k] = ((responsibilities[:, k] * np.sum(diff * diff, axis=1)).sum() / (nk[k] * n_features)) + self.reg_covar

        return weights, means, covariances

    def _num_free_parameters(self, n_features: int) -> int:
        """Count free parameters for AIC/BIC."""
        weights_params = self.n_components - 1
        means_params = self.n_components * n_features
        
        if self.covariance_type == "full":
            cov_params_per = n_features * (n_features + 1) // 2
            cov_params = self.n_components * cov_params_per
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) // 2
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        else:
            cov_params = self.n_components
        return weights_params + means_params + cov_params

