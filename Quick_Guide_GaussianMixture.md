
# 🚀 Quick Guide — Gaussian Mixture Model (GMM)

A lightweight **NumPy-based implementation** of a **Gaussian Mixture Model (GMM)** trained via the **Expectation-Maximization (EM)** algorithm, inspired by *Mathematics for Machine Learning (Deisenroth et al.)* and scikit-learn.

---

## 🧠 Theory Recap

A **Gaussian Mixture Model (GMM)** assumes that data points are generated from a combination of several Gaussian distributions, each representing a latent cluster.

\[
p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)
\]

where:  
- $\pi_k$: mixing coefficients (weights), $\sum_k \pi_k = 1$  
- $\mu_k$: mean vector of component $k$  
- $\Sigma_k$: covariance matrix of component $k$  

### Expectation-Maximization (EM) Algorithm

1. **E-step (Expectation):**
   \[
   r_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
   {\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
   \]
   Compute *responsibilities* $r_{ik}$ — how likely each point $x_i$ belongs to each Gaussian $k$.

2. **M-step (Maximization):**
   Update the model parameters:
   \[
   \pi_k = \frac{N_k}{N}, \quad
   \mu_k = \frac{1}{N_k}\sum_i r_{ik}x_i, \quad
   \Sigma_k = \frac{1}{N_k}\sum_i r_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top
   \]

---

## ⚙️ Setup and Usage

```python
import numpy as np
from gaussian_mixture import GaussianMixture

# Create dataset
X = np.vstack([
    np.random.multivariate_normal([0, 0], np.eye(2), 150),
    np.random.multivariate_normal([3, 3], np.eye(2), 150)
])

# Initialize model
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)

# Fit model
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Predict probabilities (responsibilities)
probs = gmm.predict_proba(X)
```

---

## 📊 EM Algorithm Visual Flow

```text
                ┌──────────────────────┐
                │     Input Data X     │
                └──────────┬───────────┘
                           │
             Initialization (Means, Covariances, Weights)
                           │
                           ▼
               ┌────────────────────────┐
               │        E-Step          │
               │  Compute r_ik          │
               └──────────┬─────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │        M-Step          │
               │  Update μ_k, Σ_k, π_k  │
               └──────────┬─────────────┘
                           │
                           ▼
                  Check Convergence?
                        │
               ┌────────┴────────┐
               │                 │
             Yes                 No
               │                 │
               ▼                 │
       Return Final Model ◄──────┘
```

---

## 🧩 Conditional Probability (Advanced)

You can compute conditional mixtures such as $p(X \mid Y = y)$:

\[
m_{X|Y,k} = m_{X,k} + \Sigma_{XY,k}\Sigma_{YY,k}^{-1}(y - m_{Y,k})
\]
\[
\Sigma_{X|Y,k} = \Sigma_{XX,k} - \Sigma_{XY,k}\Sigma_{YY,k}^{-1}\Sigma_{YX,k}
\]

```python
cond = gmm.conditional(X_indices=[0], Y_indices=[1], y=np.array([0.5]))
print("Conditional means:", cond["means"])
```

---

## 🧮 Model Attributes

| Attribute | Description |
|------------|--------------|
| `weights_` | Mixing coefficients $\pi_k$ |
| `means_` | Mean vectors $\mu_k$ |
| `covariances_` | Covariance matrices $\Sigma_k$ |
| `precisions_cholesky_` | Cholesky decomposition of precisions |
| `converged_` | Whether EM converged |
| `n_iter_` | Number of EM iterations |
| `lower_bound_` | Final log-likelihood value |

---

## 📈 Model Evaluation (AIC/BIC)

```python
for k in range(1, 6):
    model = GaussianMixture(n_components=k, random_state=42).fit(X)
    print(f"{k} components → AIC: {model.aic(X):.2f}, BIC: {model.bic(X):.2f}")
```

Lower AIC/BIC → better balance between model complexity and fit.

---

## 💡 Tips for New Users

1. Normalize your data before fitting.  
2. Try `covariance_type='diag'` for high-dimensional data.  
3. Use `n_init > 1` to avoid local minima.  
4. Increase `reg_covar` if covariance becomes singular.  
5. Check `gmm.converged_` after fitting.

---

## 🧠 Developer Notes

- This implementation uses **NumPy only** (no SciPy).  
- Covariances are regularized to ensure positive-definiteness.  
- Log-likelihood is computed via **Cholesky decomposition** for numerical stability.  
- Use `sample()` to generate new data from the fitted model.

### Extending the Model

You can easily add:
- Mini-batch EM for large datasets  
- Visualization tools (ellipses, 3D clusters)  
- Bayesian GMM (Dirichlet priors)

---

## 📚 References

- Deisenroth, M.P., Faisal, A.A., Ong, C.S. (2020). *Mathematics for Machine Learning*, Chapter 11  
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*  
- Dempster, Laird & Rubin (1977). *Expectation-Maximization Algorithm*  

---

*Prepared by Adarsh Raj — Gaussian Mixture Implementation Guide*
