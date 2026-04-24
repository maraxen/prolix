# Matfree Evaluation: Suitability for Coulomb Laplacian Log-Det Estimation

**Date:** 2026-04-24  
**Status:** Evaluation complete; recommend conditional adoption

## Executive Summary

**matfree** (v0.5.5, PyPI, Dec 2025) is a JAX-native matrix-free linear algebra library that provides stochastic Lanczos quadrature for log-det estimation. It is **well-suited for the MTT estimator** but with one significant caveat: **it does not explicitly support indefinite matrices**. Since the Coulomb graph Laplacian with signed weights is indefinite (negative eigenvalues from repulsive interactions), matfree would require either (a) PSD regularization or (b) custom indefinite-matrix extension.

---

## Matfree Capabilities

### Availability & Installation
- **PyPI**: Yes, `pip install matfree`
- **Latest version**: 0.5.5 (released 2025-12-02)
- **Python requirement**: >=3.11
- **Primary dependency**: JAX only (users install JAX separately)
- **Not in prolix/pyproject.toml**: Would be an optional dependency addition

### Core Features for MTT
- ✅ **Stochastic Lanczos quadrature** for tr(f(A)) including log-det
- ✅ **Matrix-free (implicit) input**: Accepts matvec functions, not explicit matrices
- ✅ **JAX native**: Full jit, vmap, grad, and pytree support
- ✅ **Uncertainty quantification**: Provides error estimates alongside estimates
- ✅ **Batching**: Can estimate multiple traces or vectors in parallel
- ✅ **Trace estimation**: Hutchinson and quadrature variants

### Limitations for MTT
- ❌ **Indefinite matrices**: No explicit support; designed for PSD or product-form matrices
  - Documentation shows `tridiag_sym()` for symmetric case (implicitly assumes PSD)
  - For Coulomb Laplacian L = D - K with K indefinite, behavior is undefined
- ⚠️  **API design**: Assumes matrix is provided or matvec is easily defined; doesn't expose raw Lanczos state for advanced control
- ⚠️  **Extensions**: Cannot easily modify algorithm without forking if indefinite support is needed

---

## API Overview: How It Works

### Input: Matrix-Free Representation
```python
import matfree
import jax
import jax.numpy as jnp

# Define implicit matrix via matvec function
def coulomb_matvec(x, positions, charges):
    """Matrix-vector product for Coulomb Laplacian (implicit)."""
    # Compute (D - K) @ x where K is Coulomb kernel interaction
    return degree_times_x(x) - coulomb_interaction(x, positions, charges)

# Or for explicit matrix (e.g., for testing):
# A = jnp.array(...)
# matvec = lambda x: A @ x
```

### Log-Determinant Estimation
```python
from matfree import decomp, stochtrace, funm

# Sample random vector z ~ N(0, I)
sampler = stochtrace.sampler_normal()
z = sampler(jax.random.PRNGKey(0), shape=(n_residues,), dtype=jnp.float32)

# Tridiagonalize via Lanczos
tridiag_fn = decomp.tridiag_sym(matvec=coulomb_matvec, depth=50)
T = tridiag_fn(z)  # Tridiagonal matrix (implicit or explicit)

# Compute log-det via Gauss-quadrature on T's eigenvalues
logdet_quad_fn = funm.integrand_funm_sym_logdet(T)

# Stochastic estimator
estimator = stochtrace.estimator(problem=logdet_quad_fn, sampler=sampler)
logdet_estimate = estimator(coulomb_matvec, jax.random.PRNGKey(seed))
```

### Expected Output
```python
# Estimate is a JAX Array with shape ()
# Also returns uncertainty estimate
estimate, std_error = estimator(coulomb_matvec, key, return_std=True)
```

---

## Indefinite Matrix Problem

### Why Coulomb Laplacian is Indefinite
- **Degree matrix D**: Positive diagonal (# of neighbors per residue)
- **Coulomb kernel K**: Mixed sign eigenvalues (repulsive → negative contribution)
- **Laplacian L = D - K**: Eigenvalue spectrum includes negative values (unbalanced)

### Current matfree Assumption
- `tridiag_sym()` performs Lanczos iteration on symmetric matrix A
- **Silent assumption**: A is PSD (or at least that quadrature weights remain positive)
- For indefinite A, quadrature nodes in Gauss-Kronrod integration may become complex, breaking algorithm

### Impact on MTT
- **Log-det of indefinite matrix**: det(L) can be negative; log(det) is complex
- **Stochastic Lanczos**: Convergence guarantees (Ubaru et al. 2017) assume PSD; behavior for indefinite is unanalyzed
- **Workaround required**: Either (a) regularize to ensure PSD, or (b) implement indefinite variant

---

## Recommended Approach: API Sketch

### Option A: With matfree (PSD Regularization)

```python
from prolix.physics import coulomb_laplacian
from matfree import stochtrace, decomp, funm
import jax
import jax.numpy as jnp

def log_det_coulomb_with_matfree(positions, charges, neighbor_idx, dt=0.5):
    """
    Estimate log-det of Coulomb Laplacian via matfree stochastic Lanczos.
    
    Regularization: Add small diagonal shift to ensure PSD.
    log(det(L + eps*I)) ≈ log(det(L)) + eps*tr(L^{-1}) for small eps
    
    Args:
        positions: (n_atoms, 3) Coulomb coordinates
        charges: (n_atoms,) atomic charges
        neighbor_idx: sparse connectivity
        dt: timestep (AKMA units)
        
    Returns:
        log_det_estimate: float, estimated log-determinant
        std_error: float, standard error of estimate
    """
    n = positions.shape[0]
    eps_reg = 1e-4  # Regularization parameter
    
    # Define implicit matvec for regularized Coulomb Laplacian
    def matvec_regularized(x):
        coulomb_part = coulomb_laplacian.matvec(x, positions, charges, neighbor_idx)
        return coulomb_part + eps_reg * x  # Add regularization
    
    # Stochastic Lanczos log-det
    sampler = stochtrace.sampler_normal()
    estimator = stochtrace.estimator(
        problem=lambda T: funm.integrand_funm_sym_logdet(T),
        sampler=sampler,
        depth=min(50, n),  # Lanczos depth
    )
    
    logdet_reg, std = estimator(matvec_regularized, jax.random.PRNGKey(dt), return_std=True)
    
    # Correction: log(det(L + eps*I)) -> log(det(L)) using trace of inverse
    # Approximation: correction ≈ eps * log_trace_inverse (expensive, skip for now)
    # return logdet_reg - correction  # More accurate but requires extra computation
    
    return logdet_reg, std
```

### Option B: Custom Indefinite Lanczos (No matfree)

```python
# If matfree proves insufficient, implement indefinite Lanczos per Ubaru et al. 2017
from prolix.linalg import lanczos_indefinite

def log_det_coulomb_custom(positions, charges, neighbor_idx, dt=0.5):
    """
    Estimate log-det via custom Lanczos adapted for indefinite matrices.
    Based on: Ubaru, S., Chen, J., Saad, Y. (2017). 
              "Fast estimation of tr(f(A)) via Stochastic Lanczos Quadrature"
    
    Handles negative eigenvalues explicitly.
    """
    def matvec(x):
        return coulomb_laplacian.matvec(x, positions, charges, neighbor_idx)
    
    # Tridiagonalize (returns T with mixed-sign spectrum)
    T = lanczos_indefinite.tridiag(matvec, depth=50, num_seeds=10)
    
    # Gaussian quadrature on indefinite T
    logdet_estimate = lanczos_indefinite.quadrature_logdet(T)
    
    return logdet_estimate
```

---

## Recommendation & Next Steps

### Verdict: **CONDITIONAL YES**

**Adopt matfree if:**
1. Coulomb Laplacian is regularized to be PSD (add diagonal shift)
2. Regularization error is acceptable for the physics
3. Need rapid prototyping with JAX integration

**Avoid matfree / implement custom if:**
1. Indefinite eigenvalue handling is critical for correctness
2. Physics requires true log(det(L)) without regularization
3. Can afford implementation burden (reference: Ubaru et al. 2017 + matfree Lanczos code)

### Implementation Path

**Phase 3a (Short-term, 1 week):**
1. Add matfree as optional dependency: `prolix[lanczos]` in pyproject.toml
2. Implement Option A (PSD regularization) with matfree
3. Validate: check that logdet estimates converge; compare with finite-difference Hessian log-det

**Phase 3b (Medium-term, 2-3 weeks, only if 3a insufficient):**
1. Implement indefinite Lanczos quadrature from Ubaru et al. 2017
2. Benchmark matfree vs custom indefinite on Coulomb Laplacian
3. Decision: keep matfree or replace with custom

### Testing Checklist
- [ ] matfree log-det converges as Lanczos depth increases
- [ ] Uncertainty estimates are reasonable (empirical coverage at 68%)
- [ ] Performance: matfree competitive with numpy/scipy for small (n<1000) graphs
- [ ] JAX grad works: can differentiate log-det w.r.t. positions
- [ ] Regularization error is <1% for typical protein networks

---

## Alternatives & Comparison

| Approach | Pros | Cons | Effort |
|----------|------|------|--------|
| **matfree (PSD)** | JAX native, fast, documented | Indefinite support missing, regularization error | Low (1-2 days) |
| **Custom Lanczos (indefinite)** | True indefinite handling, no regularization | Reimplementing research code, JAX integration | Medium (1-2 weeks) |
| **scipy.sparse (dense fallback)** | Proven, supports indefinite | Not JAX native, vmap/grad harder, slow for large n | Low initial, high long-term |
| **JAX-MD built-ins** | Already integrated | May lack log-det, no uncertainty | Low but limited |

---

## Dependencies & Integration

### Add to pyproject.toml
```toml
[project.optional-dependencies]
lanczos = [
    "matfree>=0.5.5",
]
```

### Import in prolix
```python
# prolix/physics/mtg.py or similar
try:
    import matfree
    HAS_MATFREE = True
except ImportError:
    HAS_MATFREE = False
    import warnings
    warnings.warn("matfree not installed; stochastic log-det unavailable. Install with: pip install prolix[lanczos]")
```

---

## References
- matfree docs: https://pnkraemer.github.io/matfree/
- matfree GitHub: https://github.com/pnkraemer/matfree
- PyPI: https://pypi.org/project/matfree/
- Ubaru et al. 2017: https://epubs.siam.org/doi/abs/10.1137/16M1104974
- matfree log-det tutorial: https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/

---

## Final Note

matfree is a **high-quality library** with excellent JAX integration. The indefinite matrix limitation is not a flaw; it reflects the design choice to target PSD matrices (common in ML). For Prolix MTT, either regularization or a custom indefinite extension is needed. Both are feasible; matfree + regularization is the faster path if physics permits.
