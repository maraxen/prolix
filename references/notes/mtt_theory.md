# MTT Log-Det Estimator for Coulomb Free Energy

Algorithm design document — gates Step 10 (implementation). Sprint 9.

---

## 1. Problem Statement

Compute $\log|\det(L)|$ where $L = D - K$ is the **Coulomb-weighted graph Laplacian** over $N$ atoms.

**Kernel matrix** $K \in \mathbb{R}^{N \times N}$:

$$K_{ij} = \texttt{efa\_lebedev\_coulomb\_energy}(\mathbf{r}_i, \mathbf{r}_j)$$

evaluated pairwise via `src/prolix/physics/efa_coulomb.py:209`.

**Degree matrix** $D = \text{diag}(d_i)$ where $d_i = \sum_j K_{ij}$.

**Laplacian** $L = D - K$.

Application: the log-determinant $\log|\det(L)|$ appears as an entropic free-energy correction in Coulomb-coupled molecular systems — quantifying how strongly correlated the charge distribution is across the simulation box.

---

## 2. Hutchinson Trace Estimator

For a symmetric PSD matrix $M$, the matrix logarithm satisfies:

$$\log|\det(M)| = \text{tr}(\log M)$$

The **Hutchinson estimator** converts a trace into a stochastic expectation. For probe vectors $z \sim \mathcal{N}(0, I)$:

$$\mathbb{E}[z^\top f(M) z] = \text{tr}(f(M))$$

Applying $f = \log$:

$$\text{tr}(\log M) = \mathbb{E}[z^\top \log(M) z] \approx \frac{1}{K} \sum_{k=1}^{K} z_k^\top \log(M) z_k$$

**Variance**: $\text{Var}[\hat{v}] \approx \frac{2 \|\log M\|_F^2}{K}$, so variance decays as $O(1/K)$.

**Practical parameters**: $K = 10$–$50$ probes is typical; $K = 20$ is the Sprint 9 target (see Section 6).

---

## 3. Lanczos Tridiagonalization

Lanczos converts the bilinear form $z^\top \log(M) z$ into a scalar problem on a small tridiagonal matrix.

**Algorithm sketch** ($k$-step Lanczos for a single probe $z$):

```
Input: matrix M (as matvec oracle), probe vector z, steps k
Output: tridiagonal T ∈ R^{k×k}, orthonormal basis Q ∈ R^{N×k}

q_0 = z / ||z||
β_0 = 0, q_{-1} = 0
for j = 0, 1, ..., k-1:
    v = M @ q_j
    α_j = q_j^T v
    v = v - α_j * q_j - β_{j-1} * q_{j-1}
    β_j = ||v||
    q_{j+1} = v / β_j

T = tridiag(α_0..α_{k-1}, β_0..β_{k-2})
```

The bilinear form is then approximated:

$$z^\top \log(M) z \approx \|z\|^2 \cdot e_1^\top \log(T) e_1$$

where $\log(T)$ is computed via eigendecomposition of the small $k \times k$ tridiagonal.

**JAX implementation**: use `jax.lax.while_loop` with a carry of `(j, q_prev, q_curr, alpha_list, beta_list)` to implement the recurrence without Python-level loops (enables JIT and grad).

**Re-orthogonalization tradeoff**:

| Strategy | Memory | Cost | Stability |
|---|---|---|---|
| Full (Gram-Schmidt vs all $j$ prior vectors) | $O(k^2)$ | $O(Nk^2)$ | High — required for accurate log-det |
| Partial (re-orth vs last $m$ vectors) | $O(Nm)$ | $O(Nkm)$ | Medium — loss of orthogonality after ~$m$ steps |
| None | $O(N)$ | $O(Nk)$ | Low — ghost eigenvalues degrade accuracy |

**Decision**: use full re-orthogonalization for Sprint 9 ($N \leq 256$); revisit partial for larger systems.

**Reference**: Golub & Meurant, *Matrices, Moments and Quadrature* (2010), Chapters 3–4.

---

## 4. Chebyshev Polynomial Approximation

An alternative to Lanczos: approximate $\log(\lambda)$ by a degree-$M$ Chebyshev polynomial on $[\lambda_\min, \lambda_\max]$, then apply stochastically.

**Domain shift for near-zero eigenvalues**: $L$ is a graph Laplacian with a zero eigenvalue (constant eigenvector). Regularize:

$$L_\text{reg} = L + \varepsilon I, \quad \varepsilon = 10^{-4}$$

This shifts $\lambda_\min$ away from 0, making $\log$ well-defined on the spectrum.

**Chebyshev expansion**:

Map $[\lambda_\min, \lambda_\max] \to [-1, 1]$ via $\tilde{\lambda} = \frac{2\lambda - (\lambda_\max + \lambda_\min)}{\lambda_\max - \lambda_\min}$.

Coefficients via DCT:

$$c_k = \frac{2}{M} \sum_{j=0}^{M-1} \log(\lambda_j) T_k\!\left(\cos\!\frac{j\pi}{M}\right)$$

**Stochastic estimate** ($z \sim \mathcal{N}(0, I)$):

```
Input: L_reg (as matvec), probe z, polynomial degree M, λ_min, λ_max
Compute Chebyshev coefficients c_0..c_M for log(·) on [λ_min, λ_max]
Evaluate p = c_0*z + c_1*(L_reg @ z - shift*z)/scale + ...  (Clenshaw recursion)
Return z^T p  (≈ z^T log(L_reg) z)
```

**Tradeoff vs Lanczos**:

| | Lanczos | Chebyshev |
|---|---|---|
| Spectrum info needed | None (adapts) | Requires $\lambda_\min$, $\lambda_\max$ |
| Per-probe cost | $O(Nk^2)$ with re-orth | $O(NM)$ |
| Accuracy control | Increase $k$ | Increase $M$ |
| JAX friendliness | `while_loop` needed | Fixed unroll, easy JIT |

Chebyshev is preferred when $\lambda_\min$ and $\lambda_\max$ are cheaply estimatable (e.g., via power iteration); Lanczos is preferred when the spectrum is unknown or irregular.

---

## 5. Connection to EFA Kernel

The kernel $K_{ij}$ is the `efa_lebedev_coulomb_energy` function evaluated pairwise (`:209`). The EFA architecture (`efa_coulomb.py:170`, `:135`) produces feature vectors:

$$\phi_i = \texttt{efa\_erf\_features}(\mathbf{r}_i, \theta) \in \mathbb{R}^D$$

The kernel is the inner product in feature space:

$$K_{ij} = \phi_i^\top \phi_j \quad \Rightarrow \quad K = \Phi \Phi^\top, \quad \Phi \in \mathbb{R}^{N \times D}$$

**Key consequence**: matrix-vector products are $O(ND)$, not $O(N^2)$:

$$K \mathbf{v} = \Phi (\Phi^\top \mathbf{v})$$

This makes each Lanczos step $O(ND)$ instead of $O(N^2)$. For $D \ll N$ (the typical EFA regime), this is a substantial saving.

**Do not form $K$ explicitly** in production. Pass a matvec closure to Lanczos/Chebyshev.

**Reference**: Frank et al. 2025; see `src/prolix/physics/efa_coulomb.py` for the feature extraction architecture.

---

## 6. Error Bounds and Convergence

**Hutchinson variance** (from Section 2):

$$\text{Var}[\hat{v}_K] \approx \frac{2\|\log M\|_F^2}{K}$$

Doubling probes halves variance. Relative standard error $\approx \sqrt{2}/\sqrt{K}$ when $\|\log M\|_F \approx |\text{tr}(\log M)|$ (approximately true for near-diagonal spectra).

**Lanczos convergence**: the $k$-step approximation achieves accuracy comparable to the best degree-$k$ polynomial approximation to $\log$ on the spectrum. For smooth spectra, $k = 50$ steps is typically sufficient for $< 1\%$ error in the bilinear form.

**Target parameters for Sprint 9 ($N = 64$)**:

| Parameter | Value | Rationale |
|---|---|---|
| Probes $K$ | 20 | ~10% relative std error |
| Lanczos steps $k$ | 50 | Converged for smooth Coulomb spectra |
| Regularization $\varepsilon$ | $10^{-4}$ | Shifts zero mode safely |

**Sprint 10 milestone** ($N = 256$): same parameters; verify $O(ND)$ scaling holds.

**Accuracy gate**: MTT estimate within 5% of dense baseline on $N = 64$ (see Section 7).

**Accuracy measurement**: compare against `numpy.linalg.slogdet` on small ($N \leq 64$) cases during testing.

---

## 7. Dense Baseline

For validation only — not production use.

**Construction**:

```python
# Build K explicitly — O(N^2) calls to EFA kernel
K = jnp.array([[efa_lebedev_coulomb_energy(r[i], r[j], params)
                 for j in range(N)] for i in range(N)])
D = jnp.diag(K.sum(axis=1))
L = D - K
L_reg = L + 1e-4 * jnp.eye(N)

# Dense log-det
sign, logdet = jnp.linalg.slogdet(L_reg)
# or: sign, logdet = numpy.linalg.slogdet(np.array(L_reg))
```

**Gate**: the MTT stochastic estimate (Sections 2–3) must agree with `logdet` to within 5% on $N = 64$ before Step 10 code is considered correct.

This baseline is only tractable for small $N$ ($\leq 256$ at most) due to $O(N^2)$ memory and $O(N^3)$ factorization cost.
