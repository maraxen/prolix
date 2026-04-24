# Avron & Toledo 2011: Randomized Algorithms for Trace Estimation

**Citation:** Avron, H., & Toledo, S. (2011). Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix. *Journal of the ACM*, 58(2), 8:1–8:34.

**DOI:** https://doi.org/10.1145/1944345.1944349

**PDF (alternative):** https://www.cs.tau.ac.il/~stoledo/Bib/Pubs/trace3.pdf

## Summary

Avron and Toledo analyze randomized trace estimation for implicit (matrix-free) symmetric positive semi-definite (SPSD) matrices. They provide rigorous sample complexity bounds (number of samples M needed to guarantee relative error ε with probability ≥ 1-δ) for two estimators: **Hutchinson's method** (Gaussian random vectors) and **trace estimation via quadrature**. This is the foundational work for understanding convergence guarantees in stochastic trace and log-det estimation.

## Key Results

### Hutchinson's Estimator
- **Formula**: E[z^T A z] ≈ tr(A), where z ~ N(0, I)
- **Variance**: Var(z^T A z) = 2 tr(A²) - (tr(A))² (approximately)
- **Sample complexity**: M = O(ε^{-2} ln(δ^{-1})) samples to guarantee relative error ≤ ε with probability ≥ 1-δ
- **Computational cost**: O(M · cost(matvec))

### Quadrature-Based Estimators
- **Gaussian quadrature** combined with Lanczos tridiagonalization
- **Convergence**: Often faster than Hutchinson due to matrix-dependent quadrature weights
- **Sample complexity**: Data-dependent; typically better for functions with analytic structure (e.g., log)

### Theoretical Contributions
- **Rigorous bounds**: Not just variance analysis, but guaranteed high-probability error bounds
- **Implicit matrices**: Applies to matrix-free representations (matvec functions)
- **Optimality**: Proves bounds are essentially tight

## Relevance to Prolix MTT Log-Det Estimator

**Foundation for stochastic log-det estimation**, directly applicable to Coulomb Laplacian:

1. **Log-det as trace of function**: log(det(A)) = tr(log(A)); Avron & Toledo framework applies
2. **Matrix-free suitability**: MTT computes matvec implicitly; no need for explicit Laplacian matrix
3. **Convergence guarantees**: Sample complexity bounds inform how many Lanczos samples needed for target accuracy

## Limitations and Extensions

### For Signed/Indefinite Matrices
- **Original scope**: SPSD matrices only
- **Problem with indefinite**: Coulomb graph Laplacian with signed weights is **not PSD** (has negative eigenvalues from negative weights)
- **Consequence**: Standard variance bounds may not hold; convergence could be slow or require regularization

### For Matrix Functions
- **Works well for**: tr(A^p), tr(log(A)), tr(√A) when A is SPSD
- **Requires**: Analytic structure in the function for quadrature acceleration

## Related Work
- **Ubaru et al. 2017**: "Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature" extends to general matrix functions with explicit error bounds
- **Chowdhury & Mahoney 2018**: Randomized least-squares with preconditioning

## Open Questions
1. Can Avron-Toledo theory be extended to indefinite (signed Laplacian) matrices?
2. Does log-det estimation converge for non-PSD Coulomb Laplacian? At what rate?
3. Could low-rank regularization of signed Laplacian improve convergence?
