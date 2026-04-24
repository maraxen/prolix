# Euclidean Fast Attention (EFA) — Reading Notes

**Paper**: Frank, J.T., Chmiela, S., Müller, K.-R., & Unke, O.T. (2025).  
*Euclidean Fast Attention – Machine Learning Global Atomic Representations at Linear Cost.*  
arXiv:2412.08541v2. Google DeepMind / TU Berlin / BIFOLD / MPI Informatics / Korea University.

**PDF**: `references/raw/frank2024_euclidean_fast_attention.pdf`  
**Code**: `third_party/euclidean_fast_attention` (submodule, pinned at a7f28ca / v1.0.0-publication)

---

## Core Idea

Standard self-attention is O(N²) in the number of atoms — prohibitive for large MD systems.
EFA achieves **O(N) scaling** via a linear-cost approximation of the attention kernel using:

1. **ERoPE** (Euclidean Rotary Positional Encodings): encodes relative positions and orientations of atom pairs while respecting translational/rotational invariance/equivariance
2. **Sinc kernel** via S² integral: `(1/4π) ∫_{S²} e^{iω u⃗·r⃗_{mn}} du⃗ = sin(ω r_{mn})/(ω r_{mn}) = sinc(ω r_{mn})`
3. **Lebedev quadrature** on S²: approximates the integral with a fixed set of nodes and weights (deterministic, not random)

Different frequencies ω_k allow the model to learn arbitrary radial dependence (Section 4.1).

---

## Key Equations

**EFA output** (Eq. 5):
```
EFA(X, R)_m = (1/4π) ∫_{S²} φ_{u⃗}(q_m, r⃗_m)^T [∑_n φ̄_{u⃗}(k_n, r⃗_n) v_n^T] ⊗ Y(u⃗) du⃗
```
where:
- `q_m, k_n` are query/key features (from atom embeddings X)
- `r⃗_m` is atom m's position
- `φ_{u⃗}(q, r⃗) = ERoPE(q, u⃗·r⃗)` — rotary encoding along direction u⃗
- `Y(u⃗)` — spherical harmonics for equivariance
- No softmax (interactions are additive / size-extensive)

**Sinc kernel** (Eq. 2):
```
(1/4π) ∫_{S²} e^{iω u⃗·r⃗_{mn}} du⃗ = sinc(ω r_{mn})
```

**Lebedev quadrature** (Eq. 12):
```
∫_{S²} f(u⃗) du⃗ ≈ 4π ∑_{j=1}^G λ_j f(u⃗_j)
```
with G fixed nodes u⃗_j and weights λ_j.

**Frequency initialization** (from code):
- Free BC: `ω_k = linspace(0, ω_max, num_features//2) / L_max`
- Periodic BC: `ω_k = arange(1, num_features//2 + 1) * 2π / L_cell`

---

## What EFA Is NOT

- Not a Coulomb electrostatics approximation
- Not Random Fourier Features (RFF) — the quadrature is deterministic
- Not `erfc(αr)/r` or `erf(αr)/r` — the kernel is `sinc(ωr)`, a geometric kernel
- Not a replacement for PME

---

## Integration into Prolix

EFA is designed to augment local MPNNs with global long-range information.
In prolix context, the intended use is likely:
- Add EFA as an attention layer on top of existing atom representations
- Allows the model to capture long-range structural correlations beyond the nonbonded cutoff
- Can be incorporated with minimal changes to existing model architecture

## Outstanding Questions

- [ ] What exactly is the integration point in prolix's MPNN? (atom embeddings? energy readout?)
- [ ] Which Lebedev quadrature order (number of nodes G) is appropriate for MD?
- [ ] How do frequencies ω_k get set — learned or fixed from box size?
- [ ] Periodic BC frequency init requires cell length — how does this interact with PBC in prolix?
- [ ] Equivariant output: does prolix need scalar (L=0) or higher-order (L>0) EFA output?
