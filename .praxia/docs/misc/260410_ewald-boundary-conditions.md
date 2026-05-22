# Ewald Summation and Macroscopic Boundary Conditions

When working with explicit solvent and long-range electrostatics via Smooth Particle Mesh Ewald (SPME) or standard Ewald summation, boundary condition treatment diverges from simple electrostatic intuition around finite periodic blocks.

## The Tin-Foil Boundary Condition

Standard Ewald summation derivations assume that the infinite periodically replicated simulation cell is surrounded by a continuous medium of **infinite relative permittivity** ($\epsilon' = \infty$). This is referred to as "Tin-foil boundary conditions".

In this limit, any macroscopic dipole field originating from the polarization of the unit cells is completely screened at the boundary. As a result, the standard SPME formula has no external surface correction term.

## The Vacuum Boundary Condition

If you establish an exact sum by evaluating `1/r` interactions over a large discrete block of unit cells (e.g., $10 \times 10 \times 10$) without a conducting boundary layer, you are implicitly simulating the interactions in a state surrounded by a perfect vacuum ($\epsilon' = 1$). 

If the primary unit cell has a non-zero macroscopic dipole moment $\mathbf{M} = \sum q_i \mathbf{r}_i$, the macroscopic shape of this finite crystal block will cause it to harbor a uniform depolarizing field that introduces an unexpected energy offset.

## Evaluating the Offset

For a spherical/cubic sum in a vacuum, the energetic difference (offset) between the finite "Vacuum" block and the true "Tin-foil" Ewald limit is exactly the energy of the macroscopic surface dipole layer:

$$ E_{surface} = \frac{2\pi}{3 V_{box}} \cdot C_{coulomb} \cdot |\mathbf{M}|^2 $$

### Diagnostic Impact
When verifying or validating reciprocal space kernels (like SPME matrices) against slow rigorous brute-force block sums, you **must factor in this $E_{surface}$ term** if the test case is not perfectly dipole-neutral. If testing a net dipole system, subtracting the analytical E_{surface} from the brute-force vacuum limit reproduces an exact numerical match with the SPME kernel.

**Crucially:** Never introduce artificial pre-factors or re-scale FFT grid densities if an empirical discrepancy approximately equals the surface dipole potential! The algorithm is algebraically sound; the assumptions for the analytical baseline were just differing.
