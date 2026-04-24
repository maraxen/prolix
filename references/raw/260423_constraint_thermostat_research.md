# Constraint-Aware Thermostat Research (2026-04-23)

**Status**: Complete  
**Confidence**: HIGH  
**Research Query**: Validate constraint-aware (Jacobian-projected) Langevin thermostats for Prolix v0.3.0

## Executive Summary

The constraint-aware Langevin approach (projecting thermostat noise to unconstrained DOF only) has strong mathematical precedent in peer-reviewed literature and no documented failure modes. Prolix's observed dt≤0.5fs limitation is validated by recent publications. Proceed with Phase 1 theory derivation.

## Key Sources

**Mathematical Foundations:**
- Zhang et al. (2019) "A unified efficient thermostat scheme for the canonical ensemble with holonomic or isokinetic constraints" (160+ citations)
- Peters & Goga (2014) "Stochastic dynamics with correct sampling for constrained systems" 
- Hartmann & Schütte (2005) "A geometric approach to constrained molecular dynamics and free energy"
- Walter, Hartmann & Maddocks (2011) Mass-matrix projection of covariance for constrained systems

**Validation of Prolix's Constraint:**
- Asthagiri & Beck (2023) "MD simulation of water using a rigid body description requires a small time step to ensure equipartition"
- Asthagiri et al. (2025) "Consequences of the failure of equipartition for the p–V behavior of liquid water"

**Implementation Frameworks:**
- Leimkuhler & Matthews (2016) Geodesic BAOAB with constraints (102 citations)
- Lelièvre, Rousset & Stoltz (2012) Langevin processes with mechanical constraints (280+ citations)
- Schoenholz & Cubuk (2020) JAX-MD: Differentiable physics framework

**Literature Gaps:**
- No published JAX implementation of constraint-aware Langevin (Prolix would be first)
- No quantitative stability analysis (bounds on dt as function of constraint strength)
- OpenMM's ConstrainedLangevinIntegrator design not publicly documented

## Critical Findings

1. **Feedback Loop Mechanism Confirmed**: SETTLE velocity corrections remove KE from constrained DOF → standard thermostat couples to total KE → oscillatory behavior emerges at dt > 0.5fs. This is documented in Asthagiri et al. (2025).

2. **Jacobian-Projection Approach Sound**: Noise covariance = kT * M * P_rigid ensures equipartition on 6D rigid subspace (not 9D atomic). Mathematical foundation from Walter et al. (2011) and Peters & Goga (2014).

3. **No Cautionary Literature**: Exhaustive search (20+ query variations) found zero published warnings about Jacobian-projection approach. Indicates either validated approach (underdocumented) or too specialized/recent for counterexamples.

4. **Prolix Already Implements Core Pattern**: Code has `project_ou_momentum_rigid=True` with `projection_site="post_o"`. This matches literature-recommended approach.

5. **Alternative Algorithms Insufficient**: LINCS and CCMA improve constraint-solving efficiency but do not address thermostat decoupling. Thallmair et al. (2021) showed constraint looseness still creates temperature gradients across algorithms.

## Recommendations for v0.3.0 Design

**Phase 1 (Theory Derivation):**
- Formally derive Jacobian J ∈ ℝ^(9×6) for rigid TIP3P (3 atoms, 6 rigid DOF)
- Verify noise covariance = kT * M * P_rigid matches equipartition on 6D
- Document why dt=1.0fs should work and 2.0fs should fail (stability analysis)
- Reference Zhang et al. (2019), Peters & Goga (2014) for mathematical foundation

**Phase 2 (Implementation):**
- Implement in JAX with explicit Jacobian projection
- Start with single-water test case (3 atoms) before scaling
- Use recent c-BAOAB literature (2025) to evaluate symplectic structure concerns from Phase 2B

**Phase 2 (Validation):**
- Benchmark against Asthagiri & Beck (2023): simulate rigid water at dt=1.0fs and verify equipartition stability over 50+ ps
- Compare to dt=0.5fs baseline (current v1.0)
- If equipartition stable at 1.0fs, constraint-aware approach succeeds

## Full Research Report

See .praxia/research/260423_constraint_thermostat_sources.tar.zst for complete findings with all 25 citations and detailed conflict/gap analysis.
