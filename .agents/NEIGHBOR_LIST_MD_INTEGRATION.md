# Neighbor List Integration in MD Loop

## Status: ✅ CONFIRMED

The neighbor list integration is stable and used by default in production-ready integrators.

## Implementation Summary

- **Integrators**: Langevin integrator in `prolix.physics.simulate` uses JAX-MD neighbor lists.
- **Overflow Handling**: Correctly detects and reallocates when neighbors exceed capacity.
- **Energy Functions**: Neighbor-list-aware energy functions are passed to JIT-compiled loops.

## Current State

- Used for both minimization and dynamics.
- Drastically improves O(N^2) scaling for large solvated systems.
