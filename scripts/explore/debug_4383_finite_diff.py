#!/usr/bin/env python3
"""Ephemeral diagnostic for #4383 force-parity failure: finite-diff vs jax.grad.

Not a tracked deliverable -- throwaway debugging per systematic-debugging
discipline (isolate whether jax.grad matches prolix's own energy function's
numerical derivative before assuming gold is wrong or grad is broken).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "experiments"))

from nl_omm_parity import _bundle_from_1vii_gold  # noqa: E402

from prolix.api.bundle_md import energy_fn_from_bundle  # noqa: E402
from prolix.physics import neighbor_list as nl  # noqa: E402
from prolix.physics.pbc import create_periodic_space  # noqa: E402

gold = json.loads((_REPO / "data" / "oracles" / "openmm_8.3.1" / "nl_1vii.json").read_text())
bundle = _bundle_from_1vii_gold(gold)

displacement_fn, _ = create_periodic_space(jnp.diag(bundle.box))
neighbor_fn = nl.make_neighbor_list_fn(displacement_fn, jnp.diag(bundle.box), float(bundle.cutoff_distance))
nbr0 = neighbor_fn.allocate(bundle.positions)
nbr = neighbor_fn.update(bundle.positions, nbr0)
print(f"overflow={bool(nbr.did_buffer_overflow)}")

energy_fn = energy_fn_from_bundle(
    bundle, include_nonbonded=True, lj_switch_width=1.0,
    pme_grid_points=int(gold["pme_grid_points"]),
)

e0 = float(energy_fn(bundle.positions, neighbor=nbr))
print(f"e0 = {e0:.6f} kcal/mol (gold = {gold['energy_kcal']:.6f})")

grad_all = jax.grad(energy_fn)(bundle.positions, neighbor=nbr)
f_grad = -np.asarray(grad_all)

gold_f = np.asarray(gold["forces_kcal_mol_A"], dtype=np.float64)
dev = np.sqrt(np.sum((f_grad - gold_f) ** 2, axis=1))
worst = np.argsort(-dev)[:8]
print("atoms with largest |f_prolix(grad) - f_gold| deviation:")
for a in worst:
    print(f"  atom {a}: dev={dev[a]:.4f}  f_grad={f_grad[a]}  f_gold={gold_f[a]}  "
          f"charge={gold['charges'][a]:.4f} sigma={gold['sigmas_A'][a]:.4f} eps={gold['epsilons_kcal'][a]:.6f}")

eps = 1e-4  # Angstrom, central difference
print(f"\nFinite-difference check (central diff, eps={eps} A) for atoms: {list(worst[:4])}")
for a in worst[:4]:
    fd = np.zeros(3)
    for dim in range(3):
        pos_plus = np.array(bundle.positions)
        pos_plus[a, dim] += eps
        pos_minus = np.array(bundle.positions)
        pos_minus[a, dim] -= eps
        nbr_p = neighbor_fn.update(jnp.asarray(pos_plus), nbr0)
        nbr_m = neighbor_fn.update(jnp.asarray(pos_minus), nbr0)
        e_plus = float(energy_fn(jnp.asarray(pos_plus), neighbor=nbr_p))
        e_minus = float(energy_fn(jnp.asarray(pos_minus), neighbor=nbr_m))
        fd[dim] = -(e_plus - e_minus) / (2 * eps)
    print(f"  atom {a}: f_finite_diff={fd}  f_jax_grad={f_grad[a]}  f_gold={gold_f[a]}")

print("\nDone.")
