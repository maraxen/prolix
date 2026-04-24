"""Sprint 2 Phase 5: Ablation studies for EFA hyperparameters.

Evaluates force accuracy as a function of D (feature count) and alpha (damping).
Also measures energy variance under omega resampling.

Outputs to results/ablation_efa.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from prolix.physics.electrostatic_methods import ElectrostaticMethod
from prolix.physics.eval_harness import (
    force_rmse,
    make_comparison_energies,
    make_tip3p_water_system,
)
from prolix.physics.flash_explicit import flash_explicit_energy


def ablate_d_sweep():
    """Ablate feature count D: compute relative RMSE for D in [64, 128, 256, 512, 1024].

    Returns:
        List of dicts with D, relative_rmse, and metadata.
    """
    system = make_tip3p_water_system(n_waters=32, seed=0)
    comparison_pme = make_comparison_energies(system, n_seeds=1, n_features=512)
    f_pme = comparison_pme["pme_forces"]
    mask = system.atom_mask.astype(bool)
    pme_rms = float(jnp.sqrt(jnp.mean(f_pme[mask] ** 2)))

    d_values = [64, 128, 256, 512, 1024]
    results = []

    for D in d_values:
        print(f"  D={D}...", end="", flush=True)
        comparison = make_comparison_energies(system, n_seeds=8, n_features=D)
        f_efa_seeds = comparison["efa_forces_per_seed"]

        rmse_per_seed = jnp.array(
            [force_rmse(f, f_pme, mask) for f in f_efa_seeds]
        )
        mean_rmse = float(jnp.mean(rmse_per_seed))
        std_rmse = float(jnp.std(rmse_per_seed))
        relative_rmse = mean_rmse / (pme_rms + 1e-12)

        print(f" relative_rmse={relative_rmse:.4f}")

        results.append({
            "ablation_type": "d_sweep",
            "n_features": D,
            "mean_rmse_kcal_mol_a": mean_rmse,
            "std_rmse": std_rmse,
            "relative_rmse": relative_rmse,
            "pme_rms_reference": pme_rms,
        })

    return results


def ablate_alpha_sweep():
    """Ablate damping parameter alpha: compute relative RMSE for alpha in [0.2, 0.34, 0.5].

    Uses fixed n_features=512.

    Returns:
        List of dicts with alpha, relative_rmse, and metadata.
    """
    # Note: This test would require modifying the system's pme_alpha parameter
    # and re-creating the PaddedSystem. For now, we skip this ablation as it
    # requires deeper integration with the system building pipeline.
    # A full implementation would:
    # 1. Create systems with different pme_alpha values
    # 2. Compute EFA forces for each alpha
    # 3. Compare to PME baseline
    # 4. Report relative RMSE vs alpha
    return []


def ablate_omega_resampling_variance():
    """Ablate omega resampling: measure energy variance under fixed vs resampled omega.

    Returns:
        Dict with variance ratio and per-method statistics.
    """
    system = make_tip3p_water_system(n_waters=32, seed=0)
    fixed_seed = 42

    print("  Fixed omega (200 evals)...", end="", flush=True)
    fixed_energies = []
    for _ in range(200):
        e = flash_explicit_energy(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=512,
            rff_seed=fixed_seed,
        )
        fixed_energies.append(float(e))

    fixed_var = float(jnp.var(jnp.array(fixed_energies)))
    fixed_mean = float(jnp.mean(jnp.array(fixed_energies)))
    fixed_std = float(jnp.std(jnp.array(fixed_energies)))
    print(f" var={fixed_var:.6f}")

    print("  Resampled omega (200 evals)...", end="", flush=True)
    resampled_energies = []
    for k in range(200):
        e = flash_explicit_energy(
            system,
            electrostatic_method=ElectrostaticMethod.EFA,
            n_rff_features=512,
            rff_seed=k,
        )
        resampled_energies.append(float(e))

    resampled_var = float(jnp.var(jnp.array(resampled_energies)))
    resampled_mean = float(jnp.mean(jnp.array(resampled_energies)))
    resampled_std = float(jnp.std(jnp.array(resampled_energies)))
    print(f" var={resampled_var:.6f}")

    ratio = resampled_var / (fixed_var + 1e-12)

    return {
        "ablation_type": "omega_resampling",
        "n_features": 512,
        "fixed_omega": {
            "mean_energy": fixed_mean,
            "std_energy": fixed_std,
            "variance": fixed_var,
        },
        "resampled_omega": {
            "mean_energy": resampled_mean,
            "std_energy": resampled_std,
            "variance": resampled_var,
        },
        "variance_ratio_resampled_over_fixed": ratio,
    }


def main():
    jax.config.update("jax_enable_x64", True)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "ablations": [],
    }

    print("\n=== Ablation 1: D sweep (feature count) ===")
    d_results = ablate_d_sweep()
    results["ablations"].extend(d_results)

    print("\n=== Ablation 2: Alpha sweep (damping) ===")
    # Skipped for now; would require system re-creation with different alphas
    print("  (Skipped: requires deeper system integration)")

    print("\n=== Ablation 3: Omega resampling variance ===")
    omega_result = ablate_omega_resampling_variance()
    results["ablations"].append(omega_result)

    # Write results
    output_path = Path("results/ablation_efa.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
