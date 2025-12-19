#!/usr/bin/env python3
"""Update explicit_solvent_colab.ipynb with TPU-friendly parameters."""

import json
from pathlib import Path

notebook_path = Path("notebooks/explicit_solvent_colab.ipynb")

# Load notebook
with open(notebook_path) as f:
    nb = json.load(f)

# Find and update the simulation cell
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and any("step_size_fs=" in line for line in cell.get("source", [])):
        # Update source
        new_source = [
            "# Run explicit solvent simulation (50 ps)\n",
            "positions = jnp.array(atom_array.coord)\n",
            "key = random.PRNGKey(42)\n",
            "\n",
            "# NOTE: TPU uses float32. For numerical stability with PME:\n",
            "#  - Use conservative timestep (1.0 fs instead of 2.0 fs)\n",
            "#  - Ensure adequate PME grid resolution (64 instead of 48)\n",
            "#  - Minimization convergence is monitored automatically\n",
            "\n",
            "spec = simulate.SimulationSpec(\n",
            "    total_time_ns=0.05,  # 50 ps\n",
            "    step_size_fs=1.0,  # Conservative for TPU float32 + PME\n",
            "    save_interval_ns=0.001,\n",
            '    save_path="1uao_explicit_traj.array_record",\n',
            "    temperature_k=300.0,\n",
            "    gamma=1.0,\n",
            "    box=box_size,\n",
            "    use_pbc=True,\n",
            "    pme_grid_size=64  # Increased for better accuracy\n",
            ")\n",
            "\n",
            'print("Running 50ps explicit solvent simulation...")\n',
            "final_state = simulate.run_simulation(\n",
            "    system_params=system_params,\n",
            "    r_init=positions,\n",
            "    spec=spec,\n",
            "    key=key\n",
            ")\n",
            'print(f"Done! Final energy: {final_state.potential_energy:.2f} kcal/mol")'
        ]
        cell["source"] = new_source
        print("✓ Updated simulation cell with TPU-friendly parameters")
        break

# Save updated notebook
with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=4)

print(f"✓ Saved {notebook_path}")
