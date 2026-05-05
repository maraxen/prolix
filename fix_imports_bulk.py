import os
import re

# Rules from user:
# LOCAL:
# prolix.batched_energy, prolix.batched_simulate, prolix.types, prolix.physics.system, prolix.physics.pme
# prolix.padding, prolix.export, prolix.fused_energy, prolix.analysis, prolix.compat, prolix.constants
# prolix.pallas_kernels, prolix.resource_guard, prolix.simulate
# prolix.physics.neighbor_list, prolix.physics.settle, prolix.physics.barostat, prolix.physics.pbc
# prolix.physics.units, prolix.physics.pressure, prolix.physics.stress, prolix.physics.integrator_builder
# prolix.physics.step_system, prolix.physics.noising, prolix.physics.generalized_born, prolix.physics.bonded
# prolix.physics.cell_list, prolix.physics.cell_nonbonded, prolix.physics.cmap, prolix.physics.constraints
# prolix.physics.efa_coulomb, prolix.physics.electrostatic_methods, prolix.physics.eval_harness
# prolix.physics.explicit_corrections, prolix.physics.flash_explicit, prolix.physics.flash_nonbonded
# prolix.physics.hmr, prolix.physics.ion_params, prolix.physics.kups_adapter, prolix.physics.md_potential_bundle
# prolix.physics.mtt_logdet, prolix.physics.optimization, prolix.physics.pure_utils, prolix.physics.rff_coulomb
# prolix.physics.rigid_water_ke, prolix.physics.sasa, prolix.physics.sharding, prolix.physics.solvation
# prolix.physics.spec, prolix.physics.step_result, prolix.physics.tiling, prolix.physics.topology_merger
# prolix.physics.virtual_sites, prolix.physics.virtual_sites_step, prolix.physics.water_models

# REMOTE:
# proxide.core.containers
# proxide.physics.constants
# proxide.physics.electrostatics
# proxide.physics.force_fields
# proxide.physics.projections
# proxide.physics.vdw
# proxide.physics.features
# proxide.io
# proxide.chem
# proxide.geometry
# proxide.ops
# proxide.cli

local_top_level = [
    'batched_energy', 'batched_simulate', 'types', 'padding', 'export', 
    'fused_energy', 'analysis', 'compat', 'constants', 'pallas_kernels',
    'resource_guard', 'simulate', 'utils', 'pt', 'visualization'
]

# Anything in prolix/physics except remote ones
remote_physics = [
    'constants', 'electrostatics', 'force_fields', 'projections', 'vdw', 'features'
]

def fix_content(content):
    # 1. Replace from proxide.<top_level> -> from prolix.<top_level>
    for mod in local_top_level:
        content = re.sub(rf'from proxide\.{mod}\b', f'from prolix.{mod}', content)
        content = re.sub(rf'import proxide\.{mod}\b', f'import prolix.{mod}', content)
    
    # 2. Handle proxide.physics.<local>
    # We can use a negative lookahead to avoid remote_physics
    # But it's easier to just match proxide.physics and then check the submodule
    def replace_physics(match):
        submod = match.group(1)
        if submod in remote_physics:
            return match.group(0)
        return f'prolix.physics.{submod}'

    content = re.sub(r'proxide\.physics\.(\w+)', replace_physics, content)

    # 3. Special cases like 'import proxide' used for prolix things?
    # Actually 'import proxide' is tricky. If it's used for FF_PATH, it should stay proxide.
    
    return content

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            new_content = fix_content(content)
            if new_content != content:
                with open(path, 'w') as f:
                    f.write(new_content)
                print(f"Fixed {path}")

for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            new_content = fix_content(content)
            if new_content != content:
                with open(path, 'w') as f:
                    f.write(new_content)
                print(f"Fixed {path}")
