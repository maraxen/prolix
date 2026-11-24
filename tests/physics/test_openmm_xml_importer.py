"""Tests for OpenMM <System> XML importer and unified 1-4 nonbonded exceptions (FR-6)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from prolix.batched_energy import (
    _exception_energy_masked,
    single_padded_energy,
    single_padded_force,
)
from prolix.openmm_xml_importer import (
    BAR_TO_ATM,
    KJ_NM2_TO_KCAL_A2,
    KJ_TO_KCAL,
    NM_TO_ANGSTROM,
    bundle_from_openmm_system_xml,
    parse_openmm_system_xml,
    physics_system_from_openmm_xml,
)
from prolix.physics.pbc import create_periodic_space
from prolix.types.bundles import CMAP_MAP_BUCKETS


def _make_cmap_grid_str(size: int = 24, base_kj: float = 4.184) -> str:
    """Create a smooth periodic 24x24 CMAP grid string in kJ/mol."""
    angles = np.linspace(0.0, 2.0 * np.pi, size, endpoint=False)
    phi, psi = np.meshgrid(angles, angles, indexing="ij")
    grid_kj = base_kj * (0.5 * np.cos(phi) + 0.25 * np.sin(psi))
    # OpenMM stores energy[i + size * j] where i=phi, j=psi -> flatten in F order
    flat = grid_kj.flatten(order="F")
    return " ".join(f"{val:.6f}" for val in flat)


SYNTHETIC_OPENMM_XML_ORTHO = f"""<?xml version="1.0" ?>
<System openmmVersion="8.3.1" type="System" version="1">
  <PeriodicBoxVectors>
    <A x="3.2" y="0.0" z="0.0"/>
    <B x="0.0" y="3.4" z="0.0"/>
    <C x="0.0" y="0.0" z="3.6"/>
  </PeriodicBoxVectors>
  <Particles>
    <!-- 0..4: 5-atom peptide/proline-like backbone + 1-4 ring fragment -->
    <Particle mass="12.011"/>
    <Particle mass="14.007"/>
    <Particle mass="12.011"/>
    <Particle mass="12.011"/>
    <Particle mass="14.007"/>
    <!-- 5..6: 2-atom GAFF2/OpenFF cosolvent fragment with X-H constraint -->
    <Particle mass="12.011"/>
    <Particle mass="1.008"/>
    <!-- 7..9: 3-site TIP3P-FB water molecule (O, H1, H2) -->
    <Particle mass="15.999"/>
    <Particle mass="1.008"/>
    <Particle mass="1.008"/>
  </Particles>
  <Constraints>
    <!-- Solute X-H holonomic bond constraint (atoms 5-6, d=0.109 nm -> 1.09 A) -->
    <Constraint p1="5" p2="6" d="0.109"/>
    <!-- 3-site rigid water triangle (atoms 7, 8, 9; r_OH=0.10118 nm, r_HH=0.163876 nm) -->
    <Constraint p1="7" p2="8" d="0.10118"/>
    <Constraint p1="7" p2="9" d="0.10118"/>
    <Constraint p1="8" p2="9" d="0.163876"/>
  </Constraints>
  <Forces>
    <Force type="HarmonicBondForce" version="2">
      <Bonds>
        <Bond p1="0" p2="1" d="0.1335" k="410032.0"/>
        <Bond p1="1" p2="2" d="0.1449" k="259408.0"/>
        <Bond p1="2" p2="3" d="0.1522" k="265265.6"/>
        <Bond p1="3" p2="4" d="0.1335" k="410032.0"/>
      </Bonds>
    </Force>
    <Force type="HarmonicAngleForce" version="2">
      <Angles>
        <Angle p1="0" p2="1" p3="2" a="2.124066" k="418.4"/>
        <Angle p1="1" p2="2" p3="3" a="1.937315" k="527.184"/>
        <Angle p1="2" p2="3" p3="4" a="2.035054" k="669.44"/>
      </Angles>
    </Force>
    <Force type="PeriodicTorsionForce" version="2">
      <Torsions>
        <Torsion p1="0" p2="1" p3="2" p4="3" periodicity="3" phase="0.0" k="4.184"/>
        <Torsion p1="1" p2="2" p3="3" p4="4" periodicity="2" phase="3.14159265" k="10.46"/>
      </Torsions>
    </Force>
    <Force type="CMAPTorsionForce" version="1">
      <Maps>
        <Map size="24">
          {_make_cmap_grid_str(24, 4.184)}
        </Map>
      </Maps>
      <Torsions>
        <Torsion map="0" a1="0" a2="1" a3="2" a4="3" b1="1" b2="2" b3="3" b4="4"/>
      </Torsions>
    </Force>
    <Force type="NonbondedForce" method="4" cutoff="0.9" useSwitchingFunction="1" switchingDistance="0.8" useDispersionCorrection="1" alpha="3.4" rfDielectric="78.3" version="4">
      <Particles>
        <Particle q="0.5973" sig="0.339967" eps="0.359824"/>
        <Particle q="-0.4157" sig="0.325000" eps="0.711280"/>
        <Particle q="0.0337" sig="0.339967" eps="0.457730"/>
        <Particle q="0.5973" sig="0.339967" eps="0.359824"/>
        <Particle q="-0.4157" sig="0.325000" eps="0.711280"/>
        <!-- GAFF2/OpenFF cosolvent partial charges -->
        <Particle q="-0.1825" sig="0.339967" eps="0.457730"/>
        <Particle q="0.1825" sig="0.264953" eps="0.065689"/>
        <!-- TIP3P-FB water -->
        <Particle q="-0.84844" sig="0.317796" eps="0.652140"/>
        <Particle q="0.42422" sig="0.100000" eps="0.000000"/>
        <Particle q="0.42422" sig="0.100000" eps="0.000000"/>
      </Particles>
      <Exceptions>
        <!-- 1-2 and 1-3 zero exclusions -->
        <Exception p1="0" p2="1" q="0.0" sig="0.332484" eps="0.0"/>
        <Exception p1="1" p2="2" q="0.0" sig="0.332484" eps="0.0"/>
        <Exception p1="2" p2="3" q="0.0" sig="0.339967" eps="0.0"/>
        <Exception p1="3" p2="4" q="0.0" sig="0.332484" eps="0.0"/>
        <Exception p1="0" p2="2" q="0.0" sig="0.339967" eps="0.0"/>
        <Exception p1="1" p2="3" q="0.0" sig="0.332484" eps="0.0"/>
        <Exception p1="2" p2="4" q="0.0" sig="0.332484" eps="0.0"/>
        <Exception p1="5" p2="6" q="0.0" sig="0.302460" eps="0.0"/>
        <Exception p1="7" p2="8" q="0.0" sig="0.208898" eps="0.0"/>
        <Exception p1="7" p2="9" q="0.0" sig="0.208898" eps="0.0"/>
        <Exception p1="8" p2="9" q="0.0" sig="0.100000" eps="0.0"/>
        <!-- 1-4 scaled exceptions (standard AMBER 1-4 + proline ring override) -->
        <Exception p1="0" p2="3" q="0.297306" sig="0.339967" eps="0.179912"/>
        <Exception p1="1" p2="4" q="0.144005" sig="0.325000" eps="0.355640"/>
      </Exceptions>
    </Force>
    <Force type="GBSAOBCForce" soluteDielectric="1.0" solventDielectric="78.5" surfaceAreaEnergy="2.25936" version="1">
      <Particles>
        <Particle q="0.5973" radius="0.170" scale="0.72"/>
        <Particle q="-0.4157" radius="0.155" scale="0.79"/>
        <Particle q="0.0337" radius="0.170" scale="0.72"/>
        <Particle q="0.5973" radius="0.170" scale="0.72"/>
        <Particle q="-0.4157" radius="0.155" scale="0.79"/>
        <Particle q="-0.1825" radius="0.170" scale="0.72"/>
        <Particle q="0.1825" radius="0.120" scale="0.85"/>
        <Particle q="-0.84844" radius="0.150" scale="0.85"/>
        <Particle q="0.42422" radius="0.120" scale="0.85"/>
        <Particle q="0.42422" radius="0.120" scale="0.85"/>
      </Particles>
    </Force>
    <Force type="MonteCarloBarostat" defaultPressure="1.01325" defaultTemperature="310.0" frequency="25" version="1"/>
  </Forces>
</System>
"""


SYNTHETIC_OPENMM_XML_TRICLINIC = """<?xml version="1.0" ?>
<System openmmVersion="8.3.1" type="System" version="1">
  <PeriodicBoxVectors>
    <A x="4.0" y="0.0" z="0.0"/>
    <B x="2.0" y="3.4641016" z="0.0"/>
    <C x="2.0" y="1.1547005" z="3.2659863"/>
  </PeriodicBoxVectors>
  <Particles>
    <Particle mass="12.011"/>
    <Particle mass="12.011"/>
  </Particles>
  <Forces>
    <Force type="NonbondedForce" method="4" cutoff="0.9" alpha="3.4" version="4">
      <Particles>
        <Particle q="0.25" sig="0.34" eps="0.4184"/>
        <Particle q="-0.25" sig="0.34" eps="0.4184"/>
      </Particles>
      <Exceptions>
        <Exception p1="0" p2="1" q="-0.0520833" sig="0.34" eps="0.2092"/>
      </Exceptions>
    </Force>
  </Forces>
</System>
"""


def test_openmm_xml_unit_conversions_and_fields():
    """Verify exact OpenMM (nm, kJ/mol, bar) -> Prolix AKMA (A, kcal/mol, atm) conversions."""
    data = parse_openmm_system_xml(SYNTHETIC_OPENMM_XML_ORTHO)
    bundle = bundle_from_openmm_system_xml(SYNTHETIC_OPENMM_XML_ORTHO)

    # 1. Particles
    assert data.n_atoms == 10
    assert int(bundle.n_atoms) == 10
    np.testing.assert_allclose(
        data.masses[:7],
        [12.011, 14.007, 12.011, 12.011, 14.007, 12.011, 1.008],
        rtol=1e-6,
    )

    # 2. PeriodicBoxVectors (orthorhombic: 3.2, 3.4, 3.6 nm -> 32, 34, 36 A)
    assert data.has_pbc is True
    assert data.is_triclinic is False
    assert bundle.shape_spec.has_pbc is True
    assert bundle.shape_spec.is_triclinic is False
    np.testing.assert_allclose(jnp.diag(bundle.box), [32.0, 34.0, 36.0], rtol=1e-6)

    # 3. Constraints: 1 solute constraint (5, 6) at 1.09 A + 1 rigid water triangle (7, 8, 9)
    assert int(bundle.n_waters) == 1
    assert bundle.shape_spec.has_real_water is True
    np.testing.assert_array_equal(bundle.water_indices[0], [7, 8, 9])
    np.testing.assert_allclose(float(bundle.water_r_OH), 0.10118 * NM_TO_ANGSTROM, rtol=1e-6)
    np.testing.assert_allclose(float(bundle.water_r_HH), 0.163876 * NM_TO_ANGSTROM, rtol=1e-6)

    assert int(bundle.n_constraints) == 1
    assert bundle.shape_spec.has_real_constraints is True
    np.testing.assert_array_equal(bundle.constraint_pairs[0], [5, 6])
    np.testing.assert_allclose(float(bundle.constraint_lengths[0]), 0.109 * NM_TO_ANGSTROM, rtol=1e-6)

    # 4. HarmonicBondForce: r0 = d_nm * 10, k = k_OpenMM / 418.4
    assert int(bundle.n_bonds) == 4
    np.testing.assert_array_equal(bundle.bond_idx[0], [0, 1])
    np.testing.assert_allclose(
        bundle.bond_params[0],
        [0.1335 * NM_TO_ANGSTROM, 410032.0 * KJ_NM2_TO_KCAL_A2],
        rtol=1e-6,
    )

    # 5. HarmonicAngleForce: theta0 in rad, k = k_OpenMM / 4.184
    assert int(bundle.n_angles) == 3
    np.testing.assert_array_equal(bundle.angle_idx[0], [0, 1, 2])
    np.testing.assert_allclose(
        bundle.angle_params[0],
        [2.124066, 418.4 * KJ_TO_KCAL],
        rtol=1e-6,
    )

    # 6. PeriodicTorsionForce: [periodicity, phase, k_OpenMM / 4.184]
    assert int(bundle.n_dihedrals) == 2
    np.testing.assert_array_equal(bundle.dihedral_idx[0], [0, 1, 2, 3])
    np.testing.assert_allclose(
        bundle.dihedral_params[0],
        [3.0, 0.0, 4.184 * KJ_TO_KCAL],
        rtol=1e-6,
    )

    # 7. CMAPTorsionForce: 8-atom torsion indices & precomputed cmap_coeffs padded to CMAP_MAP_BUCKETS
    assert int(bundle.n_cmap) == 1
    np.testing.assert_array_equal(bundle.cmap_torsion_idx[0], [0, 1, 2, 3, 1, 2, 3, 4])
    assert int(bundle.cmap_indices[0]) == 0
    assert bundle.cmap_coeffs is not None
    assert bundle.cmap_coeffs.shape == (CMAP_MAP_BUCKETS[0], 24, 24, 16)
    assert jnp.all(jnp.isfinite(bundle.cmap_coeffs))
    # Verify grid energy was converted from kJ/mol to kcal/mol (base_kj=4.184 -> max 0.75 kcal/mol)
    assert float(jnp.max(jnp.abs(data.cmap_energy_grids))) <= 0.75 + 1e-4

    # 8. NonbondedForce & unified 1-4 exceptions
    np.testing.assert_allclose(bundle.charges[0], 0.5973, rtol=1e-6)
    np.testing.assert_allclose(bundle.sigmas[0], 0.339967 * NM_TO_ANGSTROM, rtol=1e-6)
    np.testing.assert_allclose(bundle.epsilons[0], 0.359824 * KJ_TO_KCAL, rtol=1e-6)
    np.testing.assert_allclose(float(bundle.pme_alpha), 3.4 / NM_TO_ANGSTROM, rtol=1e-6)
    np.testing.assert_allclose(float(bundle.cutoff_distance), 0.9 * NM_TO_ANGSTROM, rtol=1e-6)
    np.testing.assert_allclose(data.lj_switch_width, 1.0, rtol=1e-6)

    # 11 zero exclusions + 2 nonzero 1-4 exceptions = 13 total excluded pairs (all with 0.0 scale)
    assert int(bundle.n_excl) == 13
    np.testing.assert_allclose(bundle.excl_scales_vdw[:13], 0.0, atol=1e-7)
    np.testing.assert_allclose(bundle.excl_scales_elec[:13], 0.0, atol=1e-7)

    # 2 nonzero 1-4 exceptions in exception_* arrays
    assert int(bundle.n_exception_pairs) == 2
    np.testing.assert_array_equal(bundle.exception_pairs[:2], [[0, 3], [1, 4]])
    np.testing.assert_allclose(bundle.exception_chargeprods[:2], [0.297306, 0.144005], rtol=1e-6)
    np.testing.assert_allclose(
        bundle.exception_sigmas[:2],
        [0.339967 * NM_TO_ANGSTROM, 0.325000 * NM_TO_ANGSTROM],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        bundle.exception_epsilons[:2],
        [0.179912 * KJ_TO_KCAL, 0.355640 * KJ_TO_KCAL],
        rtol=1e-6,
    )

    # 9. GBSAOBCForce: radii in A, scaled_radii = radii * scale
    assert data.has_implicit_solvent is True
    assert bundle.shape_spec.has_implicit_solvent is True
    np.testing.assert_allclose(bundle.radii[0], 0.170 * NM_TO_ANGSTROM, rtol=1e-6)
    np.testing.assert_allclose(
        bundle.scaled_radii[0], 0.170 * NM_TO_ANGSTROM * 0.72, rtol=1e-6
    )
    np.testing.assert_allclose(data.solute_dielectric, 1.0, rtol=1e-6)
    np.testing.assert_allclose(data.solvent_dielectric, 78.5, rtol=1e-6)
    np.testing.assert_allclose(
        data.surface_area_energy, 2.25936 * KJ_NM2_TO_KCAL_A2, rtol=1e-6
    )

    # 10. MonteCarloBarostat
    assert data.barostat is not None
    np.testing.assert_allclose(data.barostat.default_pressure_bar, 1.01325, rtol=1e-6)
    np.testing.assert_allclose(
        data.barostat.default_pressure_atm, 1.01325 * BAR_TO_ATM, rtol=1e-6
    )
    np.testing.assert_allclose(data.barostat.default_temperature_k, 310.0, rtol=1e-6)
    assert data.barostat.frequency == 25


def test_openmm_xml_triclinic_box_reduction():
    """Verify triclinic box vectors in nm are converted to A and reduced to lower-triangular form."""
    bundle = bundle_from_openmm_system_xml(SYNTHETIC_OPENMM_XML_TRICLINIC)
    assert bundle.shape_spec.has_pbc is True
    assert bundle.shape_spec.is_triclinic is True

    # Upper triangle must be strictly zero
    assert float(jnp.abs(bundle.box[0, 1])) == 0.0
    assert float(jnp.abs(bundle.box[0, 2])) == 0.0
    assert float(jnp.abs(bundle.box[1, 2])) == 0.0
    # OpenMM reduced basis conditions: |b_x| <= 0.5*a_x, |c_x| <= 0.5*a_x, |c_y| <= 0.5*b_y
    ax = float(bundle.box[0, 0])
    bx = float(bundle.box[1, 0])
    by = float(bundle.box[1, 1])
    cx = float(bundle.box[2, 0])
    cy = float(bundle.box[2, 1])
    cz = float(bundle.box[2, 2])
    assert ax == 40.0
    assert by > 0.0 and cz > 0.0
    assert abs(bx) <= 0.5 * ax + 1e-5
    assert abs(cx) <= 0.5 * ax + 1e-5
    assert abs(cy) <= 0.5 * by + 1e-5


def test_single_padded_energy_and_force_14_exception_parity():
    """Verify single_padded_energy and single_padded_force match analytical 1-4 exception energy & forces."""
    # Construct a 2-atom system interacting solely via a 1-4 exception pair
    # (so bulk LJ/Coulomb are excluded with 0.0 scale and only the 1-4 exception contributes).
    xml_14_only = """<?xml version="1.0" ?>
    <System openmmVersion="8.3.1" type="System" version="1">
      <Particles>
        <Particle mass="12.011"/>
        <Particle mass="14.007"/>
      </Particles>
      <Forces>
        <Force type="NonbondedForce" method="0" cutoff="0.9" useDispersionCorrection="0" version="4">
          <Particles>
            <Particle q="0.50" sig="0.34" eps="0.4184"/>
            <Particle q="-0.40" sig="0.32" eps="0.8368"/>
          </Particles>
          <Exceptions>
            <!-- Proline / OpenFF custom 1-4 exception override -->
            <Exception p1="0" p2="1" q="-0.15" sig="0.33" eps="0.2092"/>
          </Exceptions>
        </Force>
      </Forces>
    </System>
    """
    positions = jnp.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=jnp.float32)
    sys = physics_system_from_openmm_xml(xml_14_only, positions=positions)
    disp_fn, _ = space.free()

    # Analytical 1-4 energy at r = 3.5 A:
    r_dist = 3.5
    sig_14 = 0.33 * 10.0  # 3.3 A
    eps_14 = 0.2092 / 4.184  # 0.05 kcal/mol
    qprod_14 = -0.15
    sr6 = (sig_14 / r_dist) ** 6
    expected_e_lj = 4.0 * eps_14 * (sr6 * sr6 - sr6)
    expected_e_coul = 332.0637 * qprod_14 / r_dist
    expected_e_total = expected_e_lj + expected_e_coul

    e_actual = single_padded_energy(
        sys, disp_fn, implicit_solvent=False, use_dispersion_correction=False
    )
    assert jnp.isfinite(e_actual)
    np.testing.assert_allclose(float(e_actual), expected_e_total, rtol=1e-5)

    # Verify forces match -grad(single_padded_energy)
    f_autodiff = -jax.grad(
        lambda pos: single_padded_energy(
            sys.replace(positions=pos),
            disp_fn,
            implicit_solvent=False,
            use_dispersion_correction=False,
        )
    )(sys.positions)
    assert jnp.all(jnp.isfinite(f_autodiff))

    # Analytical force magnitude along x on atom 1 (at x = +3.5 A):
    # dE/dr = 4*eps*(-12*sig^12/r^13 + 6*sig^6/r^7) - 332.0637*qprod/r^2
    # F_1x = -dE/dr
    de_dr = (
        4.0 * eps_14 * (-12.0 * (sig_14**12) / (r_dist**13) + 6.0 * (sig_14**6) / (r_dist**7))
        - 332.0637 * qprod_14 / (r_dist**2)
    )
    expected_f1_x = -de_dr
    np.testing.assert_allclose(float(f_autodiff[1, 0]), expected_f1_x, rtol=1e-5)
    np.testing.assert_allclose(float(f_autodiff[0, 0]), -expected_f1_x, rtol=1e-5)

    # Also test full synthetic system energy and forces are finite
    pos_10 = jnp.array(
        [
            [5.0, 5.0, 5.0],
            [6.3, 5.0, 5.0],
            [7.0, 6.2, 5.0],
            [8.4, 6.2, 5.3],
            [9.2, 7.3, 5.3],
            [15.0, 15.0, 15.0],
            [16.09, 15.0, 15.0],
            [22.0, 22.0, 22.0],
            [22.96, 22.0, 22.0],
            [21.70, 22.93, 22.0],
        ],
        dtype=jnp.float32,
    )
    sys_full = physics_system_from_openmm_xml(SYNTHETIC_OPENMM_XML_ORTHO, positions=pos_10)
    disp_pbc, _ = create_periodic_space(jnp.array([32.0, 34.0, 36.0], dtype=jnp.float32))
    e_full = single_padded_energy(sys_full, disp_pbc, implicit_solvent=True)
    f_full = single_padded_force(
        sys_full, disp_pbc, implicit_solvent=True, explicit_solvent=False, use_flash=False
    )
    assert jnp.isfinite(e_full)
    assert jnp.all(jnp.isfinite(f_full))

    # Verify _exception_energy_masked matches expected 1-4 pair energy on full system
    e_exc_full = _exception_energy_masked(
        sys_full.positions,
        sys_full.exception_pairs,
        sys_full.exception_sigmas,
        sys_full.exception_epsilons,
        sys_full.exception_chargeprods,
        sys_full.exception_mask,
        disp_pbc,
    )
    assert jnp.isfinite(e_exc_full)
    assert float(e_exc_full) != 0.0
