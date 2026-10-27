"""OpenMM System XML importer with unified 1-4 nonbonded exceptions and zero C++ runtime dependency.

Parses serialized OpenMM ``<System>`` XML files or strings directly via Python's
standard-library ``xml.etree.ElementTree`` into :class:`OpenMMSystemData`,
:class:`~prolix.types.bundles.MolecularBundle`, and :class:`~prolix.typing.PhysicsSystem`.

Unit conversions (OpenMM XML -> Prolix AKMA):
  - Length / distance / sigma / GB radii / box vectors: ``nm -> A`` (``* 10.0``)
  - Energy / LJ epsilon / CMAP grid values: ``kJ/mol -> kcal/mol`` (``/ 4.184``)
  - Harmonic bond force constant: ``kJ/(mol*nm^2) -> kcal/(mol*A^2)`` (``/ 418.4``)
  - Harmonic angle force constant: ``kJ/(mol*rad^2) -> kcal/(mol*rad^2)`` (``/ 4.184``)
  - Periodic torsion force constant: ``kJ/mol -> kcal/mol`` (``/ 4.184``)
  - PME Ewald alpha: ``nm^-1 -> A^-1`` (``/ 10.0``)
  - GBSA surface area energy: ``kJ/(mol*nm^2) -> kcal/(mol*A^2)`` (``/ 418.4``)
  - Barostat pressure: ``bar`` (and ``atm = bar / 1.01325``)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np

from prolix.batched_energy import solve_ewald_alpha
from prolix.physics.cmap import precompute_cmap_coefficients
from prolix.physics.neighbor_list import pair_list_to_dense_padded
from prolix.physics.pbc import reduce_triclinic_box_vectors
from prolix.physics.system import (
    _mask,
    _next_bucket,
    _pad_1d,
    _pad_2d,
    _pad_3d,
    _pad_positions_with_ghost_lattice,
)
from prolix.types.bundles import (
    ANGLE_BUCKETS,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    CMAP_BUCKETS,
    CMAP_MAP_BUCKETS,
    CONSTRAINT_BUCKETS,
    DIHEDRAL_BUCKETS,
    EXCEPTION_BUCKETS,
    EXCL_BUCKETS,
    WATER_BUCKETS,
    MolecularBundle,
    MolecularShapeSpec,
    _bucket_idx,
)
from prolix.typing import ArrayLike, PhysicsSystem

NM_TO_ANGSTROM: float = 10.0
KJ_TO_KCAL: float = 1.0 / 4.184
KJ_NM2_TO_KCAL_A2: float = 1.0 / 418.4
BAR_TO_ATM: float = 1.0 / 1.01325

DEFAULT_TIP3P_R_OH: float = 0.9572
DEFAULT_TIP3P_R_HH: float = 1.5139
_EXCL_DENSE_MAX: int = 32


@dataclass(frozen=True)
class BarostatSpec:
    """Parsed OpenMM MonteCarloBarostat parameters."""

    default_pressure_bar: float
    default_pressure_atm: float
    default_temperature_k: float
    frequency: int


@dataclass
class OpenMMSystemData:
    """Unpadded Prolix AKMA system data extracted from an OpenMM ``<System>`` XML."""

    n_atoms: int
    positions: jnp.ndarray  # (N, 3) float32, A
    masses: jnp.ndarray  # (N,) float32, Da
    charges: jnp.ndarray  # (N,) float32, e
    sigmas: jnp.ndarray  # (N,) float32, A
    epsilons: jnp.ndarray  # (N,) float32, kcal/mol
    radii: jnp.ndarray  # (N,) float32, A
    obc_scale: jnp.ndarray  # (N,) float32, dimensionless
    scaled_radii: jnp.ndarray  # (N,) float32, A (radii * obc_scale)
    box: jnp.ndarray  # (3, 3) float32, A (lower-triangular reduced)
    box_size: jnp.ndarray | None  # (3,) if orthorhombic, (3, 3) if triclinic, None if free
    has_pbc: bool
    is_triclinic: bool
    has_implicit_solvent: bool
    bonds: jnp.ndarray  # (N_bonds, 2) int32
    bond_params: jnp.ndarray  # (N_bonds, 2) float32 [r0_A, k_kcal_mol_A2]
    angles: jnp.ndarray  # (N_angles, 3) int32
    angle_params: jnp.ndarray  # (N_angles, 2) float32 [theta0_rad, k_kcal_mol_rad2]
    dihedrals: jnp.ndarray  # (N_dih, 4) int32
    dihedral_params: jnp.ndarray  # (N_dih, 3) float32 [periodicity, phase_rad, k_kcal_mol]
    impropers: jnp.ndarray  # (0, 4) int32
    improper_params: jnp.ndarray  # (0, 3) float32
    urey_bradley_bonds: jnp.ndarray  # (0, 3) int32
    urey_bradley_params: jnp.ndarray  # (0, 2) float32
    cmap_torsions: jnp.ndarray  # (N_cmap, 8) int32 [a1, a2, a3, a4, b1, b2, b3, b4]
    cmap_indices: jnp.ndarray  # (N_cmap,) int32
    cmap_energy_grids: jnp.ndarray  # (N_maps, G, G) float32, kcal/mol
    cmap_coeffs: jnp.ndarray  # (M_bucket, G, G, 16) float32
    water_indices: jnp.ndarray  # (N_waters, 3) int32 [O, H1, H2]
    water_r_OH: float  # A
    water_r_HH: float  # A
    constraint_pairs: jnp.ndarray  # (N_constraints, 2) int32
    constraint_lengths: jnp.ndarray  # (N_constraints,) float32, A
    excl_indices: jnp.ndarray  # (N_excl, 2) int32 (both 1-2/1-3 and 1-4 pairs)
    excl_scales_vdw: jnp.ndarray  # (N_excl,) float32 (all 0.0)
    excl_scales_elec: jnp.ndarray  # (N_excl,) float32 (all 0.0)
    exception_pairs: jnp.ndarray  # (N_exc14, 2) int32 (nonzero q or eps exceptions)
    exception_chargeprods: jnp.ndarray  # (N_exc14,) float32, e^2
    exception_sigmas: jnp.ndarray  # (N_exc14,) float32, A
    exception_epsilons: jnp.ndarray  # (N_exc14,) float32, kcal/mol
    pme_alpha: float  # A^-1
    nonbonded_cutoff: float  # A
    lj_switch_width: float  # A
    use_dispersion_correction: bool
    electrostatic_method: str
    reaction_field_dielectric: float
    solute_dielectric: float
    solvent_dielectric: float
    surface_area_energy: float  # kcal/(mol*A^2)
    barostat: BarostatSpec | None
    molecule_id: jnp.ndarray  # (N,) int32
    cosolvent_group_id: jnp.ndarray  # (N,) int32


def _load_xml_root(xml_path_or_str: str | Path | ET.Element) -> ET.Element:
    """Resolve an XML string, filesystem path, or Element into an ElementTree root."""
    if isinstance(xml_path_or_str, ET.Element):
        return xml_path_or_str
    if isinstance(xml_path_or_str, Path):
        if not xml_path_or_str.exists():
            raise FileNotFoundError(f"OpenMM XML file not found: {xml_path_or_str}")
        return ET.parse(xml_path_or_str).getroot()
    if isinstance(xml_path_or_str, str):
        stripped = xml_path_or_str.lstrip()
        if stripped.startswith("<"):
            return ET.fromstring(stripped)
        path = Path(xml_path_or_str)
        if not path.exists():
            raise FileNotFoundError(f"OpenMM XML file not found: {path}")
        return ET.parse(path).getroot()
    raise TypeError(f"Unsupported xml_path_or_str type: {type(xml_path_or_str)!r}")


def _parse_pdb_positions(pdb_path_or_str: str | Path) -> jnp.ndarray:
    """Extract (N, 3) atomic coordinates in Angstroms from a PDB file or string."""
    if isinstance(pdb_path_or_str, Path):
        lines = pdb_path_or_str.read_text().splitlines()
    elif isinstance(pdb_path_or_str, str):
        p = Path(pdb_path_or_str)
        if "\n" not in pdb_path_or_str and p.exists():
            lines = p.read_text().splitlines()
        else:
            lines = pdb_path_or_str.splitlines()
    else:
        raise TypeError(f"Unsupported PDB source: {type(pdb_path_or_str)!r}")

    coords: list[list[float]] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    if not coords:
        raise ValueError("No ATOM/HETATM coordinate records found in PDB input.")
    return jnp.asarray(coords, dtype=jnp.float32)


def _find_force_elements(root: ET.Element, force_type: str) -> list[ET.Element]:
    """Find all XML elements matching a given OpenMM Force type or tag name."""
    matches: list[ET.Element] = []
    seen: set[int] = set()
    for elem in root.iter():
        is_match = (
            elem.tag == force_type
            or (
                elem.tag == "Force"
                and (
                    elem.attrib.get("type") == force_type
                    or elem.attrib.get("name") == force_type
                )
            )
        )
        if is_match and id(elem) not in seen:
            seen.add(id(elem))
            matches.append(elem)
    return matches


def _separate_water_and_solute_constraints(
    raw_constraints: list[tuple[int, int, float]],
    masses_np: np.ndarray,
    explicit_water_indices: ArrayLike | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Separate 3-site rigid water triangles from solute holonomic bond constraints.

    Returns:
        (water_indices, water_r_OH, water_r_HH, constraint_pairs, constraint_lengths)
    """
    r_oh_vals: list[float] = []
    r_hh_vals: list[float] = []

    if explicit_water_indices is not None:
        wi = np.asarray(explicit_water_indices, dtype=np.int32).reshape(-1, 3)
        water_triples: list[tuple[int, int, int]] = [
            (int(row[0]), int(row[1]), int(row[2])) for row in wi
        ]
        water_edge_map: dict[tuple[int, int], tuple[int, int, int]] = {}
        for o_idx, h1_idx, h2_idx in water_triples:
            for a, b in ((o_idx, h1_idx), (o_idx, h2_idx), (h1_idx, h2_idx)):
                edge = (min(a, b), max(a, b))
                water_edge_map[edge] = (o_idx, h1_idx, h2_idx)

        solute_pairs: list[list[int]] = []
        solute_lengths: list[float] = []
        for p1, p2, d_a in raw_constraints:
            edge = (min(p1, p2), max(p1, p2))
            if edge in water_edge_map:
                o_idx, h1_idx, h2_idx = water_edge_map[edge]
                if p1 == o_idx or p2 == o_idx:
                    r_oh_vals.append(d_a)
                else:
                    r_hh_vals.append(d_a)
            else:
                solute_pairs.append([p1, p2])
                solute_lengths.append(d_a)

        r_oh = float(np.mean(r_oh_vals)) if r_oh_vals else DEFAULT_TIP3P_R_OH
        r_hh = float(np.mean(r_hh_vals)) if r_hh_vals else DEFAULT_TIP3P_R_HH
        c_pairs = (
            np.asarray(solute_pairs, dtype=np.int32)
            if solute_pairs
            else np.zeros((0, 2), dtype=np.int32)
        )
        c_lens = (
            np.asarray(solute_lengths, dtype=np.float32)
            if solute_lengths
            else np.zeros((0,), dtype=np.float32)
        )
        return wi, r_oh, r_hh, c_pairs, c_lens

    # Automatic detection of 3-site water triangles from the <Constraints> graph
    adj: dict[int, dict[int, float]] = {}
    for p1, p2, d_a in raw_constraints:
        if p1 == p2:
            continue
        adj.setdefault(p1, {})[p2] = d_a
        adj.setdefault(p2, {})[p1] = d_a

    visited_water_atoms: set[int] = set()
    water_Constraint_edges: set[tuple[int, int]] = set()
    detected_waters: list[list[int]] = []

    for u in sorted(adj.keys()):
        if u in visited_water_atoms:
            continue
        nbrs_u = sorted(adj[u].keys())
        # A rigid 3-site water molecule forms an isolated triangle (degree 2 for all 3 vertices)
        if len(nbrs_u) != 2:
            continue
        v, w = nbrs_u[0], nbrs_u[1]
        if v in visited_water_atoms or w in visited_water_atoms:
            continue
        if len(adj.get(v, {})) != 2 or len(adj.get(w, {})) != 2:
            continue
        if w not in adj[v]:
            continue

        # Candidate triangle (u, v, w)
        tri = [u, v, w]
        tri_masses = [float(masses_np[idx]) if idx < len(masses_np) else 1.0 for idx in tri]
        heavy = [idx for idx, m in zip(tri, tri_masses) if m > 4.0]
        light = [idx for idx, m in zip(tri, tri_masses) if m <= 4.0]

        if len(heavy) == 1 and len(light) == 2:
            o_idx = heavy[0]
            h1_idx, h2_idx = sorted(light)
        else:
            # Fallback by geometry: the vertex shared by the two shorter (equal) O-H edges is O
            d_uv = adj[u][v]
            d_uw = adj[u][w]
            d_vw = adj[v][w]
            diffs = [
                (abs(d_uv - d_uw), u, v, w),
                (abs(d_uv - d_vw), v, u, w),
                (abs(d_uw - d_vw), w, u, v),
            ]
            diffs.sort(key=lambda item: item[0])
            _, o_idx, h1_idx, h2_idx = diffs[0]
            if h1_idx > h2_idx:
                h1_idx, h2_idx = h2_idx, h1_idx

        d_oh1 = adj[o_idx][h1_idx]
        d_oh2 = adj[o_idx][h2_idx]
        d_hh = adj[h1_idx][h2_idx]

        detected_waters.append([o_idx, h1_idx, h2_idx])
        r_oh_vals.extend([d_oh1, d_oh2])
        r_hh_vals.append(d_hh)
        visited_water_atoms.update(tri)
        for a, b in ((o_idx, h1_idx), (o_idx, h2_idx), (h1_idx, h2_idx)):
            water_Constraint_edges.add((min(a, b), max(a, b)))

    solute_pairs = []
    solute_lengths = []
    for p1, p2, d_a in raw_constraints:
        edge = (min(p1, p2), max(p1, p2))
        if edge in water_Constraint_edges:
            continue
        solute_pairs.append([p1, p2])
        solute_lengths.append(d_a)

    wi_arr = (
        np.asarray(detected_waters, dtype=np.int32)
        if detected_waters
        else np.zeros((0, 3), dtype=np.int32)
    )
    r_oh = float(np.mean(r_oh_vals)) if r_oh_vals else DEFAULT_TIP3P_R_OH
    r_hh = float(np.mean(r_hh_vals)) if r_hh_vals else DEFAULT_TIP3P_R_HH
    c_pairs = (
        np.asarray(solute_pairs, dtype=np.int32)
        if solute_pairs
        else np.zeros((0, 2), dtype=np.int32)
    )
    c_lens = (
        np.asarray(solute_lengths, dtype=np.float32)
        if solute_lengths
        else np.zeros((0,), dtype=np.float32)
    )
    return wi_arr, r_oh, r_hh, c_pairs, c_lens


def _compute_molecule_ids(
    n_atoms: int,
    bonds: np.ndarray,
    constraint_pairs: np.ndarray,
    water_indices: np.ndarray,
) -> np.ndarray:
    """Compute per-atom connected-component molecule IDs from bonds and constraints."""
    parent = list(range(n_atoms))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        if 0 <= a < n_atoms and 0 <= b < n_atoms:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    for row in bonds:
        _union(int(row[0]), int(row[1]))
    for row in constraint_pairs:
        _union(int(row[0]), int(row[1]))
    for row in water_indices:
        _union(int(row[0]), int(row[1]))
        _union(int(row[0]), int(row[2]))

    root_to_mol: dict[int, int] = {}
    mol_ids = np.zeros(n_atoms, dtype=np.int32)
    for i in range(n_atoms):
        r = _find(i)
        if r not in root_to_mol:
            root_to_mol[r] = len(root_to_mol)
        mol_ids[i] = root_to_mol[r]
    return mol_ids


def parse_openmm_system_xml(
    xml_path_or_str: str | Path | ET.Element,
    positions: ArrayLike | str | Path | None = None,
    *,
    water_indices: ArrayLike | None = None,
) -> OpenMMSystemData:
    """Parse an OpenMM ``<System>`` XML into unpadded Prolix AKMA :class:`OpenMMSystemData`.

    Args:
        xml_path_or_str: Path to an OpenMM ``<System>`` XML file, raw XML string,
            or parsed ``xml.etree.ElementTree.Element``.
        positions: Optional ``(N, 3)`` array in Angstroms, or path to a PDB file.
            Defaults to zeros ``(N, 3)`` when ``None``.
        water_indices: Optional ``(N_waters, 3)`` ``[O, H1, H2]`` array. When ``None``,
            3-site rigid water triangles are automatically separated from ``<Constraints>``.

    Returns:
        :class:`OpenMMSystemData` with all parameters converted to Prolix AKMA units.
    """
    # Support swapped (pdb_path, system_xml_path) call order transparently
    if (
        isinstance(xml_path_or_str, (str, Path))
        and str(xml_path_or_str).lower().endswith(".pdb")
        and isinstance(positions, (str, Path))
    ):
        xml_path_or_str, positions = positions, xml_path_or_str

    root = _load_xml_root(xml_path_or_str)

    # 1. <Particles>
    particles_elem = root.find("Particles")
    if particles_elem is None:
        for elem in root.iter("Particles"):
            if elem is root or elem in list(root):
                particles_elem = elem
                break
    masses_list: list[float] = []
    if particles_elem is not None:
        for p_elem in particles_elem.findall("Particle"):
            masses_list.append(float(p_elem.attrib.get("mass", 0.0)))
    n_atoms = len(masses_list)
    if n_atoms == 0:
        raise ValueError("OpenMM <System> XML contains 0 <Particle> entries.")
    masses_np = np.asarray(masses_list, dtype=np.float32)

    # Resolve positions
    if positions is None:
        pos_arr = jnp.zeros((n_atoms, 3), dtype=jnp.float32)
    elif isinstance(positions, (str, Path)):
        pos_arr = _parse_pdb_positions(positions)
    else:
        pos_arr = jnp.asarray(positions, dtype=jnp.float32)
    if pos_arr.shape != (n_atoms, 3):
        raise ValueError(
            f"positions shape {pos_arr.shape} does not match particle count ({n_atoms}, 3)"
        )

    # 2. <PeriodicBoxVectors> (nm -> Angstrom, reduced lower-triangular)
    box_elem = root.find("PeriodicBoxVectors")
    if box_elem is not None:
        a_el = box_elem.find("A")
        b_el = box_elem.find("B")
        c_el = box_elem.find("C")
        if a_el is not None and b_el is not None and c_el is not None:
            raw_box = jnp.array(
                [
                    [
                        float(a_el.attrib.get("x", 0.0)),
                        float(a_el.attrib.get("y", 0.0)),
                        float(a_el.attrib.get("z", 0.0)),
                    ],
                    [
                        float(b_el.attrib.get("x", 0.0)),
                        float(b_el.attrib.get("y", 0.0)),
                        float(b_el.attrib.get("z", 0.0)),
                    ],
                    [
                        float(c_el.attrib.get("x", 0.0)),
                        float(c_el.attrib.get("y", 0.0)),
                        float(c_el.attrib.get("z", 0.0)),
                    ],
                ],
                dtype=jnp.float32,
            ) * NM_TO_ANGSTROM
            box_reduced = reduce_triclinic_box_vectors(raw_box)
            diag_vals = jnp.diag(box_reduced)
            has_pbc = bool(jnp.all(diag_vals > 0.0))
            off_diag_norm = float(
                jnp.sum(jnp.abs(box_reduced - jnp.diag(diag_vals)))
            )
            is_triclinic = bool(has_pbc and off_diag_norm > 1e-5)
            if has_pbc and not is_triclinic:
                box_mat = jnp.diag(diag_vals)
                box_size: jnp.ndarray | None = diag_vals
            elif has_pbc:
                box_mat = box_reduced
                box_size = box_reduced
            else:
                box_mat = jnp.zeros((3, 3), dtype=jnp.float32)
                box_size = None
        else:
            box_mat = jnp.zeros((3, 3), dtype=jnp.float32)
            box_size = None
            has_pbc = False
            is_triclinic = False
    else:
        box_mat = jnp.zeros((3, 3), dtype=jnp.float32)
        box_size = None
        has_pbc = False
        is_triclinic = False

    # 3. <Constraints> (nm -> Angstrom, separate 3-site water triangles vs solute constraints)
    constraints_elem = root.find("Constraints")
    raw_constraints: list[tuple[int, int, float]] = []
    if constraints_elem is not None:
        for c_elem in constraints_elem.findall("Constraint"):
            p1 = int(c_elem.attrib.get("p1", c_elem.attrib.get("particle1", 0)))
            p2 = int(c_elem.attrib.get("p2", c_elem.attrib.get("particle2", 0)))
            d_nm = float(c_elem.attrib.get("d", c_elem.attrib.get("distance", 0.0)))
            raw_constraints.append((p1, p2, d_nm * NM_TO_ANGSTROM))

    wi_np, water_r_OH, water_r_HH, c_pairs_np, c_lens_np = (
        _separate_water_and_solute_constraints(
            raw_constraints, masses_np, explicit_water_indices=water_indices
        )
    )

    # 4. <HarmonicBondForce> (r0 = d_nm * 10, k_prolix = k_OpenMM / 418.4)
    bond_indices_list: list[list[int]] = []
    bond_params_list: list[list[float]] = []
    for f_elem in _find_force_elements(root, "HarmonicBondForce"):
        bonds_container = f_elem.find("Bonds")
        bond_iter = (
            bonds_container.findall("Bond")
            if bonds_container is not None
            else f_elem.findall("Bond")
        )
        for b_elem in bond_iter:
            p1 = int(b_elem.attrib.get("p1", b_elem.attrib.get("particle1", 0)))
            p2 = int(b_elem.attrib.get("p2", b_elem.attrib.get("particle2", 0)))
            d_nm = float(b_elem.attrib.get("d", b_elem.attrib.get("length", 0.0)))
            k_omm = float(b_elem.attrib.get("k", 0.0))
            bond_indices_list.append([p1, p2])
            bond_params_list.append([d_nm * NM_TO_ANGSTROM, k_omm * KJ_NM2_TO_KCAL_A2])

    bonds_np = (
        np.asarray(bond_indices_list, dtype=np.int32)
        if bond_indices_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    bond_params_np = (
        np.asarray(bond_params_list, dtype=np.float32)
        if bond_params_list
        else np.zeros((0, 2), dtype=np.float32)
    )

    # 5. <HarmonicAngleForce> (theta0 in rad, k_prolix = k_OpenMM / 4.184)
    angle_indices_list: list[list[int]] = []
    angle_params_list: list[list[float]] = []
    for f_elem in _find_force_elements(root, "HarmonicAngleForce"):
        angles_container = f_elem.find("Angles")
        angle_iter = (
            angles_container.findall("Angle")
            if angles_container is not None
            else f_elem.findall("Angle")
        )
        for a_elem in angle_iter:
            p1 = int(a_elem.attrib.get("p1", a_elem.attrib.get("particle1", 0)))
            p2 = int(a_elem.attrib.get("p2", a_elem.attrib.get("particle2", 0)))
            p3 = int(a_elem.attrib.get("p3", a_elem.attrib.get("particle3", 0)))
            theta0 = float(
                a_elem.attrib.get(
                    "a", a_elem.attrib.get("angle", a_elem.attrib.get("theta0", 0.0))
                )
            )
            k_omm = float(a_elem.attrib.get("k", 0.0))
            angle_indices_list.append([p1, p2, p3])
            angle_params_list.append([theta0, k_omm * KJ_TO_KCAL])

    angles_np = (
        np.asarray(angle_indices_list, dtype=np.int32)
        if angle_indices_list
        else np.zeros((0, 3), dtype=np.int32)
    )
    angle_params_np = (
        np.asarray(angle_params_list, dtype=np.float32)
        if angle_params_list
        else np.zeros((0, 2), dtype=np.float32)
    )

    # 6. <PeriodicTorsionForce> (periodicity, phase, k_prolix = k_OpenMM / 4.184)
    dih_indices_list: list[list[int]] = []
    dih_params_list: list[list[float]] = []
    for f_elem in _find_force_elements(root, "PeriodicTorsionForce"):
        torsions_container = f_elem.find("Torsions")
        torsion_iter = (
            torsions_container.findall("Torsion")
            if torsions_container is not None
            else f_elem.findall("Torsion")
        )
        for t_elem in torsion_iter:
            p1 = int(t_elem.attrib.get("p1", t_elem.attrib.get("particle1", 0)))
            p2 = int(t_elem.attrib.get("p2", t_elem.attrib.get("particle2", 0)))
            p3 = int(t_elem.attrib.get("p3", t_elem.attrib.get("particle3", 0)))
            p4 = int(t_elem.attrib.get("p4", t_elem.attrib.get("particle4", 0)))
            periodicity = float(t_elem.attrib.get("periodicity", 1.0))
            phase = float(t_elem.attrib.get("phase", 0.0))
            k_omm = float(t_elem.attrib.get("k", 0.0))
            dih_indices_list.append([p1, p2, p3, p4])
            dih_params_list.append([periodicity, phase, k_omm * KJ_TO_KCAL])

    dihedrals_np = (
        np.asarray(dih_indices_list, dtype=np.int32)
        if dih_indices_list
        else np.zeros((0, 4), dtype=np.int32)
    )
    dihedral_params_np = (
        np.asarray(dih_params_list, dtype=np.float32)
        if dih_params_list
        else np.zeros((0, 3), dtype=np.float32)
    )

    # 7. <CMAPTorsionForce> (maps in kJ/mol -> / 4.184 kcal/mol, 8-atom torsion indices)
    cmap_maps_list: list[np.ndarray] = []
    cmap_torsions_list: list[list[int]] = []
    cmap_map_idx_list: list[int] = []
    for f_elem in _find_force_elements(root, "CMAPTorsionForce"):
        map_offset = len(cmap_maps_list)
        maps_container = f_elem.find("Maps")
        map_iter = (
            maps_container.findall("Map")
            if maps_container is not None
            else f_elem.findall("Map")
        )
        for m_elem in map_iter:
            text_parts: list[str] = []
            if m_elem.text and m_elem.text.strip():
                text_parts.extend(m_elem.text.split())
            if "energy" in m_elem.attrib:
                text_parts.extend(m_elem.attrib["energy"].split())
            for child in m_elem:
                if child.text and child.text.strip():
                    text_parts.extend(child.text.split())
            vals = np.asarray([float(x) * KJ_TO_KCAL for x in text_parts], dtype=np.float32)
            grid_size = int(m_elem.attrib.get("size", int( round(np.sqrt(vals.size)) )))
            if vals.size != grid_size * grid_size:
                raise ValueError(
                    f"CMAP <Map size='{grid_size}'> expected {grid_size * grid_size} "
                    f"values, got {vals.size}"
                )
            cmap_maps_list.append(vals.reshape(grid_size, grid_size))

        torsions_container = f_elem.find("Torsions")
        torsion_iter = (
            torsions_container.findall("Torsion")
            if torsions_container is not None
            else f_elem.findall("Torsion")
        )
        for t_elem in torsion_iter:
            m_idx = int(t_elem.attrib.get("map", 0)) + map_offset
            if "a1" in t_elem.attrib:
                a1 = int(t_elem.attrib["a1"])
                a2 = int(t_elem.attrib["a2"])
                a3 = int(t_elem.attrib["a3"])
                a4 = int(t_elem.attrib["a4"])
                b1 = int(t_elem.attrib.get("b1", a2))
                b2 = int(t_elem.attrib.get("b2", a3))
                b3 = int(t_elem.attrib.get("b3", a4))
                b4 = int(t_elem.attrib["b4"])
                cmap_torsions_list.append([a1, a2, a3, a4, b1, b2, b3, b4])
            else:
                p1 = int(t_elem.attrib["p1"])
                p2 = int(t_elem.attrib["p2"])
                p3 = int(t_elem.attrib["p3"])
                p4 = int(t_elem.attrib["p4"])
                if "p8" in t_elem.attrib:
                    p5 = int(t_elem.attrib["p5"])
                    p6 = int(t_elem.attrib["p6"])
                    p7 = int(t_elem.attrib["p7"])
                    p8 = int(t_elem.attrib["p8"])
                    cmap_torsions_list.append([p1, p2, p3, p4, p5, p6, p7, p8])
                else:
                    p5 = int(t_elem.attrib["p5"])
                    cmap_torsions_list.append([p1, p2, p3, p4, p2, p3, p4, p5])
            cmap_map_idx_list.append(m_idx)

    if cmap_maps_list:
        cmap_grids_jnp = jnp.asarray(np.stack(cmap_maps_list, axis=0), dtype=jnp.float32)
        raw_coeffs = precompute_cmap_coefficients(cmap_grids_jnp).astype(jnp.float32)
        n_maps = int(raw_coeffs.shape[0])
        g_size = int(raw_coeffs.shape[1])
        mb = _next_bucket(max(n_maps, 1), CMAP_MAP_BUCKETS, use_size_buckets=True)
        pad_m = mb - n_maps
        if pad_m > 0:
            cmap_coeffs_jnp = jnp.pad(raw_coeffs, ((0, pad_m), (0, 0), (0, 0), (0, 0)))
        else:
            cmap_coeffs_jnp = raw_coeffs
    else:
        g_size = 24
        cmap_grids_jnp = jnp.zeros((0, g_size, g_size), dtype=jnp.float32)
        cmap_coeffs_jnp = jnp.zeros(
            (CMAP_MAP_BUCKETS[0], g_size, g_size, 16), dtype=jnp.float32
        )

    cmap_torsions_np = (
        np.asarray(cmap_torsions_list, dtype=np.int32)
        if cmap_torsions_list
        else np.zeros((0, 8), dtype=np.int32)
    )
    cmap_indices_np = (
        np.asarray(cmap_map_idx_list, dtype=np.int32)
        if cmap_map_idx_list
        else np.zeros((0,), dtype=np.int32)
    )

    # 8. <NonbondedForce> (q, sigma_A = sigma_nm * 10, eps_kcal = eps_kJ / 4.184, unified exceptions)
    charges_np = np.zeros(n_atoms, dtype=np.float32)
    sigmas_np = np.ones(n_atoms, dtype=np.float32)
    epsilons_np = np.zeros(n_atoms, dtype=np.float32)

    excl_pairs_list: list[list[int]] = []
    excl_vdw_list: list[float] = []
    excl_elec_list: list[float] = []

    exc14_pairs_list: list[list[int]] = []
    exc14_qprod_list: list[float] = []
    exc14_sigma_list: list[float] = []
    exc14_eps_list: list[float] = []

    nonbonded_cutoff = 9.0
    lj_switch_width = 0.0
    use_dispersion_correction = True
    electrostatic_method = "pme" if has_pbc else "coulomb"
    reaction_field_dielectric = 78.3
    pme_alpha = 0.34 if has_pbc else 0.0

    for f_elem in _find_force_elements(root, "NonbondedForce"):
        if "cutoff" in f_elem.attrib:
            nonbonded_cutoff = float(f_elem.attrib["cutoff"]) * NM_TO_ANGSTROM
        use_sw = f_elem.attrib.get("useSwitchingFunction", "0") in ("1", "true", "True")
        if use_sw and "switchingDistance" in f_elem.attrib:
            sw_dist_a = float(f_elem.attrib["switchingDistance"]) * NM_TO_ANGSTROM
            lj_switch_width = max(nonbonded_cutoff - sw_dist_a, 0.0)
        if "useDispersionCorrection" in f_elem.attrib:
            use_dispersion_correction = f_elem.attrib["useDispersionCorrection"] in (
                "1",
                "true",
                "True",
            )
        if "rfDielectric" in f_elem.attrib:
            reaction_field_dielectric = float(f_elem.attrib["rfDielectric"])

        method_raw = f_elem.attrib.get(
            "method", f_elem.attrib.get("nonbondedMethod", "4" if has_pbc else "0")
        )
        if str(method_raw) in ("0", "NoCutoff"):
            electrostatic_method = "coulomb"
            pme_alpha = 0.0
        elif str(method_raw) in ("1", "CutoffNonPeriodic"):
            electrostatic_method = "cutoff"
            pme_alpha = 0.0
        elif str(method_raw) in ("2", "CutoffPeriodic"):
            electrostatic_method = "reaction_field"
            pme_alpha = 0.0
        else:
            electrostatic_method = "pme"
            alpha_nm_inv = float(f_elem.attrib.get("alpha", 0.0))
            ewald_tol = float(f_elem.attrib.get("ewaldErrorTolerance", 5e-4))
            if alpha_nm_inv > 0.0:
                pme_alpha = alpha_nm_inv / NM_TO_ANGSTROM
            else:
                pme_alpha = solve_ewald_alpha(nonbonded_cutoff, ewald_tol)

        nb_particles_container = f_elem.find("Particles")
        nb_particle_iter = (
            nb_particles_container.findall("Particle")
            if nb_particles_container is not None
            else f_elem.findall("Particle")
        )
        for idx, p_elem in enumerate(nb_particle_iter):
            if idx >= n_atoms:
                break
            q = float(p_elem.attrib.get("q", p_elem.attrib.get("charge", 0.0)))
            sig_nm = float(p_elem.attrib.get("sig", p_elem.attrib.get("sigma", 0.1)))
            eps_kj = float(p_elem.attrib.get("eps", p_elem.attrib.get("epsilon", 0.0)))
            charges_np[idx] = q
            sigmas_np[idx] = sig_nm * NM_TO_ANGSTROM
            epsilons_np[idx] = eps_kj * KJ_TO_KCAL

        nb_exc_container = f_elem.find("Exceptions")
        nb_exc_iter = (
            nb_exc_container.findall("Exception")
            if nb_exc_container is not None
            else f_elem.findall("Exception")
        )
        for e_elem in nb_exc_iter:
            p1 = int(e_elem.attrib.get("p1", e_elem.attrib.get("particle1", 0)))
            p2 = int(e_elem.attrib.get("p2", e_elem.attrib.get("particle2", 0)))
            q_prod = float(
                e_elem.attrib.get("q", e_elem.attrib.get("chargeProd", 0.0))
            )
            sig_nm = float(e_elem.attrib.get("sig", e_elem.attrib.get("sigma", 0.1)))
            eps_kj = float(e_elem.attrib.get("eps", e_elem.attrib.get("epsilon", 0.0)))

            sig_a = sig_nm * NM_TO_ANGSTROM
            eps_kcal = eps_kj * KJ_TO_KCAL

            # Both zero exceptions (1-2/1-3) and nonzero exceptions (1-4) are excluded
            # from bulk nonbonded kernels with 0.0 scales so 1-4 pairs are never
            # double-counted in bulk LJ/Coulomb or PME exclusion corrections.
            excl_pairs_list.append([p1, p2])
            excl_vdw_list.append(0.0)
            excl_elec_list.append(0.0)

            if abs(q_prod) > 0.0 or abs(eps_kcal) > 0.0:
                exc14_pairs_list.append([p1, p2])
                exc14_qprod_list.append(q_prod)
                exc14_sigma_list.append(sig_a)
                exc14_eps_list.append(eps_kcal)

    excl_indices_np = (
        np.asarray(excl_pairs_list, dtype=np.int32)
        if excl_pairs_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    excl_scales_vdw_np = (
        np.asarray(excl_vdw_list, dtype=np.float32)
        if excl_vdw_list
        else np.zeros((0,), dtype=np.float32)
    )
    excl_scales_elec_np = (
        np.asarray(excl_elec_list, dtype=np.float32)
        if excl_elec_list
        else np.zeros((0,), dtype=np.float32)
    )

    exc14_pairs_np = (
        np.asarray(exc14_pairs_list, dtype=np.int32)
        if exc14_pairs_list
        else np.zeros((0, 2), dtype=np.int32)
    )
    exc14_qprod_np = (
        np.asarray(exc14_qprod_list, dtype=np.float32)
        if exc14_qprod_list
        else np.zeros((0,), dtype=np.float32)
    )
    exc14_sigma_np = (
        np.asarray(exc14_sigma_list, dtype=np.float32)
        if exc14_sigma_list
        else np.zeros((0,), dtype=np.float32)
    )
    exc14_eps_np = (
        np.asarray(exc14_eps_list, dtype=np.float32)
        if exc14_eps_list
        else np.zeros((0,), dtype=np.float32)
    )

    # 9. <GBSAOBCForce> (radii_A = r_nm * 10, scaled_radii = radii_A * scale)
    radii_np = np.ones(n_atoms, dtype=np.float32)
    obc_scale_np = np.ones(n_atoms, dtype=np.float32)
    scaled_radii_np = np.ones(n_atoms, dtype=np.float32)
    has_implicit_solvent = False
    solute_dielectric = 1.0
    solvent_dielectric = 78.5
    surface_area_energy = 2.25936 * KJ_NM2_TO_KCAL_A2

    for f_elem in _find_force_elements(root, "GBSAOBCForce"):
        has_implicit_solvent = True
        solute_dielectric = float(f_elem.attrib.get("soluteDielectric", 1.0))
        solvent_dielectric = float(f_elem.attrib.get("solventDielectric", 78.5))
        sa_kj_nm2 = float(f_elem.attrib.get("surfaceAreaEnergy", 2.25936))
        surface_area_energy = sa_kj_nm2 * KJ_NM2_TO_KCAL_A2

        gb_particles_container = f_elem.find("Particles")
        gb_particle_iter = (
            gb_particles_container.findall("Particle")
            if gb_particles_container is not None
            else f_elem.findall("Particle")
        )
        for idx, p_elem in enumerate(gb_particle_iter):
            if idx >= n_atoms:
                break
            r_nm = float(
                p_elem.attrib.get(
                    "radius", p_elem.attrib.get("r", p_elem.attrib.get("radii", 0.15))
                )
            )
            scale = float(
                p_elem.attrib.get("scale", p_elem.attrib.get("scalingFactor", 0.8))
            )
            r_a = r_nm * NM_TO_ANGSTROM
            radii_np[idx] = r_a
            obc_scale_np[idx] = scale
            scaled_radii_np[idx] = r_a * scale
            if charges_np[idx] == 0.0 and ("q" in p_elem.attrib or "charge" in p_elem.attrib):
                charges_np[idx] = float(
                    p_elem.attrib.get("q", p_elem.attrib.get("charge", 0.0))
                )

    # 10. <MonteCarloBarostat> (defaultPressure, defaultTemperature, frequency)
    barostat_spec: BarostatSpec | None = None
    barostat_elems = _find_force_elements(root, "MonteCarloBarostat")
    if barostat_elems:
        b_elem = barostat_elems[0]
        p_bar = float(
            b_elem.attrib.get("defaultPressure", b_elem.attrib.get("pressure", 1.0))
        )
        t_k = float(
            b_elem.attrib.get(
                "defaultTemperature", b_elem.attrib.get("temperature", 300.0)
            )
        )
        freq = int(b_elem.attrib.get("frequency", 25))
        barostat_spec = BarostatSpec(
            default_pressure_bar=p_bar,
            default_pressure_atm=p_bar * BAR_TO_ATM,
            default_temperature_k=t_k,
            frequency=freq,
        )

    # 11. Optional <CustomNonbondedForce> interaction groups for cosolvents
    cosolvent_group_id_np = np.zeros(n_atoms, dtype=np.int32)
    seen_group_sets: dict[frozenset[int], int] = {}
    next_group_idx = 1
    for f_elem in _find_force_elements(root, "CustomNonbondedForce"):
        ig_container = f_elem.find("InteractionGroups")
        ig_iter = (
            ig_container.findall("InteractionGroup")
            if ig_container is not None
            else f_elem.findall("InteractionGroup")
        )
        for ig_elem in ig_iter:
            group_particles: list[int] = []
            for set_tag in ("Set1", "Set2"):
                s_elem = ig_elem.find(set_tag)
                if s_elem is not None:
                    for p_elem in s_elem.findall("Particle"):
                        p_idx = int(p_elem.attrib.get("index", p_elem.attrib.get("p", -1)))
                        if 0 <= p_idx < n_atoms:
                            group_particles.append(p_idx)
            if not group_particles:
                continue
            key_set = frozenset(group_particles)
            if key_set not in seen_group_sets:
                seen_group_sets[key_set] = next_group_idx
                next_group_idx += 1
            g_id = seen_group_sets[key_set]
            for p_idx in key_set:
                if cosolvent_group_id_np[p_idx] == 0:
                    cosolvent_group_id_np[p_idx] = g_id

    molecule_id_np = _compute_molecule_ids(n_atoms, bonds_np, c_pairs_np, wi_np)

    return OpenMMSystemData(
        n_atoms=n_atoms,
        positions=pos_arr,
        masses=jnp.asarray(masses_np, dtype=jnp.float32),
        charges=jnp.asarray(charges_np, dtype=jnp.float32),
        sigmas=jnp.asarray(sigmas_np, dtype=jnp.float32),
        epsilons=jnp.asarray(epsilons_np, dtype=jnp.float32),
        radii=jnp.asarray(radii_np, dtype=jnp.float32),
        obc_scale=jnp.asarray(obc_scale_np, dtype=jnp.float32),
        scaled_radii=jnp.asarray(scaled_radii_np, dtype=jnp.float32),
        box=box_mat,
        box_size=box_size,
        has_pbc=has_pbc,
        is_triclinic=is_triclinic,
        has_implicit_solvent=has_implicit_solvent,
        bonds=jnp.asarray(bonds_np, dtype=jnp.int32),
        bond_params=jnp.asarray(bond_params_np, dtype=jnp.float32),
        angles=jnp.asarray(angles_np, dtype=jnp.int32),
        angle_params=jnp.asarray(angle_params_np, dtype=jnp.float32),
        dihedrals=jnp.asarray(dihedrals_np, dtype=jnp.int32),
        dihedral_params=jnp.asarray(dihedral_params_np, dtype=jnp.float32),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float32),
        urey_bradley_bonds=jnp.zeros((0, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((0, 2), dtype=jnp.float32),
        cmap_torsions=jnp.asarray(cmap_torsions_np, dtype=jnp.int32),
        cmap_indices=jnp.asarray(cmap_indices_np, dtype=jnp.int32),
        cmap_energy_grids=cmap_grids_jnp,
        cmap_coeffs=cmap_coeffs_jnp,
        water_indices=jnp.asarray(wi_np, dtype=jnp.int32),
        water_r_OH=water_r_OH,
        water_r_HH=water_r_HH,
        constraint_pairs=jnp.asarray(c_pairs_np, dtype=jnp.int32),
        constraint_lengths=jnp.asarray(c_lens_np, dtype=jnp.float32),
        excl_indices=jnp.asarray(excl_indices_np, dtype=jnp.int32),
        excl_scales_vdw=jnp.asarray(excl_scales_vdw_np, dtype=jnp.float32),
        excl_scales_elec=jnp.asarray(excl_scales_elec_np, dtype=jnp.float32),
        exception_pairs=jnp.asarray(exc14_pairs_np, dtype=jnp.int32),
        exception_chargeprods=jnp.asarray(exc14_qprod_np, dtype=jnp.float32),
        exception_sigmas=jnp.asarray(exc14_sigma_np, dtype=jnp.float32),
        exception_epsilons=jnp.asarray(exc14_eps_np, dtype=jnp.float32),
        pme_alpha=pme_alpha,
        nonbonded_cutoff=nonbonded_cutoff,
        lj_switch_width=lj_switch_width,
        use_dispersion_correction=use_dispersion_correction,
        electrostatic_method=electrostatic_method,
        reaction_field_dielectric=reaction_field_dielectric,
        solute_dielectric=solute_dielectric,
        solvent_dielectric=solvent_dielectric,
        surface_area_energy=surface_area_energy,
        barostat=barostat_spec,
        molecule_id=jnp.asarray(molecule_id_np, dtype=jnp.int32),
        cosolvent_group_id=jnp.asarray(cosolvent_group_id_np, dtype=jnp.int32),
    )


def bundle_from_openmm_system_xml(
    xml_path_or_str: str | Path | ET.Element,
    positions: ArrayLike | str | Path | None = None,
    *,
    water_indices: ArrayLike | None = None,
    boundary_condition: Literal["free", "periodic"] | None = None,
    use_size_buckets: bool = True,
    _preparsed_data: OpenMMSystemData | None = None,
) -> MolecularBundle:
    """Parse an OpenMM ``<System>`` XML into a bucket-padded :class:`MolecularBundle`.

    Args:
        xml_path_or_str: Path to an OpenMM ``<System>`` XML file, raw XML string,
            or ``xml.etree.ElementTree.Element``.
        positions: Optional ``(N, 3)`` coordinates in Angstroms (or path to PDB file).
        water_indices: Optional ``(N_waters, 3)`` ``[O, H1, H2]`` water atom indices.
            When ``None``, 3-site water triangles in ``<Constraints>`` are detected
            automatically and separated from solute bond constraints.
        boundary_condition: Optional override ``"periodic"`` or ``"free"``.
            Defaults to ``"periodic"`` when ``<PeriodicBoxVectors>`` is present and
            non-zero, else ``"free"``.
        use_size_buckets: Whether to pad arrays to the standard bucket ladders.

    Returns:
        :class:`MolecularBundle` ready for energy, force, and ensemble evaluation.
    """
    data = (
        _preparsed_data
        if _preparsed_data is not None
        else parse_openmm_system_xml(
            xml_path_or_str, positions=positions, water_indices=water_indices
        )
    )

    n = data.n_atoms

    def _nb(count: int, ladder: tuple[int, ...]) -> int:
        return _next_bucket(count, ladder, use_size_buckets=use_size_buckets)

    a = _nb(n, ATOM_BUCKETS)

    nb = int(data.bonds.shape[0])
    bb = _nb(max(nb, 1), BOND_BUCKETS)

    na = int(data.angles.shape[0])
    ab = _nb(max(na, 1), ANGLE_BUCKETS)

    nd = int(data.dihedrals.shape[0])
    db = _nb(max(nd, 1), DIHEDRAL_BUCKETS)

    nw = int(data.water_indices.shape[0])
    wb = _nb(max(nw, 1), WATER_BUCKETS)

    ne = int(data.excl_indices.shape[0])
    eb = _nb(max(ne, 1), EXCL_BUCKETS)

    nx = int(data.exception_pairs.shape[0])
    xb = _nb(max(nx, 1), EXCEPTION_BUCKETS)

    nc = int(data.cmap_torsions.shape[0])
    cb = _nb(max(nc, 1), CMAP_BUCKETS)

    n_maps = int(data.cmap_energy_grids.shape[0])
    cmap_map_bucket_idx = _bucket_idx(max(n_maps, 1), CMAP_MAP_BUCKETS)

    n_constr = int(data.constraint_pairs.shape[0])
    constraint_bucket_idx = _bucket_idx(n_constr, CONSTRAINT_BUCKETS)
    cb_constr = (
        CONSTRAINT_BUCKETS[constraint_bucket_idx] if use_size_buckets else n_constr
    )

    # Per-atom-row dense exclusions (a, 32) for NL and Flash kernels
    if ne > 0:
        excl_dense_indices, excl_dense_scales_vdw, excl_dense_scales_elec = (
            pair_list_to_dense_padded(
                np.asarray(data.excl_indices),
                np.asarray(data.excl_scales_vdw),
                np.asarray(data.excl_scales_elec),
                jnp.ones(ne, dtype=bool),
                a,
                max_exclusions=_EXCL_DENSE_MAX,
            )
        )
    else:
        excl_dense_indices = jnp.full((a, _EXCL_DENSE_MAX), -1, dtype=jnp.int32)
        excl_dense_scales_vdw = jnp.ones((a, _EXCL_DENSE_MAX), dtype=jnp.float32)
        excl_dense_scales_elec = jnp.ones((a, _EXCL_DENSE_MAX), dtype=jnp.float32)

    # Expand per-torsion CMAP energy grids to (cb, G, G)
    g_size = int(data.cmap_coeffs.shape[1]) if data.cmap_coeffs.ndim == 4 else 24
    if nc > 0 and n_maps > 0:
        safe_m = jnp.clip(data.cmap_indices, 0, max(n_maps - 1, 0))
        per_torsion_grids = data.cmap_energy_grids[safe_m]
        cmap_energy_grids_padded = _pad_3d(
            per_torsion_grids, cb, g_size, g_size, dtype=jnp.float32
        )
    else:
        cmap_energy_grids_padded = jnp.zeros(
            (cb, g_size, g_size), dtype=jnp.float32
        )

    bc: Literal["free", "periodic"] = (
        boundary_condition
        if boundary_condition is not None
        else ("periodic" if data.has_pbc else "free")
    )

    spec = MolecularShapeSpec(
        atom_bucket_idx=_bucket_idx(n, ATOM_BUCKETS),
        bond_bucket_idx=_bucket_idx(max(nb, 1), BOND_BUCKETS),
        angle_bucket_idx=_bucket_idx(max(na, 1), ANGLE_BUCKETS),
        dihedral_bucket_idx=_bucket_idx(max(nd, 1), DIHEDRAL_BUCKETS),
        water_bucket_idx=_bucket_idx(max(nw, 1), WATER_BUCKETS),
        excl_bucket_idx=_bucket_idx(max(ne, 1), EXCL_BUCKETS),
        cmap_bucket_idx=_bucket_idx(max(nc, 1), CMAP_BUCKETS),
        exception_bucket_idx=_bucket_idx(max(nx, 1), EXCEPTION_BUCKETS),
        has_pbc=data.has_pbc and bc == "periodic",
        has_implicit_solvent=data.has_implicit_solvent,
        boundary_condition=bc,
        has_real_water=nw > 0,
        use_size_buckets=use_size_buckets,
        exact_n_atoms=0 if use_size_buckets else int(a),
        exact_n_waters=0 if use_size_buckets else int(wb),
        cmap_map_bucket_idx=cmap_map_bucket_idx,
        constraint_bucket_idx=constraint_bucket_idx,
        has_real_constraints=n_constr > 0,
        is_triclinic=data.is_triclinic,
    )

    masses_padded = jnp.pad(
        data.masses, (0, a - n), constant_values=1.0
    ).astype(jnp.float32)
    radii_padded = jnp.pad(
        data.radii, (0, a - n), constant_values=1.0
    ).astype(jnp.float32)
    scaled_radii_padded = jnp.pad(
        data.scaled_radii, (0, a - n), constant_values=1.0
    ).astype(jnp.float32)

    box_diag_for_ghost = jnp.diag(data.box) if data.has_pbc else None

    return MolecularBundle(
        positions=_pad_positions_with_ghost_lattice(
            data.positions, a, box_diag_for_ghost, dtype=jnp.float32
        ),
        masses=masses_padded,
        charges=_pad_1d(data.charges, a, dtype=jnp.float32),
        sigmas=_pad_1d(data.sigmas, a, dtype=jnp.float32),
        epsilons=_pad_1d(data.epsilons, a, dtype=jnp.float32),
        radii=radii_padded,
        scaled_radii=scaled_radii_padded,
        atom_mask=_mask(n, a),
        n_atoms=jnp.array(n, dtype=jnp.int32),
        box=data.box,
        bond_idx=_pad_2d(data.bonds, bb, 2, dtype=jnp.int32),
        bond_params=_pad_2d(data.bond_params, bb, 2, dtype=jnp.float32),
        bond_mask=_mask(nb, bb),
        n_bonds=jnp.array(nb, dtype=jnp.int32),
        angle_idx=_pad_2d(data.angles, ab, 3, dtype=jnp.int32),
        angle_params=_pad_2d(data.angle_params, ab, 2, dtype=jnp.float32),
        angle_mask=_mask(na, ab),
        n_angles=jnp.array(na, dtype=jnp.int32),
        dihedral_idx=_pad_2d(data.dihedrals, db, 4, dtype=jnp.int32),
        dihedral_params=_pad_2d(data.dihedral_params, db, 3, dtype=jnp.float32),
        dihedral_mask=_mask(nd, db),
        n_dihedrals=jnp.array(nd, dtype=jnp.int32),
        improper_idx=_pad_2d(data.impropers, db, 4, dtype=jnp.int32),
        improper_params=_pad_2d(data.improper_params, db, 3, dtype=jnp.float32),
        improper_mask=_mask(0, db),
        improper_is_periodic=jnp.array(False, dtype=bool),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=_pad_2d(data.urey_bradley_bonds, ab, 3, dtype=jnp.int32),
        urey_bradley_params=_pad_2d(data.urey_bradley_params, ab, 2, dtype=jnp.float32),
        urey_bradley_mask=_mask(0, ab),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=_pad_2d(data.cmap_torsions, cb, 8, dtype=jnp.int32),
        cmap_energy_grids=cmap_energy_grids_padded,
        cmap_mask=_mask(nc, cb),
        n_cmap=jnp.array(nc, dtype=jnp.int32),
        water_indices=_pad_2d(data.water_indices, wb, 3, dtype=jnp.int32),
        water_mask=_mask(nw, wb),
        n_waters=jnp.array(nw, dtype=jnp.int32),
        excl_indices=_pad_2d(data.excl_indices, eb, 2, dtype=jnp.int32),
        excl_scales_vdw=_pad_1d(data.excl_scales_vdw, eb, dtype=jnp.float32),
        excl_scales_elec=_pad_1d(data.excl_scales_elec, eb, dtype=jnp.float32),
        excl_mask=_mask(ne, eb),
        n_excl=jnp.array(ne, dtype=jnp.int32),
        exception_pairs=_pad_2d(data.exception_pairs, xb, 2, dtype=jnp.int32),
        exception_sigmas=_pad_1d(data.exception_sigmas, xb, dtype=jnp.float32),
        exception_epsilons=_pad_1d(data.exception_epsilons, xb, dtype=jnp.float32),
        exception_chargeprods=_pad_1d(data.exception_chargeprods, xb, dtype=jnp.float32),
        exception_mask=_mask(nx, xb),
        n_exception_pairs=jnp.array(nx, dtype=jnp.int32),
        pme_alpha=jnp.array(float(data.pme_alpha), dtype=jnp.float32),
        cutoff_distance=jnp.array(float(data.nonbonded_cutoff), dtype=jnp.float32),
        shape_spec=spec,
        excl_dense_indices=excl_dense_indices,
        excl_dense_scales_vdw=excl_dense_scales_vdw,
        excl_dense_scales_elec=excl_dense_scales_elec,
        cmap_indices=_pad_1d(data.cmap_indices, cb, dtype=jnp.int32),
        cmap_coeffs=data.cmap_coeffs,
        constraint_pairs=_pad_2d(data.constraint_pairs, cb_constr, 2, dtype=jnp.int32),
        constraint_lengths=_pad_1d(data.constraint_lengths, cb_constr, dtype=jnp.float32),
        constraint_mask=_mask(n_constr, cb_constr),
        n_constraints=jnp.array(n_constr, dtype=jnp.int32),
        water_r_OH=jnp.array(float(data.water_r_OH), dtype=jnp.float32),
        water_r_HH=jnp.array(float(data.water_r_HH), dtype=jnp.float32),
    )


def physics_system_from_openmm_xml(
    xml_path_or_str: str | Path | ET.Element,
    positions: ArrayLike | str | Path | None = None,
    *,
    water_indices: ArrayLike | None = None,
    pme_grid_points: int = 64,
    use_size_buckets: bool = True,
) -> PhysicsSystem:
    """Build a :class:`PhysicsSystem` with unified 1-4 exceptions from OpenMM ``<System>`` XML."""
    data = parse_openmm_system_xml(
        xml_path_or_str, positions=positions, water_indices=water_indices
    )
    bundle = bundle_from_openmm_system_xml(
        xml_path_or_str,
        positions=positions,
        water_indices=water_indices,
        use_size_buckets=use_size_buckets,
        _preparsed_data=data,
    )
    n = int(bundle.positions.shape[0])

    # Build dense (n, n) exclusion scale matrices for dense O(N^2) kernels
    dense_vdw = jnp.ones((n, n), dtype=jnp.float32)
    dense_elec = jnp.ones((n, n), dtype=jnp.float32)
    pairs = bundle.excl_indices
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    valid = (
        bundle.excl_mask
        & (i_idx >= 0)
        & (j_idx >= 0)
        & (i_idx < n)
        & (j_idx < n)
    )
    i_safe = jnp.clip(i_idx, 0, max(n - 1, 0))
    j_safe = jnp.clip(j_idx, 0, max(n - 1, 0))
    sv = jnp.where(valid, bundle.excl_scales_vdw.astype(jnp.float32), 1.0)
    se = jnp.where(valid, bundle.excl_scales_elec.astype(jnp.float32), 1.0)
    dense_vdw = dense_vdw.at[i_safe, j_safe].set(
        jnp.where(valid, sv, dense_vdw[i_safe, j_safe])
    )
    dense_vdw = dense_vdw.at[j_safe, i_safe].set(
        jnp.where(valid, sv, dense_vdw[j_safe, i_safe])
    )
    dense_elec = dense_elec.at[i_safe, j_safe].set(
        jnp.where(valid, se, dense_elec[i_safe, j_safe])
    )
    dense_elec = dense_elec.at[j_safe, i_safe].set(
        jnp.where(valid, se, dense_elec[j_safe, i_safe])
    )

    excl_pair_indices = jnp.where(bundle.excl_mask[:, None], bundle.excl_indices, n)

    if data.has_pbc:
        if data.is_triclinic:
            box_size_static: np.ndarray | None = np.asarray(data.box, dtype=np.float32)
        else:
            box_size_static = np.asarray(jnp.diag(data.box), dtype=np.float32)
    else:
        box_size_static = None

    is_h = (bundle.masses > 0.5) & (bundle.masses < 3.5) & bundle.atom_mask
    # Use 5-column [a1, a2, a3, a4, b4] view on PhysicsSystem.cmap_torsions for CmapTorsionIndices compatibility
    cmap_t_5 = jnp.concatenate(
        [bundle.cmap_torsion_idx[:, 0:4], bundle.cmap_torsion_idx[:, 7:8]], axis=-1
    )

    return PhysicsSystem(
        positions=bundle.positions,
        charges=bundle.charges,
        sigmas=bundle.sigmas,
        epsilons=bundle.epsilons,
        radii=bundle.radii,
        scaled_radii=bundle.scaled_radii,
        masses=bundle.masses,
        element_ids=jnp.where(is_h, 1, jnp.where(bundle.atom_mask, 6, 0)).astype(jnp.int32),
        atom_mask=bundle.atom_mask,
        is_hydrogen=is_h,
        is_backbone=jnp.zeros(n, dtype=bool),
        is_heavy=bundle.atom_mask & (~is_h),
        protein_atom_mask=bundle.atom_mask & (~jnp.isin(jnp.arange(n, dtype=jnp.int32), data.water_indices.reshape(-1))),
        water_atom_mask=bundle.atom_mask & jnp.isin(jnp.arange(n, dtype=jnp.int32), data.water_indices.reshape(-1)),
        bonds=bundle.bond_idx,
        bond_params=bundle.bond_params,
        bond_mask=bundle.bond_mask,
        angles=bundle.angle_idx,
        angle_params=bundle.angle_params,
        angle_mask=bundle.angle_mask,
        dihedrals=bundle.dihedral_idx,
        dihedral_params=bundle.dihedral_params[..., :3],
        dihedral_mask=bundle.dihedral_mask,
        impropers=bundle.improper_idx,
        improper_params=bundle.improper_params[..., :3],
        improper_mask=bundle.improper_mask,
        urey_bradley_bonds=bundle.urey_bradley_idx,
        urey_bradley_params=bundle.urey_bradley_params,
        urey_bradley_mask=bundle.urey_bradley_mask,
        cmap_torsions=cmap_t_5,
        cmap_indices=bundle.cmap_indices,
        cmap_mask=bundle.cmap_mask,
        cmap_coeffs=bundle.cmap_coeffs,
        excl_indices=bundle.excl_dense_indices,
        excl_scales_vdw=bundle.excl_dense_scales_vdw,
        excl_scales_elec=bundle.excl_dense_scales_elec,
        excl_pair_indices=excl_pair_indices,
        excl_pair_scales_vdw=bundle.excl_scales_vdw,
        excl_pair_scales_elec=bundle.excl_scales_elec,
        constraint_pairs=bundle.constraint_pairs,
        constraint_lengths=bundle.constraint_lengths,
        constraint_mask=bundle.constraint_mask,
        exception_pairs=bundle.exception_pairs,
        exception_sigmas=bundle.exception_sigmas,
        exception_epsilons=bundle.exception_epsilons,
        exception_chargeprods=bundle.exception_chargeprods,
        exception_mask=bundle.exception_mask,
        cosolvent_group_id=_pad_1d(data.cosolvent_group_id, n, dtype=jnp.int32),
        cosolvent_mol_id=_pad_1d(data.molecule_id, n, dtype=jnp.int32),
        n_real_atoms=bundle.n_atoms,
        n_padded_atoms=n,
        bucket_size=n,
        water_indices=bundle.water_indices,
        water_mask=bundle.water_mask,
        box_size=box_size_static,
        pme_alpha=float(data.pme_alpha),
        pme_grid_points=pme_grid_points,
        nonbonded_cutoff=float(data.nonbonded_cutoff),
        lj_switch_width=float(data.lj_switch_width),
        electrostatic_method=data.electrostatic_method,
        reaction_field_dielectric=float(data.reaction_field_dielectric),
        use_dispersion_correction=bool(data.use_dispersion_correction),
        solvent_dielectric=float(data.solvent_dielectric),
        solute_dielectric=float(data.solute_dielectric),
        surface_tension=float(data.surface_area_energy),
        dense_excl_scale_vdw=dense_vdw,
        dense_excl_scale_elec=dense_elec,
    )


__all__ = [
    "BAR_TO_ATM",
    "DEFAULT_TIP3P_R_HH",
    "DEFAULT_TIP3P_R_OH",
    "KJ_NM2_TO_KCAL_A2",
    "KJ_TO_KCAL",
    "NM_TO_ANGSTROM",
    "BarostatSpec",
    "OpenMMSystemData",
    "bundle_from_openmm_system_xml",
    "parse_openmm_system_xml",
    "physics_system_from_openmm_xml",
]
