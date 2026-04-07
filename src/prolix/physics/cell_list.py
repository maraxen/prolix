"""Cell-list decomposition for explicit solvent MD.

Assigns atoms to 3D spatial cells for O(N) neighbor finding.
Each cell has a fixed capacity M (max_atoms_per_cell). Atoms
beyond capacity are silently dropped (must be monitored via
overflow detection).

Ghost atom convention:
  - Ghost positions are placed at box_center (not infinity)
  - Ghost sigmas = 1.0 (not 1e-6)
  - Ghost epsilons = 0.0
  This prevents cell-index overflow and LJ overflow in the kernel.

Usage:
  cells = build_cell_list(positions, box_size, atom_mask, ...)
  # cells.occupancy: (Nx, Ny, Nz, M) int32  -- atom indices
  # cells.offsets:   (Nx, Ny, Nz, M, 3) float32 -- positions
  # cells.counts:    (Nx, Ny, Nz) int32 -- atoms per cell
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CellList(NamedTuple):
    """Cell-list data structure for spatial decomposition.

    Attributes:
        occupancy: (Nx, Ny, Nz, M) int32 — global atom indices per cell.
            Padding slots filled with N (sentinel, points to ghost atom).
        positions: (Nx, Ny, Nz, M, 3) float32 — atom positions per cell.
            Padding slots at box_center.
        counts: (Nx, Ny, Nz) int32 — number of real atoms in each cell.
        sigmas: (Nx, Ny, Nz, M) float32 — LJ sigma per cell slot.
        epsilons: (Nx, Ny, Nz, M) float32 — LJ epsilon per cell slot.
        charges: (Nx, Ny, Nz, M) float32 — partial charges per cell slot.
        mask: (Nx, Ny, Nz, M) bool — True for real atoms in each slot.
        cell_size: float32 — edge length of each cubic cell.
        grid_shape: tuple[int, int, int] — (Nx, Ny, Nz).
        overflow: bool — True if any cell exceeded capacity M.
    """
    occupancy: jnp.ndarray   # (Nx, Ny, Nz, M) int32
    positions: jnp.ndarray   # (Nx, Ny, Nz, M, 3) float32
    counts: jnp.ndarray      # (Nx, Ny, Nz) int32
    sigmas: jnp.ndarray      # (Nx, Ny, Nz, M) float32
    epsilons: jnp.ndarray    # (Nx, Ny, Nz, M) float32
    charges: jnp.ndarray     # (Nx, Ny, Nz, M) float32
    mask: jnp.ndarray        # (Nx, Ny, Nz, M) bool
    cell_size: float
    grid_shape: tuple[int, int, int]
    overflow: jnp.ndarray    # scalar bool


def compute_grid_shape(
    box_size: jnp.ndarray,
    cutoff: float,
    min_cells: int = 3,
) -> tuple[int, int, int]:
    """Compute cell grid dimensions from box size and cutoff.

    Each cell must be at least as large as the cutoff to ensure
    all interacting pairs are in the same or neighboring cells.

    Args:
        box_size: (3,) array of box dimensions in Å.
        cutoff: Interaction cutoff distance in Å.
        min_cells: Minimum cells per dimension (must be >= 3 for
            full 27-cell stencil without self-image).

    Returns:
        (Nx, Ny, Nz) grid shape as Python ints.
    """
    # Floor division: cell_size >= cutoff
    n_cells = jnp.floor(box_size / cutoff).astype(jnp.int32)
    n_cells = jnp.maximum(n_cells, min_cells)
    return (int(n_cells[0]), int(n_cells[1]), int(n_cells[2]))


def compute_cell_size(
    box_size: jnp.ndarray,
    grid_shape: tuple[int, int, int],
) -> jnp.ndarray:
    """Compute actual cell size from box and grid dimensions.

    Returns:
        (3,) array of cell sizes per dimension.
    """
    grid = jnp.array(grid_shape, dtype=jnp.float32)
    return box_size / grid


def build_cell_list(
    positions: jnp.ndarray,
    box_size: jnp.ndarray,
    atom_mask: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    charges: jnp.ndarray,
    cutoff: float,
    max_atoms_per_cell: int = 32,
    grid_shape: tuple[int, int, int] | None = None,
) -> CellList:
    """Build a cell list from atom positions.

    Ghost atom sanitization contract:
      - Ghost positions → box_center (not infinity)
      - Ghost sigmas → 1.0 (not 1e-6, prevents LJ overflow)
      - Ghost epsilons → 0.0
      - Ghost charges → 0.0

    This function is NOT JIT-compatible (uses Python int for shapes).
    It runs once per neighbor-list rebuild (typically every 10-50 steps).

    Args:
        positions: (N, 3) atom positions in Å.
        box_size: (3,) periodic box dimensions.
        atom_mask: (N,) boolean mask for real atoms.
        sigmas: (N,) LJ sigma parameters.
        epsilons: (N,) LJ epsilon parameters.
        charges: (N,) partial charges.
        cutoff: Interaction cutoff in Å.
        max_atoms_per_cell: Fixed capacity M per cell.
        grid_shape: Optional override for (Nx, Ny, Nz). If None,
            computed from box_size / cutoff.

    Returns:
        CellList with all fields populated.
    """
    N = positions.shape[0]
    M = max_atoms_per_cell

    # --- Ghost atom sanitization (API contract) ---
    box_center = box_size / 2.0
    safe_positions = jnp.where(
        atom_mask[:, None], positions, box_center[None, :]
    )
    safe_sigmas = jnp.where(atom_mask, sigmas, jnp.float32(1.0))
    safe_epsilons = jnp.where(atom_mask, epsilons, jnp.float32(0.0))
    safe_charges = jnp.where(atom_mask, charges, jnp.float32(0.0))

    # --- Wrap positions into [0, box_size) ---
    wrapped = safe_positions % box_size[None, :]

    # --- Compute grid ---
    if grid_shape is None:
        grid_shape = compute_grid_shape(box_size, cutoff)
    Nx, Ny, Nz = grid_shape
    cell_sizes = compute_cell_size(box_size, grid_shape)

    # --- Assign atoms to cells ---
    # Cell indices for each atom: (N, 3) int32
    cell_idx = jnp.floor(wrapped / cell_sizes[None, :]).astype(jnp.int32)
    # Clamp to valid range (safety for numerical edge cases)
    cell_idx = jnp.clip(
        cell_idx,
        jnp.array([0, 0, 0]),
        jnp.array([Nx - 1, Ny - 1, Nz - 1]),
    )

    # --- Build cell list via scatter ---
    # Linearize cell index for scatter operations
    linear_cell = (
        cell_idx[:, 0] * (Ny * Nz)
        + cell_idx[:, 1] * Nz
        + cell_idx[:, 2]
    )  # (N,) int32

    # Count atoms per cell
    n_cells_total = Nx * Ny * Nz
    counts_flat = jnp.zeros(n_cells_total, dtype=jnp.int32)

    # Use numpy for the scatter since this runs on CPU at rebuild time
    import numpy as np
    counts_np = np.zeros(n_cells_total, dtype=np.int32)
    linear_np = np.asarray(linear_cell)
    mask_np = np.asarray(atom_mask)

    # Initialize output arrays
    occ = np.full((n_cells_total, M), N, dtype=np.int32)  # sentinel = N
    pos_out = np.zeros((n_cells_total, M, 3), dtype=np.float32)
    sig_out = np.full((n_cells_total, M), 1.0, dtype=np.float32)
    eps_out = np.zeros((n_cells_total, M), dtype=np.float32)
    chg_out = np.zeros((n_cells_total, M), dtype=np.float32)
    msk_out = np.zeros((n_cells_total, M), dtype=bool)

    # Fill box_center into position padding
    bc = np.asarray(box_center)
    pos_out[:, :, :] = bc[None, None, :]

    overflow = False
    wrapped_np = np.asarray(wrapped)
    safe_sig_np = np.asarray(safe_sigmas)
    safe_eps_np = np.asarray(safe_epsilons)
    safe_chg_np = np.asarray(safe_charges)

    for i in range(N):
        if not mask_np[i]:
            continue
        c = linear_np[i]
        slot = counts_np[c]
        if slot >= M:
            overflow = True
            continue
        occ[c, slot] = i
        pos_out[c, slot] = wrapped_np[i]
        sig_out[c, slot] = safe_sig_np[i]
        eps_out[c, slot] = safe_eps_np[i]
        chg_out[c, slot] = safe_chg_np[i]
        msk_out[c, slot] = True
        counts_np[c] += 1

    # Reshape to 3D grid + M
    occ_3d = jnp.array(occ.reshape(Nx, Ny, Nz, M))
    pos_3d = jnp.array(pos_out.reshape(Nx, Ny, Nz, M, 3))
    cnt_3d = jnp.array(counts_np.reshape(Nx, Ny, Nz))
    sig_3d = jnp.array(sig_out.reshape(Nx, Ny, Nz, M))
    eps_3d = jnp.array(eps_out.reshape(Nx, Ny, Nz, M))
    chg_3d = jnp.array(chg_out.reshape(Nx, Ny, Nz, M))
    msk_3d = jnp.array(msk_out.reshape(Nx, Ny, Nz, M))

    return CellList(
        occupancy=occ_3d,
        positions=pos_3d,
        counts=cnt_3d,
        sigmas=sig_3d,
        epsilons=eps_3d,
        charges=chg_3d,
        mask=msk_3d,
        cell_size=float(cell_sizes[0]),  # assumes cubic cells
        grid_shape=grid_shape,
        overflow=jnp.array(overflow),
    )


# ==========================================================================
# 13 positive half-shell shift vectors (Newton's 3rd Law)
# ==========================================================================

# For a 3D stencil, there are 27 neighbors (including self).
# Self is handled separately. The remaining 26 split into 13 pairs
# related by inversion: (dx, dy, dz) and (-dx, -dy, -dz).
# We compute only the 13 "positive" shifts and apply Newton's 3rd law.
HALF_SHELL_SHIFTS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (1, -1, -1),
]


def get_half_shell_shifts() -> list[tuple[int, int, int]]:
    """Return the 13 positive half-shell shift vectors."""
    return HALF_SHELL_SHIFTS
