"""Plotting utilities for trajectory analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from prolix import analysis

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  from .trajectory import TrajectoryReader


def plot_rmsd(
  trajectory: TrajectoryReader,
  reference_positions: np.ndarray | None = None,
  mask: np.ndarray | None = None,
  output_path: str | None = None,
) -> plt.Figure:
  """Plot RMSD over time."""
  positions = trajectory.get_positions()  # (T, N, 3)
  times = trajectory.get_time()

  # If no reference, use first frame
  if reference_positions is None:
    reference_positions = positions[0]

  # Convert to JAX arrays for analysis
  pos_jax = jnp.array(positions)
  ref_jax = jnp.array(reference_positions)
  mask_jax = jnp.array(mask) if mask is not None else None

  # Compute RMSD
  # analysis.compute_rmsd expects (positions, reference)
  # If positions is (T, N, 3), it broadcasts?
  # analysis.compute_rmsd:
  # diff = p - r  -> (T, N, 3) - (N, 3) = (T, N, 3) works.
  # sq_dist = sum(diff**2, axis=-1) -> (T, N)
  # mean = mean(..., axis=-1) -> (T,)
  # sqrt -> (T,)
  rmsd = analysis.compute_rmsd(pos_jax, ref_jax, mask_jax)

  rmsd_np = np.array(rmsd)

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(times, rmsd_np)
  ax.set_xlabel("Time (ns)")
  ax.set_ylabel("RMSD (Å)")
  ax.set_title("RMSD vs Time")
  ax.grid(visible=True, alpha=0.3)

  if output_path:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

  return fig


def plot_energy(trajectory: TrajectoryReader, output_path: str | None = None) -> plt.Figure:
  """Plot Potential and Kinetic Energy."""
  energies = trajectory.get_energies()
  times = trajectory.get_time()

  fig, ax = plt.subplots(figsize=(8, 4))

  if energies["potential"] is not None:
    ax.plot(times, energies["potential"], label="Potential")

  if energies["kinetic"] is not None:
    ax.plot(times, energies["kinetic"], label="Kinetic")

  if energies["potential"] is not None and energies["kinetic"] is not None:
    total = energies["potential"] + energies["kinetic"]
    ax.plot(times, total, label="Total", color="black", alpha=0.5, linestyle="--")

  ax.set_xlabel("Time (ns)")
  ax.set_ylabel("Energy (kcal/mol)")
  ax.set_title("Energy vs Time")
  ax.legend()
  ax.grid(visible=True, alpha=0.3)

  if output_path:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

  return fig


def plot_contact_map(
  trajectory: TrajectoryReader, frame_idx: int = -1, output_path: str | None = None
) -> plt.Figure:
  """Plot contact map for a specific frame (default: last)."""
  state = trajectory.get_state(frame_idx)
  pos = jnp.array(state.positions)

  cmap = analysis.compute_contact_map(pos)

  fig, ax = plt.subplots(figsize=(6, 6))
  ax.imshow(cmap, origin="lower", cmap="Greys")
  ax.set_xlabel("Atom Index")
  ax.set_ylabel("Atom Index")
  ax.set_title(f"Contact Map (t={state.time_ns:.3f} ns)")

  if output_path:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

  return fig


def plot_ramachandran(
  trajectory: TrajectoryReader,
  phi_indices: np.ndarray,
  psi_indices: np.ndarray,
  output_path: str | None = None,
) -> plt.Figure:
  """Plot Ramachandran (Phi vs Psi).

  Args:
      trajectory: Reader
      phi_indices: (M, 4) atom indices for Phi angles
      psi_indices: (M, 4) atom indices for Psi angles
      output_path: Output file path

  """
  positions = jnp.array(trajectory.get_positions())  # (T, N, 3)

  # analysis.compute_dihedrals handles (..., N, 3) and returns (..., M)
  phi = analysis.compute_dihedrals(positions, jnp.array(phi_indices))
  psi = analysis.compute_dihedrals(positions, jnp.array(psi_indices))

  # Flatten time dimension for density plot
  phi_flat = np.array(phi).flatten()
  psi_flat = np.array(psi).flatten()

  fig, ax = plt.subplots(figsize=(6, 6))
  # Convert to degrees
  phi_deg = np.degrees(phi_flat)
  psi_deg = np.degrees(psi_flat)

  ax.hist2d(
    phi_deg, psi_deg, bins=60, range=[[-180, 180], [-180, 180]], cmap="viridis", density=True
  )

  ax.set_xlim(-180, 180)
  ax.set_ylim(-180, 180)
  ax.set_xlabel("Phi (degrees)")
  ax.set_ylabel("Psi (degrees)")
  ax.set_title("Ramachandran Plot")
  ax.grid(visible=True, alpha=0.3)

  if output_path:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

  return fig


def plot_free_energy_surface(
  trajectory: TrajectoryReader,
  metric_fn: Callable[[Any], Any],
  bins: int = 50,
  temperature: float = 300.0,
  output_path: str | None = None,
) -> plt.Figure:
  """Plot 1D Free Energy Surface of a metric."""
  positions = jnp.array(trajectory.get_positions())
  metric_vals = metric_fn(positions)  # Should return (T,)

  centers, fes = analysis.compute_free_energy_surface(
    metric_vals, temperature=temperature, bins=bins
  )

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.plot(centers, fes)
  ax.set_xlabel("Reaction Coordinate")
  ax.set_ylabel("Free Energy (kcal/mol)")
  ax.set_title("Free Energy Surface")
  ax.grid(visible=True, alpha=0.3)

  if output_path:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

  return fig
