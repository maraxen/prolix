"""Field-access auditing for prolix PhysicsSystem energy functions.

This module provides non-intrusive field-access logging via a proxy wrapper
around PhysicsSystem. It is used to identify which fields are accessed when
computing bonded energies, enabling audit documentation of the field paths.

Limitations:
- Reads inside jax.jit after the first trace (cache hit) are NOT re-triggered.
  For audit purposes, the first trace enumerates all accessed fields, which is
  sufficient.
- Works at Python level (factory time); does not track field reads that occur
  inside JAX-compiled kernels after the initial factory setup.

Usage:
    from prolix.physics.field_audit import audit_bonded_fields

    fields_accessed = audit_bonded_fields(system, displacement_fn, positions)
    print(f"Accessed fields: {fields_accessed}")
"""

from __future__ import annotations

import dataclasses
from typing import Any

try:
    import equinox as eqx
    import jax.numpy as jnp
    from jax_md import space

    from prolix.physics import bonded
except ImportError:
    eqx = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    space = None  # type: ignore[assignment]
    bonded = None  # type: ignore[assignment]


class FieldAuditProxy:
    """Non-intrusive proxy wrapper around PhysicsSystem / MolecularBundle / KnobSurface for field-access logging.

    Intercepts __getattr__ calls to record which attributes are read, recursively wraps nested
    dataclass/bundle objects with a shared access log, and preserves proxy tracking across
    ``replace`` / ``_replace`` calls.
    """

    def __init__(
        self,
        system: Any,
        *,
        _accessed: set[str] | None = None,
        _prefix: str = "",
    ):
        """Initialize proxy wrapping a PhysicsSystem or config object."""
        object.__setattr__(self, "_system", system)
        object.__setattr__(
            self,
            "_accessed_fields",
            _accessed if _accessed is not None else set(),
        )
        object.__setattr__(self, "_prefix", _prefix)

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access and log field reads."""
        if name in ("_system", "_accessed_fields", "_prefix"):
            return object.__getattribute__(self, name)

        system = object.__getattribute__(self, "_system")
        accessed = object.__getattribute__(self, "_accessed_fields")
        prefix = object.__getattribute__(self, "_prefix")

        if name in ("replace", "_replace", "__replace__"):
            def _replace_wrapper(**changes: Any) -> FieldAuditProxy:
                if hasattr(system, "replace"):
                    new_sys = system.replace(**changes)
                elif hasattr(system, "_replace"):
                    new_sys = system._replace(**changes)
                elif dataclasses.is_dataclass(system):
                    new_sys = dataclasses.replace(system, **changes)
                else:
                    new_sys = eqx.tree_at(
                        lambda t: [getattr(t, k) for k in changes],
                        system,
                        list(changes.values()),
                    )
                return FieldAuditProxy(new_sys, _accessed=accessed, _prefix=prefix)

            return _replace_wrapper

        val = getattr(system, name)
        if callable(val):
            return val
        if (
            not name.startswith("_")
            and dataclasses.is_dataclass(val)
            and not isinstance(val, type)
            and (jnp is None or not isinstance(val, jnp.ndarray))
        ):
            sub_prefix = f"{prefix}.{name}" if prefix else name
            return FieldAuditProxy(val, _accessed=accessed, _prefix=sub_prefix)
        if not name.startswith("_"):
            key = f"{prefix}.{name}" if prefix else name
            accessed.add(key)
            if prefix:
                accessed.add(name)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of wrapped system."""
        if name in ("_system", "_accessed_fields", "_prefix"):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Proxy is read-only; cannot set attributes")

    def get_accessed_fields(self) -> frozenset[str]:
        """Return the set of fields accessed so far."""
        accessed = object.__getattribute__(self, "_accessed_fields")
        return frozenset(accessed)

    @property
    def accessed_fields(self) -> set[str]:
        """Mutable set of accessed field names (compatible with redsox FieldAuditProxy)."""
        return object.__getattribute__(self, "_accessed_fields")

    def unread_fields(self, candidate_fields: set[str] | frozenset[str] | None = None) -> set[str]:
        """Return fields on the wrapped dataclass that were never read."""
        system = object.__getattribute__(self, "_system")
        accessed = object.__getattribute__(self, "_accessed_fields")
        if candidate_fields is None:
            if dataclasses.is_dataclass(system):
                candidate_fields = {f.name for f in dataclasses.fields(system)}
            else:
                candidate_fields = {
                    k for k in dir(system) if not k.startswith("_") and not callable(getattr(system, k, None))
                }
        return set(candidate_fields) - accessed


def audit_bonded_fields(system: Any, displacement_fn: space.DisplacementFn,
                        positions: Any) -> frozenset[str]:
    """Run bonded energy functions on a system with field-access logging.

    Wraps the PhysicsSystem in a FieldAuditProxy, calls the per-term bonded
    energy functions, and returns the set of fields accessed.

    Note: This function uses the direct-path bonded energy factories (params
    at call time) from prolix.physics.bonded, bypassing make_energy_fn.

    Args:
        system: PhysicsSystem to audit.
        displacement_fn: JAX MD displacement function (free or periodic).
        positions: (N, 3) float64 array of atomic positions in Angstroms.

    Returns:
        Frozenset of field names accessed during bonded energy evaluation.
    """
    # Create auditing proxy
    proxy = FieldAuditProxy(system)

    # Import bonded energy factories

    # Define a combined bonded energy function that accesses fields via proxy
    def total_bonded_energy(r):
        e_total = 0.0

        # Bonds
        if proxy.bonds is not None and proxy.bonds.shape[0] > 0:
            bond_fn = bonded.make_bond_energy_fn(displacement_fn, proxy.bonds)
            e_total = e_total + bond_fn(r, proxy.bond_params)

        # Angles
        if proxy.angles is not None and proxy.angles.shape[0] > 0:
            angle_fn = bonded.make_angle_energy_fn(displacement_fn, proxy.angles)
            e_total = e_total + angle_fn(r, proxy.angle_params)

        # Dihedrals
        if proxy.dihedrals is not None and proxy.dihedrals.shape[0] > 0:
            dih_fn = bonded.make_dihedral_energy_fn(displacement_fn, proxy.dihedrals)
            e_total = e_total + dih_fn(r, proxy.dihedral_params)

        # Impropers (separate or mixed with dihedrals)
        if proxy.impropers is not None and proxy.impropers.shape[0] > 0:
            imp_fn = bonded.make_dihedral_energy_fn(displacement_fn, proxy.impropers)
            e_total = e_total + imp_fn(r, proxy.improper_params)

        return e_total

    # Convert positions to JAX array if needed
    if not isinstance(positions, jnp.ndarray):
        positions = jnp.array(positions, dtype=jnp.float64)

    # Call the energy function to trigger field reads
    _ = total_bonded_energy(positions)

    # Return the accessed fields
    return proxy.get_accessed_fields()
