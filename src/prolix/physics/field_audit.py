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

from typing import FrozenSet, Any
import jax.numpy as jnp
from jax_md import space


class FieldAuditProxy:
    """Non-intrusive proxy wrapper around PhysicsSystem for field-access logging.

    Intercepts __getattr__ calls to record which attributes are read.
    """

    def __init__(self, system: Any):
        """Initialize proxy wrapping a PhysicsSystem.

        Args:
            system: PhysicsSystem instance to wrap.
        """
        object.__setattr__(self, '_system', system)
        object.__setattr__(self, '_accessed_fields', set())

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access and log field reads.

        Args:
            name: Attribute name being accessed.

        Returns:
            The attribute value from the wrapped system.
        """
        # Avoid recursion on special attributes
        if name in ('_system', '_accessed_fields'):
            return object.__getattribute__(self, name)

        # Log the access
        accessed = object.__getattribute__(self, '_accessed_fields')
        accessed.add(name)

        # Return the attribute from the wrapped system
        system = object.__getattribute__(self, '_system')
        return getattr(system, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of wrapped system.

        Args:
            name: Attribute name.
            value: Attribute value.

        Raises:
            AttributeError: Always, to prevent modification.
        """
        if name in ('_system', '_accessed_fields'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Proxy is read-only; cannot set attributes")

    def get_accessed_fields(self) -> FrozenSet[str]:
        """Return the set of fields accessed so far.

        Returns:
            Frozenset of field names.
        """
        accessed = object.__getattribute__(self, '_accessed_fields')
        return frozenset(accessed)


def audit_bonded_fields(system: Any, displacement_fn: space.DisplacementFn,
                        positions: Any) -> FrozenSet[str]:
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
    from prolix.physics import bonded

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
