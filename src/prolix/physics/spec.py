from dataclasses import dataclass

from jax_md import util

Array = util.Array

@dataclass(frozen=True)
class PhysicsSpec:
    """Configuration for all physics-related parameters."""
    dt: float = 2.0  # fs
    temperature_k: float = 300.0
    gamma: float = 1.0  # 1/ps
    rigid_water: bool = False
    remove_linear_com_momentum: bool = False
    settle_velocity_tol: float | None = None
    
    # PME / PBC / Electrostatics
    pme_alpha: float = 0.34
    neighbor_cutoff: float = 9.0
    
    @property
    def kT(self) -> float:
        # BOLTZMANN_KCAL = 0.0019872041
        return self.temperature_k * 0.0019872041
