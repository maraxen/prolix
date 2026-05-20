"""Fitting module: differentiable bonded force-field parameter optimization.

Provides:
  - BondedTopology: static bonded connectivity
  - BondedParams: trainable bonded parameters (equinox Module)
  - bonded_energy: JAX-pure energy computation
  - bonded_loss: per-molecule loss with energy + force + regularization
  - load_params_init_json: load Phase A parameterization from JSON
"""

from prolix.fitting.energy import bonded_energy
from prolix.fitting.init import load_params_init_json
from prolix.fitting.loss import bonded_loss, default_sigma
from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology

__all__ = [
    "BondedTopology",
    "BondedParams",
    "bonded_energy",
    "bonded_loss",
    "default_sigma",
    "load_params_init_json",
]
