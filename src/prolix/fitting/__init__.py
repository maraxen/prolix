"""Fitting module: differentiable bonded force-field parameter optimization.

Provides:
  - BondedTopology: static bonded connectivity
  - BondedParams: trainable bonded parameters (equinox Module)
  - BondedParamsBundle, BondedTopologyBundle: stacked params/topology for vmap
  - bonded_energy: JAX-pure energy computation
  - bonded_loss: per-molecule loss with energy + force + regularization
  - load_params_init_json: load Phase A parameterization from JSON
  - TrainState: immutable training state for scan
  - scheduler: conformer sampling
  - train: training loops (step, scan, looped, batched)
"""

from prolix.fitting.batched import (
    BondedParamsBundle,
    BondedTopologyBundle,
    stack_molecules,
    unbatch_params,
)
from prolix.fitting.energy import bonded_energy
from prolix.fitting.init import load_params_init_json
from prolix.fitting.loss import bonded_loss, default_sigma
from prolix.fitting.params import BondedParams
from prolix.fitting.scheduler import ConformerBatch, sample_one_conformer
from prolix.fitting.state import TrainState
from prolix.fitting.topology import BondedTopology
from prolix.fitting.train import (
    TrainMetrics,
    train_loop_batched,
    train_loop_looped_baseline,
    train_loop_one_mol,
    train_step_one_mol,
)

__all__ = [
    "BondedTopology",
    "BondedParams",
    "BondedParamsBundle",
    "BondedTopologyBundle",
    "stack_molecules",
    "unbatch_params",
    "bonded_energy",
    "bonded_loss",
    "default_sigma",
    "load_params_init_json",
    "TrainState",
    "ConformerBatch",
    "sample_one_conformer",
    "TrainMetrics",
    "train_step_one_mol",
    "train_loop_one_mol",
    "train_loop_looped_baseline",
    "train_loop_batched",
]
