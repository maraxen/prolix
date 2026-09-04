"""Host-side nonbonded engine config (closed over outside JIT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class NonbondedEngine(str, Enum):
    DENSE = "dense"
    NEIGHBOR_LIST = "neighbor_list"
    FLASH = "flash"


OverflowPolicy = Literal["reallocate", "raise"]


@dataclass(frozen=True)
class NeighborListConfig:
    cutoff: float
    skin: float = 1.0
    switch_width: float | None = None
    capacity_multiplier: float = 1.25
    safety_factor: float = 1.5
    # backlog #4699 RESOLVED: the cell list dropped pairs because jax_md bins on
    # raw position (truncating toward zero, no wrap), while prolix passes
    # origin-centred coordinates. make_neighbor_list_fn now wraps into [0, box)
    # before handing positions to jax_md, which restores the exact brute-force
    # candidate set. Cell list is safe and on by default; True falls back to
    # O(N^2) construction.
    disable_cell_list: bool = False
    max_exclusions: int = 32
    nl_update_every: int | None = None
    on_overflow: OverflowPolicy = "reallocate"


@dataclass(frozen=True)
class NonbondedConfig:
    engine: NonbondedEngine
    neighbor_list: NeighborListConfig | None = None
    flash_tile_size: int = 256
    flash_remat: bool = True

    def __post_init__(self) -> None:
        if self.engine is NonbondedEngine.NEIGHBOR_LIST and self.neighbor_list is None:
            raise ValueError("NonbondedEngine.NEIGHBOR_LIST requires neighbor_list=")
        if self.engine is NonbondedEngine.FLASH and self.neighbor_list is not None:
            raise ValueError("FLASH cannot carry a NeighborListConfig")


def legacy_neighbor_list_config(
    cutoff: float,
    nl_update_every: int = 20,
) -> NeighborListConfig:
    """Flag-based EnsemblePlan path: keep 0.5 Å skin and step-stride default."""
    return NeighborListConfig(
        cutoff=float(cutoff),
        skin=0.5,
        switch_width=None,
        nl_update_every=int(nl_update_every),
    )


def resolve_nonbonded_config(
    *,
    engine_nl: bool,
    engine_flash: bool,
    cutoff: float,
    nl_update_every: int,
    nonbonded: NonbondedConfig | None,
    used_legacy_flags: bool,
) -> NonbondedConfig:
    """Build the host config. Mixing ``nonbonded=`` with explicit engine flags raises."""
    if nonbonded is not None and used_legacy_flags:
        raise ValueError(
            "Pass either nonbonded=NonbondedConfig(...) or the legacy "
            "use_neighbor_list/use_flash_forces/nl_update_every kwargs, not both"
        )
    if nonbonded is not None:
        return nonbonded
    if engine_nl and engine_flash:
        raise ValueError("neighbor_list and flash engines are mutually exclusive")
    if engine_nl:
        return NonbondedConfig(
            engine=NonbondedEngine.NEIGHBOR_LIST,
            neighbor_list=legacy_neighbor_list_config(cutoff, nl_update_every),
        )
    if engine_flash:
        return NonbondedConfig(engine=NonbondedEngine.FLASH)
    return NonbondedConfig(engine=NonbondedEngine.DENSE)


def engine_flags(cfg: NonbondedConfig) -> tuple[bool, bool]:
    return (
        cfg.engine is NonbondedEngine.NEIGHBOR_LIST,
        cfg.engine is NonbondedEngine.FLASH,
    )


def lj_switch_width_from_config(cfg: NonbondedConfig | None) -> float:
    """Closed-over LJ switch width for ``energy_fn_from_bundle``. Off → 0.0."""
    if cfg is None:
        return 0.0
    nlc = cfg.neighbor_list
    if nlc is None or nlc.switch_width is None:
        return 0.0
    return float(nlc.switch_width)
