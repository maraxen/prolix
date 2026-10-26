"""NonbondedConfig host contract (Track A0)."""

from __future__ import annotations

import pytest

from prolix.physics.nonbonded_config import (
    NeighborListConfig,
    NonbondedConfig,
    NonbondedEngine,
    resolve_nonbonded_config,
)


def test_nl_engine_requires_nl_block():
    with pytest.raises(ValueError, match="requires neighbor_list"):
        NonbondedConfig(engine=NonbondedEngine.NEIGHBOR_LIST)


def test_flash_rejects_nl_block():
    with pytest.raises(ValueError, match="FLASH cannot"):
        NonbondedConfig(
            engine=NonbondedEngine.FLASH,
            neighbor_list=NeighborListConfig(cutoff=9.0),
        )


def test_mixing_nonbonded_and_legacy_flags_raises():
    cfg = NonbondedConfig(engine=NonbondedEngine.FLASH)
    with pytest.raises(ValueError, match="not both"):
        resolve_nonbonded_config(
            engine_nl=False,
            engine_flash=True,
            cutoff=9.0,
            nl_update_every=20,
            nonbonded=cfg,
            used_legacy_flags=True,
        )


def test_legacy_nl_keeps_half_angstrom_skin():
    cfg = resolve_nonbonded_config(
        engine_nl=True,
        engine_flash=False,
        cutoff=9.0,
        nl_update_every=20,
        nonbonded=None,
        used_legacy_flags=False,
    )
    assert cfg.engine is NonbondedEngine.NEIGHBOR_LIST
    assert cfg.neighbor_list is not None
    assert cfg.neighbor_list.skin == 0.5
    assert cfg.neighbor_list.switch_width is None
    assert cfg.neighbor_list.nl_update_every == 20


def test_lj_switch_width_from_config():
    from prolix.physics.nonbonded_config import lj_switch_width_from_config

    assert lj_switch_width_from_config(None) == 0.0
    dense = NonbondedConfig(engine=NonbondedEngine.DENSE)
    assert lj_switch_width_from_config(dense) == 0.0
    nlc = NeighborListConfig(cutoff=9.0, switch_width=1.0)
    cfg = NonbondedConfig(engine=NonbondedEngine.NEIGHBOR_LIST, neighbor_list=nlc)
    assert lj_switch_width_from_config(cfg) == 1.0
