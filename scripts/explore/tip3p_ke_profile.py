"""Shared TIP3P KE / Langevin benchmark profile IDs and OpenMM COM metadata helpers."""

from __future__ import annotations

PROFILE_DIAG_LINEAR_COM_OFF = "diag_linear_com_off"
PROFILE_OPENMM_REF_LINEAR_COM_ON = "openmm_ref_linear_com_on"


def profile_id_from_remove_cmmotion(remove_cmmotion: bool) -> str:
  """Map OpenMM ``removeCMMotion`` policy to the canonical benchmark profile id."""
  return PROFILE_OPENMM_REF_LINEAR_COM_ON if remove_cmmotion else PROFILE_DIAG_LINEAR_COM_OFF


def openmm_system_has_cmmotion_remover(omm_system, openmm_mod) -> bool:
  cm_cls = getattr(openmm_mod, "CMMotionRemover", None)
  if cm_cls is None:
    return False
  for i in range(omm_system.getNumForces()):
    if isinstance(omm_system.getForce(i), cm_cls):
      return True
  return False


def assert_runs_profile_consistency(runs: list[dict], *, context: str = "") -> None:
  """Fail closed: every run dict must carry the same ``profile_id`` when present."""
  ids = [r.get("profile_id") for r in runs if r.get("profile_id") is not None]
  if not ids:
    msg = f"{context}: missing profile_id on all runs"
    raise ValueError(msg)
  if len(set(ids)) != 1:
    msg = f"{context}: mixed profile_id values: {ids!r}"
    raise ValueError(msg)
