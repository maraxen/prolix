"""Post-release scaffold for L3 explicit-solvent observables (e.g., RDF).

See ``docs/source/explicit_solvent/l3_observables_protocol.md``.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="L3 RDF observables are post-release per validation plan P2b")
def test_explicit_rdf_water_water_placeholder():
  """Placeholder: compare water–water RDF vs a reference OpenMM trajectory."""
  raise AssertionError("not implemented — post-release")
