# Prolix Documentation

Welcome to the Prolix documentation!

```{toctree}
:maxdepth: 2
:caption: Contents

source/index
```

## Overview

Prolix is a specialized library for Protein Physics and molecular dynamics simulations in JAX.

## Quick Links

- [API Reference](source/api/index.md)
- [Examples](source/examples/index.md)
- [GitHub Repository](https://github.com/maraxen/prolix)

## Getting Started

Install Prolix:

```bash
pip install prolix
```

Run your first simulation:

```python
from prolix.physics import system
from priox.physics import force_fields

# Load force field and run MD
ff = force_fields.load_force_field("ff19SB.eqx")
# ... (see examples for more)
```
