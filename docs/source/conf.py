"""Configuration file for the Prolix Sphinx documentation builder."""

from __future__ import annotations

import importlib.util
import os
import re
import sys

# Ensure local source tree (and optional sibling workspace checkouts on HPC)
# takes precedence during autodoc imports.
for _rel_src in ("../../src", "../../../xtrax/src"):
    _abs_src = os.path.abspath(os.path.join(os.path.dirname(__file__), _rel_src))
    if os.path.isdir(_abs_src) and _abs_src not in sys.path:
        sys.path.insert(0, _abs_src)

# Force JAX onto CPU during documentation builds so ReadTheDocs / headless
# builders never probe CUDA / Triton devices at import time.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# -- Project information -----------------------------------------------------
project = "Prolix"
copyright = "2026, Marielle Russo"
author = "Marielle Russo"
version = "1.0.0"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]
try:
    if importlib.util.find_spec("sphinxcontrib.mermaid") is not None:
        extensions.append("sphinxcontrib.mermaid")
except (ImportError, ValueError):
    pass

templates_path: list[str] = []
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path: list[str] = []
html_title = "Prolix Documentation"

html_theme_options = {
    "repository_url": "https://github.com/maraxen/prolix",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "use_download_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

# -- Dynamic mock imports for optional / GPU-only dependencies ---------------
# Only mock packages that are genuinely absent from the active Python env so
# full environments (e.g., Engaging CI) inspect real classes and signatures,
# while lightweight CPU doc builders never fail on missing native extensions.
_OPTIONAL_OR_GPU_PACKAGES = (
    "triton",
    "cuda",
    "nvidia",
    "iree",
    "kups",
    "dmff",
    "torch",
    "torchmd",
    "espaloma",
    "dgl",
    "proxide",
    "xtrax",
    "openmm",
    "openmmtools",
    "parmed",
    "mdtraj",
    "rdkit",
    "pymbar",
    "pdbfixer",
    "netCDF4",
    "mpiplus",
    "numba",
    "py2Dmol",
    "jaxopt",
    "biotite",
    "h5py",
    "msgpack_numpy",
    "msgpack",
    "jax",
    "jax_md",
    "equinox",
    "flax",
    "optax",
    "ml_collections",
    "absl",
    "jaxtyping",
    "numpy",
    "scipy",
    "chex",
    "matplotlib",
    "imageio",
    "pandas",
    "yaml",
    "termcolor",
)

autodoc_mock_imports: list[str] = []
for _pkg in _OPTIONAL_OR_GPU_PACKAGES:
    try:
        if importlib.util.find_spec(_pkg) is None:
            autodoc_mock_imports.append(_pkg)
    except (ImportError, ValueError):
        autodoc_mock_imports.append(_pkg)


# Patch Sphinx's ``_MockObject`` with arithmetic and item-assignment dunders so
# module-level unit expressions (e.g., ``u.nanometer / u.picosecond``,
# ``u.nanometer**2``, ``10.0 * unit.kilojoules_per_mole / unit.nanometer`` in
# ``prolix.openmm.units`` and ``prolix.openmm.simulation``) evaluate cleanly in
# docs-only CPU environments where ``openmm`` or ``numpy`` is mocked.
def _patch_sphinx_mock_object() -> None:
    def _return_self(self: object, *_args: object) -> object:
        return self

    def _noop_setitem(_self: object, _key: object, _val: object) -> None:
        return None

    for _mod_name in ("sphinx.ext.autodoc.mock", "sphinx.ext.autodoc._mock"):
        try:
            _mock_mod = importlib.import_module(_mod_name)
        except ImportError:
            continue
        _mock_cls = getattr(_mock_mod, "_MockObject", None)
        if _mock_cls is None:
            continue
        if getattr(_mock_cls, "_prolix_patched", False):
            continue
        _mock_cls.__doc__ = None
        _orig_getattr = _mock_cls.__getattr__

        def _safe_mock_getattr(
            self: object,
            key: str,
            _orig: object = _orig_getattr,
        ) -> object:
            if key == "__version__":
                return "0.0.0"
            if key.startswith("__") and key.endswith("__"):
                raise AttributeError(key)
            return _orig(self, key)  # type: ignore[operator]

        _mock_cls.__getattr__ = _safe_mock_getattr
        for _dunder in (
            "__truediv__",
            "__rtruediv__",
            "__floordiv__",
            "__rfloordiv__",
            "__mul__",
            "__rmul__",
            "__matmul__",
            "__rmatmul__",
            "__pow__",
            "__rpow__",
            "__add__",
            "__radd__",
            "__sub__",
            "__rsub__",
            "__neg__",
            "__pos__",
            "__abs__",
        ):
            if not hasattr(_mock_cls, _dunder):
                setattr(_mock_cls, _dunder, _return_self)
        if not hasattr(_mock_cls, "__setitem__"):
            _mock_cls.__setitem__ = _noop_setitem
        _mock_cls._prolix_patched = True


_patch_sphinx_mock_object()

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# Autodoc & Autosummary settings (built-in PEP 563 / jaxtyping safe typehints)
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autosummary_generate = False

# MyST & MyST-NB settings
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 3

# Intersphinx: fetch remote inventories on ReadTheDocs; keep offline/HPC builds
# deterministic so restricted-network nodes never emit fetch warnings under -W.
if os.environ.get("READTHEDOCS") == "True":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "jax": ("https://docs.jax.dev/en/latest/", None),
    }
else:
    intersphinx_mapping = {}

suppress_warnings = [
    "myst.mathjax",
]

# Sanitize Markdown-style docstring idioms into valid reStructuredText so
# autodoc never emits docutils inline-markup or list-indentation warnings
# under ``sphinx-build -W``:
#   1. Plural backticks (e.g., `System`s -> ``System``\ s)
#   2. Bare prose *args / **kwargs (e.g., F(R, **kwargs) -> F(R, ``**kwargs``))
#   3. Missing blank lines around bullet / enumerated lists and nested sublists
_BACKTICK_PLURAL_RE = re.compile(
    r"(``[^`\s][^`\n]*?[^`\s]``|``[^`\s]``|`[^`\s][^`\n]*?[^`\s]`|`[^`\s]`)(s|es)\b"
)
_BARE_VARARGS_RE = re.compile(r"(?<![`*\w:\\])(\*{1,2}(?:args|kwargs)\b)(?![`*])")
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+\.|[a-zA-Z]\.)\s+")
_RST_DIRECTIVE_OR_FIELD_RE = re.compile(r"^\s*(?::[^:]+:|\.\.\s)")


def _quote_bare_varargs(line: str) -> str:
    if "args" not in line or _RST_DIRECTIVE_OR_FIELD_RE.match(line):
        # Still handle F(R, **kwargs) inside the body of a `:param name: ...` line
        if ":param " in line and "**kwargs" in line:
            head, sep, tail = line.partition(": ")
            if sep:
                return head + sep + _BARE_VARARGS_RE.sub(r"``\1``", tail)
        return line
    parts = re.split(r"(``[^`]*``|`[^`]*`)", line)
    for idx in range(0, len(parts), 2):
        parts[idx] = _BARE_VARARGS_RE.sub(r"``\1``", parts[idx])
    return "".join(parts)


def _sanitize_autodoc_docstring(
    _app: object,
    _what: str,
    _name: str,
    _obj: object,
    _options: object,
    lines: list[str],
) -> None:
    out: list[str] = []
    list_indent: int | None = None
    body_indent: int | None = None

    for raw_line in lines:
        line = raw_line
        if "`" in line:
            line = _BACKTICK_PLURAL_RE.sub(
                lambda m: f"``{m.group(1).strip('`')}``\\ {m.group(2)}",
                line,
            )
        if "*" in line and "args" in line:
            line = _quote_bare_varargs(line)

        if line.strip() == "":
            out.append(line)
            list_indent = None
            body_indent = None
            continue

        m = _LIST_ITEM_RE.match(line)
        if m:
            cur_indent = len(m.group(1))
            cur_body_indent = len(m.group(0))
            if out and out[-1].strip() != "" and list_indent != cur_indent:
                out.append("")
            out.append(line)
            list_indent = cur_indent
            body_indent = cur_body_indent
        else:
            cur_indent = len(line) - len(line.lstrip(" "))
            if list_indent is not None and (
                body_indent is None or cur_indent < body_indent
            ):
                if out and out[-1].strip() != "":
                    out.append("")
                list_indent = None
                body_indent = None
            out.append(line)

    lines[:] = out


def setup(app: object) -> None:
    app.connect("autodoc-process-docstring", _sanitize_autodoc_docstring, priority=600)  # type: ignore[attr-defined]
    if "sphinxcontrib.mermaid" not in extensions:
        import html as _html

        from docutils import nodes
        from docutils.parsers.rst import Directive, directives

        class _MermaidFallbackDirective(Directive):
            has_content = True
            optional_arguments = 1
            final_argument_whitespace = True
            option_spec = {
                "align": directives.unchanged,
                "alt": directives.unchanged,
                "caption": directives.unchanged,
                "name": directives.unchanged,
                "zoom": directives.flag,
            }

            def run(self) -> list[nodes.Node]:
                diagram = _html.escape("\n".join(self.content))
                raw_html = f'<div class="mermaid">\n<pre class="mermaid">{diagram}</pre>\n</div>'
                return [nodes.raw("", raw_html, format="html")]

        app.add_directive("mermaid", _MermaidFallbackDirective)  # type: ignore[attr-defined]


