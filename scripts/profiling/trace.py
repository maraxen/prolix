"""Pure trace parsing: named_scope wall-clock attribution + dispatch counts.

No prolix imports (grep-enforced in tests/profiling/test_claim_contract.py
and tests/profiling/test_trace_parse.py). See
.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section P4
for the full design rationale.

**How named_scope labels actually surface (resolved empirically BEFORE
writing this parser, per P4's own requirement to resolve this before, not
after):** neither as a nested Perfetto duration slice, nor as a
TraceAnnotation/XLA-Modules row. On this JAX install's CPU backend, the
EXECUTED trace's per-thunk events are named after their (post-fusion) HLO
op/thunk name (e.g. "wrapped_multiply", "reduce_add_fusion" -- carried in
each event's ``args["hlo_op"]``), which by itself names no named_scope at
all. The named_scope path survives instead as a ``/``-delimited prefix on
the separately-obtained COMPILED HLO text's per-instruction ``op_name``
metadata (e.g. ``metadata={op_name="jit(f)/outer_scope/inner_a/..."}``,
confirmed via ``jax.jit(fn).lower(*args).compile().as_text()``).
Attribution is therefore two-input, not trace-only: the executed trace
supplies exclusive durations/occurrence counts per thunk name; the compiled
HLO text supplies the thunk-name -> scope-path mapping (``scope_map_from_hlo_text``
below). Per spec P4's own fallback clause ("if labels surface as flat name
prefixes instead, state that exclusive time is computed by prefix-depth
attribution instead"): each entry-computation instruction's time is
attributed to the DEEPEST known scope label appearing in its op_name path,
resolved by following ``calls=``/``to_apply=`` references down through
nested computations when the top-level instruction itself carries no
metadata of its own. A resolved label is therefore always exactly one
string per instruction -- there is no mixed-scope ambiguity to arbitrate
under this rule.
"""

from __future__ import annotations

import re
from typing import Any

_HEADER_RE = re.compile(
    r"^(?:ENTRY\s+)?%([\w.\-]+)\s*\([^)]*\)\s*->\s*\S+\s*\{\s*$"
)
_CLOSE_RE = re.compile(r"^\}\s*$")
_INSTR_NAME_RE = re.compile(r"^\s*(ROOT\s+)?%([\w.\-]+)\s*=")
_OP_NAME_RE = re.compile(r'op_name="([^"]*)"')
_CALLS_RE = re.compile(r"(?:calls|to_apply)=%([\w.\-]+)")


def _split_computations(hlo_text: str) -> dict[str, list[str]]:
    """Split HLO text into {computation_name: [body_lines]}."""
    computations: dict[str, list[str]] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in hlo_text.splitlines():
        if current_name is None:
            m = _HEADER_RE.match(line.strip())
            if m:
                current_name = m.group(1)
                current_lines = []
            continue
        if _CLOSE_RE.match(line.strip()):
            computations[current_name] = current_lines
            current_name = None
            current_lines = []
            continue
        current_lines.append(line)
    return computations


def _find_instr_line(lines: list[str], instr_name: str) -> str | None:
    prefix_root = f"ROOT %{instr_name} ="
    prefix_plain = f"%{instr_name} ="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix_root) or stripped.startswith(prefix_plain):
            return line
    return None


def _find_root_instr_name(lines: list[str]) -> str | None:
    for line in lines:
        m = _INSTR_NAME_RE.match(line)
        if m and m.group(1):
            return m.group(2)
    return None


_TRANSFORM_WRAPPER_RE = re.compile(r"^\w+\((.*)\)$")


def _unwrap_transform(segment: str) -> str:
    """Peel JAX transformation wrappers from a path segment.

    Under vmap/jvp/grad/etc, a named_scope path segment is decorated as
    e.g. "vmap(jvp(dense_bonded_improper))" rather than the bare label --
    confirmed on a real multi-step, vmapped+autodiffed water-plan program
    (op_name="jit(_batched)/vmap(jvp(dense_bonded_improper))/reduce_sum").
    Repeatedly strips a "word(...)" wrapper until reaching a bare
    identifier (or giving up if the segment isn't of that shape).
    """
    while True:
        m = _TRANSFORM_WRAPPER_RE.match(segment)
        if not m:
            return segment
        segment = m.group(1)


def _deepest_known_label(op_name: str, known_labels: frozenset[str]) -> str | None:
    """Resolve the deepest named_scope label in an op_name path.

    Known limitations (#4330 Fix 2b):
    1. Scopes in certain optimization patterns may not resolve. Examples:
       - Scopes wrapping scan(..., unroll=N) at the outer level may be absorbed
         into fusion, e.g. flash_nonbonded_tiles (production flash_explicit.py:244)
       - Scopes inside conditional branches (if/else) backed by control flow may
         resolve inconsistently depending on the run (e.g. pme_bwd_gather appearing
         in some production traces but not others). Root cause: control flow with
         conditional execution patterns interacts with scope path rewriting in
         unpredictable ways under the XLA backend's optimization passes.

    Evidence: compiled HLO dump from a production-scale MD force computation shows
    these scopes simply absent from the op_name metadata, even though the underlying
    computation is verified correct and executes. No simple pattern (e.g.
    "replace X wrapper with Y") has been found to recover the labels -- the issue
    appears to be a structural fusion or control-flow interaction at the backend
    level.
    """
    parts = op_name.split("/")
    if parts and parts[0].startswith("jit("):
        parts = parts[1:]
    # Exclude the final segment (the leaf primitive name, e.g. "reduce_sum")
    # -- only path segments between the jit(...) prefix and the leaf can be
    # named_scope labels. Search from the end for the DEEPEST match.
    for seg in reversed(parts[:-1]):
        unwrapped = _unwrap_transform(seg)
        if unwrapped in known_labels:
            return unwrapped
    return None


def _resolve_scope(
    instr_name: str,
    lines: list[str],
    computations: dict[str, list[str]],
    known_labels: frozenset[str],
    depth: int = 0,
) -> str | None:
    if depth > 16:
        return None
    line = _find_instr_line(lines, instr_name)
    if line is None:
        return None
    op_name_match = _OP_NAME_RE.search(line)
    if op_name_match:
        return _deepest_known_label(op_name_match.group(1), known_labels)
    calls_match = _CALLS_RE.search(line)
    if calls_match:
        callee = calls_match.group(1)
        callee_lines = computations.get(callee)
        if callee_lines is not None:
            root_name = _find_root_instr_name(callee_lines)
            if root_name is not None:
                return _resolve_scope(
                    root_name, callee_lines, computations, known_labels, depth + 1
                )
    return None


def scope_map_from_hlo_text(
    hlo_text: str, known_labels: frozenset[str]
) -> dict[str, str | None]:
    """Map every instruction name (in every computation) to its scope.

    Returns {instruction_name: label_or_None}. An instruction's ``name``
    matches exactly the trace event's ``args["hlo_op"]`` field for the
    corresponding runtime thunk. Deliberately not ENTRY-only: a real
    integrator's step loop compiles to a ``while(...)`` instruction at the
    ENTRY level whose ``condition=``/``body=`` computations contain the
    actual per-step leaf ops (confirmed: on a real multi-step water-plan
    program, restricting this to the ENTRY computation resolved zero labels
    -- every settle_*-wrapped op lived inside the while-body computation,
    not ENTRY itself). Each instruction is resolved independently from
    wherever it lives, so a while-body's, fusion's, or ENTRY's instructions
    are all covered uniformly by the same lookup.
    """
    computations = _split_computations(hlo_text)
    result: dict[str, str | None] = {}
    for lines in computations.values():
        for line in lines:
            m = _INSTR_NAME_RE.match(line)
            if not m:
                continue
            instr_name = m.group(2)
            if instr_name in result:
                continue
            result[instr_name] = _resolve_scope(
                instr_name, lines, computations, known_labels
            )
    return result


def parse_scopes(
    trace_events: list[dict[str, Any]],
    scope_map: dict[str, str | None],
) -> dict[str, tuple[float, int]]:
    """Attribute executed wall-clock time to named_scope labels.

    Returns {label: (total_exclusive_seconds, n_occurrences)}, summed across
    every occurrence in ``trace_events`` (e.g. every repeated call in a
    multi-trial capture) -- the sum-across-re-entrant-occurrences convention
    P4's spec mandates. Perfetto trace_event durations (``dur``) are in
    microseconds; converted to seconds here. Only "X"-phase (complete)
    events whose name does not start with "end: " (XLA's own thunk-
    completion marker, a near-zero-duration bookkeeping event, not real
    work) and whose ``args["hlo_op"]`` resolves to a known label via
    ``scope_map`` are counted.
    """
    totals: dict[str, list[float]] = {}
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if isinstance(name, str) and name.startswith("end: "):
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        hlo_op = args.get("hlo_op")
        if hlo_op is None:
            continue
        label = scope_map.get(hlo_op)
        if label is None:
            continue
        dur_seconds = float(event.get("dur", 0.0)) / 1e6
        bucket = totals.setdefault(label, [0.0, 0])
        bucket[0] += dur_seconds
        bucket[1] += 1
    return {label: (seconds, count) for label, (seconds, count) in totals.items()}


def parse_hlo_op_times(
    trace_events: list[dict[str, Any]],
) -> dict[str, tuple[float, int]]:
    """Attribute executed wall-clock time to every ``hlo_op``, not known labels.

    Same event filter as ``parse_scopes`` (complete ``X`` events, skip
    ``end: `` bookkeeping, require ``args["hlo_op"]``). Use this when
    named_scope recovery is a few percent of host ``total_step_seconds``:
    on GPU the missing mass is often one ``command_buffer_*`` thunk
    (CUDA graph). Returns ``{hlo_op: (exclusive_seconds, n_occurrences)}``.
    """
    totals: dict[str, list[float]] = {}
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if isinstance(name, str) and name.startswith("end: "):
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        hlo_op = args.get("hlo_op")
        if hlo_op is None:
            continue
        dur_seconds = float(event.get("dur", 0.0)) / 1e6
        bucket = totals.setdefault(str(hlo_op), [0.0, 0])
        bucket[0] += dur_seconds
        bucket[1] += 1
    return {op: (seconds, count) for op, (seconds, count) in totals.items()}


def parse_dispatch_counts(
    trace_events: list[dict[str, Any]], fn_name: str | None = None
) -> dict[str, int]:
    """Derive the three dispatch-count metrics from a Perfetto trace capture.

    n_host_syncs is NOT included -- removed from the DISPATCH_COUNT contract
    (see claims.py's REQUIRED_METRICS comment) after this module's own
    feasibility spike found no trace event on this CPU install distinguishes
    a device->host transfer from execution completion (CPU device buffers
    ARE host memory).

    - n_executions: count of "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice"
      events -- XLA executable invocations over the timed region. (NOT
      "ExecuteReplicated.__call__": confirmed by direct experiment (5 calls
      to the same jitted function -> ExecuteReplicated.__call__ stayed at 1
      while ExecuteHelperOnSingleDevice tracked all 5) that the former does
      not scale 1:1 with actual dispatch count on this CPU backend.)
    - n_compilations: count of "backend_compile_and_load" events -- distinct
      XLA compilations triggered.
    - n_jit_traces: count of "PjitFunction(<fn_name>)" events -- Python-level
      jit trace events. If ``fn_name`` is given, only that function's
      traces are counted (excluding internal helper jits like
      convert_element_type/_reduce_sum); otherwise all PjitFunction(...)
      events are counted. Confirmed 2x per actual Python-level call on this
      JAX install (enter+exit bookkeeping) -- callers comparing across
      configurations should treat this as a relative, not absolute, count.
    """
    n_executions = 0
    n_compilations = 0
    n_jit_traces = 0
    target_jit_name = f"PjitFunction({fn_name})" if fn_name else None
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if name == "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice":
            n_executions += 1
        elif name == "backend_compile_and_load":
            n_compilations += 1
        elif isinstance(name, str) and name.startswith("PjitFunction("):
            if target_jit_name is None or name == target_jit_name:
                n_jit_traces += 1
    return {
        "n_executions": n_executions,
        "n_compilations": n_compilations,
        "n_jit_traces": n_jit_traces,
    }
