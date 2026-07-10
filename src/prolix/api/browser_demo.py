"""Static browser smoke demo builder for Claim 2 W4 (#278).

Builds a self-contained ``index.html`` plus ``trace.json`` from a JIT smoke
diagnostics export. Optional IREE-WASM artifact when ``iree-compile`` is present.

``dt`` defaults to **0.5 fs** (XR-VACUUM-DT); see ``make_smoke_diagnostics_fn``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import jax
import jax.numpy as jnp

from prolix.api.export_run import make_single_trajectory_fn, make_smoke_diagnostics_fn

_BROWSER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prolix Claim 2 — Browser Smoke (W4)</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --temp: #f78166;
      --energy: #7ee787;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      padding: 1.5rem 2rem;
      border-bottom: 1px solid #30363d;
      background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    }
    h1 { margin: 0 0 0.25rem; font-size: 1.35rem; font-weight: 600; }
    .subtitle { color: var(--muted); font-size: 0.9rem; }
    main { padding: 1.5rem 2rem 2.5rem; max-width: 960px; }
    .grid { display: grid; gap: 1.25rem; }
    .panel {
      background: var(--panel);
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 1rem 1.25rem 1.25rem;
    }
    .panel h2 {
      margin: 0 0 0.75rem;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    canvas {
      width: 100%;
      height: 220px;
      display: block;
      border-radius: 4px;
      background: #0d1117;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem 2rem;
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 1rem;
    }
    .meta strong { color: var(--text); font-weight: 500; }
    footer {
      padding: 1rem 2rem 2rem;
      color: var(--muted);
      font-size: 0.8rem;
    }
  </style>
</head>
<body>
  <header>
    <h1>Prolix browser smoke — solvated explicit-solvent MD</h1>
    <p class="subtitle">Claim 2 W4: energy and temperature traces from EnsemblePlan export path</p>
  </header>
  <main>
    <div class="meta" id="meta"></div>
    <div class="grid">
      <section class="panel">
        <h2 style="color: var(--temp)">Kinetic temperature (K)</h2>
        <canvas id="temp-chart" width="880" height="220"></canvas>
      </section>
      <section class="panel">
        <h2 style="color: var(--energy)">Kinetic energy (kcal/mol)</h2>
        <canvas id="energy-chart" width="880" height="220"></canvas>
      </section>
    </div>
  </main>
  <footer>
    Static demo — no server compute. Trace generated via <code>make_smoke_diagnostics_fn</code>;
    WASM artifact bundled when IREE compile is available (W3 path).
  </footer>
  <script id="trace-data" type="application/json">__TRACE_JSON__</script>
  <script>
    const trace = JSON.parse(document.getElementById("trace-data").textContent);

    const meta = document.getElementById("meta");
    meta.innerHTML = [
      ["System", trace.meta.system],
      ["Steps", trace.meta.n_steps],
      ["Atoms", trace.meta.n_atoms],
      ["dt (fs)", trace.meta.dt_fs],
      ["Target T (K)", trace.meta.target_temperature_k],
      ["WASM", trace.meta.wasm_bytes ? trace.meta.wasm_bytes + " bytes" : "not bundled"],
    ].map(([k, v]) => `<span><strong>${k}:</strong> ${v}</span>`).join("");

    function drawSeries(canvasId, series, color, yLabel) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      const pad = { l: 48, r: 16, t: 12, b: 32 };
      const plotW = w - pad.l - pad.r;
      const plotH = h - pad.t - pad.b;
      ctx.clearRect(0, 0, w, h);

      const minY = Math.min(...series);
      const maxY = Math.max(...series);
      const span = maxY - minY || 1;
      const n = series.length;

      ctx.strokeStyle = "#30363d";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.t + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(w - pad.r, y);
        ctx.stroke();
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.forEach((v, i) => {
        const x = pad.l + (plotW * i) / Math.max(n - 1, 1);
        const y = pad.t + plotH - ((v - minY) / span) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = "#8b949e";
      ctx.font = "11px system-ui";
      ctx.fillText(yLabel, 8, pad.t + 12);
      ctx.fillText(minY.toFixed(2), 4, h - pad.b);
      ctx.fillText(maxY.toFixed(2), 4, pad.t + 10);
      ctx.fillText("0", pad.l, h - 8);
      ctx.fillText(String(n - 1), w - pad.r - 16, h - 8);
    }

    drawSeries("temp-chart", trace.temperatures, "#f78166", "T (K)");
    drawSeries("energy-chart", trace.energies, "#7ee787", "E (kcal/mol)");
  </script>
</body>
</html>
"""


def build_browser_smoke_demo(
    output_dir: str | pathlib.Path,
    bundle: Any,
    *,
    n_steps: int = 100,
    dt: float = 0.5,
    kT: float = 0.596,
    seed: int = 278,
    compile_wasm: bool = True,
    system_label: str = "B=1 smoke export (Claim 2 EnsemblePlan path)",
) -> dict[str, Any]:
    """Write static W4 demo artifacts and return trace metadata for tests."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed_arr = jnp.array(seed, dtype=jnp.uint32)
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    kT_arr = jnp.asarray(kT, dtype=jnp.float32)

    diag_fn = jax.jit(make_smoke_diagnostics_fn(bundle, n_steps=n_steps))
    positions, temperatures, energies = diag_fn(seed_arr, dt_arr, kT_arr)

    n_atoms = int(jnp.asarray(bundle.n_atoms))
    temps_list = [float(x) for x in temperatures]
    energies_list = [float(x) for x in energies]

    wasm_path = out / "trajectory.wasm"
    wasm_bytes: int | None = None
    if compile_wasm:
        from prolix import export

        if export.find_iree_compile() is not None:
            traj_fn = make_single_trajectory_fn(bundle, n_steps=n_steps)
            lowered = jax.jit(traj_fn).lower(seed_arr, dt_arr, kT_arr)
            export.compile_lowered_to_wasm(lowered, wasm_path)
            wasm_bytes = export.assert_wasm_artifact_under_limit(wasm_path)

    from prolix.simulate import BOLTZMANN_KCAL

    target_t_k = float(kT / BOLTZMANN_KCAL)
    trace_payload = {
        "meta": {
            "n_steps": n_steps,
            "n_atoms": n_atoms,
            "dt_fs": dt,
            "kT_kcal_mol": float(kT),
            "target_temperature_k": round(target_t_k, 1),
            "seed": seed,
            "system": system_label,
            "wasm_bytes": wasm_bytes,
        },
        "temperatures": temps_list,
        "energies": energies_list,
    }
    trace_path = out / "trace.json"
    trace_path.write_text(json.dumps(trace_payload, indent=2))

    html = _BROWSER_HTML.replace(
        "__TRACE_JSON__",
        json.dumps(trace_payload).replace("</", "<\\/"),
    )
    html_path = out / "index.html"
    html_path.write_text(html)

    return {
        "n_steps": n_steps,
        "n_atoms": n_atoms,
        "temperatures": temps_list,
        "energies": energies_list,
        "positions_shape": tuple(int(x) for x in positions.shape),
        "html_path": html_path,
        "trace_path": trace_path,
        "wasm_path": wasm_path if wasm_bytes is not None else None,
        "wasm_bytes": wasm_bytes,
    }
