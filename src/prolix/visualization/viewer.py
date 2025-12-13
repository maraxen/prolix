"""Interactive viewer integration."""
from __future__ import annotations

import logging
from typing import Optional, Union, Any
import numpy as np

# Try importing py2Dmol
try:
    import py2Dmol
except ImportError:
    py2Dmol = None

from .trajectory import TrajectoryReader

logger = logging.getLogger(__name__)

def _require_py2dmol():
    if py2Dmol is None:
        raise ImportError("py2Dmol is required for interactive visualization. Please install it with `pip install py2Dmol`.")

def view_structure(pdb_path: str, style: str = "cartoon") -> Any:
    """View a single structure using py2Dmol.
    
    Args:
        pdb_path: Path to PDB file
        style: Visualization style ('cartoon', 'stick', 'line', 'sphere')
    """
    _require_py2dmol()
    
    with open(pdb_path, 'r') as f:
        pdb_data = f.read()
        
    view = py2Dmol.view()
    view.addModel(pdb_data, "pdb")
    view.setStyle({style: {'color': 'spectrum'}})
    view.zoomTo()
    return view

def view_trajectory(
    trajectory: Union[TrajectoryReader, str], 
    pdb_path: str,
    stride: int = 1,
    style: str = "cartoon"
) -> Any:
    """View trajectory using py2Dmol.
    
    Args:
        trajectory: TrajectoryReader instance or path
        pdb_path: Topology PDB file
        stride: Frame stride
    """
    _require_py2dmol()
    
    if isinstance(trajectory, str):
        trajectory = TrajectoryReader(trajectory)
        
    # We need to construct a multi-model PDB or similar format.
    # Reading the PDB template
    with open(pdb_path, 'r') as f:
        pdb_lines = f.readlines()
        
    # Extract ATOM/HETATM lines
    atom_lines = [l for l in pdb_lines if l.startswith("ATOM") or l.startswith("HETATM")]
    
    # Get positions
    positions = trajectory.get_positions()[::stride] # (T, N, 3)
    n_frames = len(positions)
    
    # Check atom count match
    if len(atom_lines) != positions.shape[1]:
        logger.warning(f"PDB atom count ({len(atom_lines)}) does not match trajectory ({positions.shape[1]}). Visualization may be corrupted.")
        
    # Construct multi-model PDB string
    # This is inefficient for large trajectories, but works for demos.
    out_lines = []
    
    for i in range(n_frames):
        out_lines.append(f"MODEL     {i+1}\n")
        frame_pos = positions[i]
        
        for j, line in enumerate(atom_lines):
            # Replace coordinates
            # PDB format: x[30:38], y[38:46], z[46:54]
            # Formatted %8.3f
            if j < len(frame_pos):
                x, y, z = frame_pos[j]
                new_line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                out_lines.append(new_line)
            else:
                out_lines.append(line)
                
        out_lines.append("ENDMDL\n")
        
    pdb_data = "".join(out_lines)
    
    view = py2Dmol.view()
    view.addModelsAsFrames(pdb_data, "pdb")
    view.setStyle({style: {'color': 'spectrum'}})
    view.animate({'loop': "forward"})
    view.zoomTo()
    
    return view

def save_trajectory_html(
    trajectory: Union[TrajectoryReader, str],
    pdb_path: str,
    output_path: str,
    stride: int = 1,
    style: str = "cartoon",
    title: str = "Trajectory Visualization",
    custom_styles: Optional[list[tuple[dict, dict]]] = None
):
    """Save trajectory visualization to a standalone HTML file.
    
    This method does NOT require py2Dmol to be installed, as it generates
    standard HTML using the 3Dmol.js CDN.
    
    Args:
        trajectory: TrajectoryReader instance or path
        pdb_path: Topology PDB file
        output_path: Output HTML file path
        stride: Frame stride
        style: Default visualization style (applied to everything not matched by custom_styles)
        title: Page title
        custom_styles: List of (selection, style) tuples. 
                       Example: [({'resn': 'WAT'}, {'sphere': {'radius': 0.5}})]
    """
    if isinstance(trajectory, str):
        trajectory = TrajectoryReader(trajectory)
        
    # Read PDB template
    with open(pdb_path, 'r') as f:
        pdb_lines = f.readlines()
        
    atom_lines = [l for l in pdb_lines if l.startswith("ATOM") or l.startswith("HETATM")]
    
    # Get positions
    positions = trajectory.get_positions()[::stride]
    n_frames = len(positions)
    
    # Construct multi-model PDB string
    out_lines = []
    
    for i in range(n_frames):
        out_lines.append(f"MODEL     {i+1}\\n")
        frame_pos = positions[i]
        
        for j, line in enumerate(atom_lines):
            if j < len(frame_pos):
                x, y, z = frame_pos[j]
                # PDB format: %8.3f at cols 30, 38, 46
                coords = f"{x:8.3f}{y:8.3f}{z:8.3f}"
                new_line = line[:30] + coords + line[54:].rstrip()
                out_lines.append(new_line + "\\n")
            else:
                out_lines.append(line.rstrip() + "\\n")
        out_lines.append("ENDMDL\\n")
        
    pdb_data = "".join(out_lines)
    
    # Build JS for styles
    style_js = ""
    
    # If no custom styles, just apply default style to everything
    if not custom_styles:
        style_js = f"viewer.setStyle({{ {style}: {{ color: 'spectrum' }} }});"
    else:
        # Apply default style first to everything (or exclude things?)
        # Better approach: Apply default style to "all", then override?
        # Or apply default style to "not water"?
        # Let's assume 'style' argument is the "base" style.
        style_js += f"viewer.setStyle({{ {style}: {{ color: 'spectrum' }} }});\n"
        
        import json
        for sel, sty in custom_styles:
            sel_json = json.dumps(sel)
            sty_json = json.dumps(sty)
            style_js += f"            viewer.addStyle({sel_json}, {sty_json});\n"

    # HTML Template
    # We use backticks for template string in JS to handle newlines in pdb_data safely
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; overflow: hidden; font-family: sans-serif; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #info {{ position: absolute; top: 10px; left: 10px; z-index: 10; background: rgba(255,255,255,0.8); padding: 5px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>{title}</h3>
        <p>Frames: {n_frames} | Stride: {stride}</p>
    </div>
    <div id="container"></div>

    <script>
        $(function() {{
            let element = $("#container");
            let config = {{ backgroundColor: "white" }};
            let viewer = $3Dmol.createViewer(element, config);
            
            let pdbData = `{pdb_data}`;
            
            viewer.addModelsAsFrames(pdbData, "pdb");
            
            // Set Styles
{style_js}
            
            viewer.zoomTo();
            viewer.animate({{ loop: "forward", reps: 0 }});
            viewer.render();
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html_content)
        
    logger.info(f"Saved visualization to {output_path}")
