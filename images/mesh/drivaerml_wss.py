"""Render DrivAerML car with wall shear stress magnitude coloring."""

from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "drivaerml_wss.png"
VTP_PATH = Path.home() / "gh/aerodynamics_datasets/drivaerml/drivaer_data/run_1/boundary_1.vtp"

mesh = pv.read(VTP_PATH)

### Compute WSS magnitude from the vector field
wss = mesh.cell_data["wallShearStressMeanTrim"]
wss_mag = np.linalg.norm(wss, axis=-1)
mesh.cell_data["WSS Magnitude"] = wss_mag

plotter = pv.Plotter(window_size=(1920, 1080))
plotter.add_mesh(
    mesh,
    scalars="WSS Magnitude",
    cmap="magma",
    clim=(0, np.percentile(wss_mag, 97)),
    show_edges=False,
    scalar_bar_args={"title": "Wall Shear Stress Magnitude (Pa)", "color": "black"},
)
plotter.set_background("white")
plotter.camera_position = [(7, -5, 3), (1.5, 0, 0.4), (0, 0, 1)]
plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
