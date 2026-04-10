"""Render DrivAerML car with pressure coefficient (Cp) coloring."""

from pathlib import Path

import pyvista as pv

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "hero_drivaerml_cp.png"
VTP_PATH = Path.home() / "gh/aerodynamics_datasets/drivaerml/drivaer_data/run_1/boundary_1.vtp"

mesh = pv.read(VTP_PATH)

plotter = pv.Plotter(window_size=(1920, 1080))
plotter.add_mesh(
    mesh,
    scalars="CpMeanTrim",
    cmap="RdBu_r",
    clim=(-1.5, 1.0),
    show_edges=False,
    scalar_bar_args={"title": "Pressure Coefficient (Cp)", "color": "black"},
)
plotter.set_background("white")

### Front-driver-side view, centered on mesh, auto-zoomed
center = mesh.center
eye = [center[0] - 5, center[1] - 4, center[2] + 2.5]
plotter.camera_position = [eye, center, (0, 0, 1)]
plotter.reset_camera()

plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
