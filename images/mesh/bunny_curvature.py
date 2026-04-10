"""Render Stanford bunny colored by mean curvature."""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista, to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "bunny_mean_curvature.png"

### Load the bunny from PyVista examples and convert
pv_bunny = pv.examples.download_bunny()
mesh = from_pyvista(pv_bunny)

### The raw PyVista bunny has duplicate vertices that cause NaN in
### curvature computations. Cleaning removes them.
mesh = mesh.clean()

### Subdivide for smoother curvature estimation
mesh = mesh.subdivide(levels=1, filter="loop")

### Compute mean curvature with log1p regularization for visualization
H = mesh.mean_curvature_vertices
H = torch.nan_to_num(H, nan=0.0)
H_reg = H.sign() * H.abs().log1p()
mesh.point_data["mean_curvature"] = H_reg

H_np = H_reg.numpy()
low, high = np.percentile(H_np, 3), np.percentile(H_np, 97)

pv_mesh = to_pyvista(mesh)

plotter = pv.Plotter(window_size=(1400, 1000))
plotter.add_mesh(
    pv_mesh,
    scalars="mean_curvature",
    cmap="PuOr_r",
    clim=(low, high),
    show_edges=False,
    scalar_bar_args={"title": "Mean Curvature", "color": "black"},
)
plotter.set_background("white")
### Center on the mesh and view from front-left, above ground plane
center = mesh.points.mean(dim=0).numpy().tolist()
eye = [center[0] - 0.30, center[1] + 0.15, center[2] - 0.18]
plotter.camera_position = [eye, center, (0, 1, 0)]
plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
