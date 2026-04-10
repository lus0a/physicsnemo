"""Compute and visualize the streamwise pressure gradient on DrivAerML car.

Projects the surface pressure gradient onto the local flow direction
(wall shear stress unit vector) to obtain dp/ds. Positive values indicate
adverse pressure gradients (destabilizing), negative values indicate
favorable (stabilizing).
"""

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from physicsnemo.mesh.io import from_pyvista, to_pyvista

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "drivaerml_adverse_pg.png"
VTP_PATH = Path.home() / "gh/aerodynamics_datasets/drivaerml/drivaer_data/run_1/boundary_1.vtp"

### Load and convert to PhysicsNeMo Mesh
mesh = from_pyvista(pv.read(VTP_PATH))

### Convert cell data to point data for gradient computation
mesh = mesh.cell_data_to_point_data()

### Compute the surface pressure gradient (extrinsic, in ambient 3D coordinates)
mesh = mesh.compute_point_derivatives(keys="pMeanTrim", gradient_type="extrinsic")

### Project pressure gradient onto the local flow direction (WSS unit vector)
grad_p = mesh.point_data["pMeanTrim_gradient"]
wss = mesh.point_data["wallShearStressMeanTrim"]
wss_hat = wss / wss.norm(dim=-1, keepdim=True).clamp(min=1e-8)
alignment = (grad_p * wss_hat).sum(dim=-1)
alignment = torch.nan_to_num(alignment, nan=0.0)

### Apply arcsinh to compress extreme values while preserving sign
alignment = torch.arcsinh(alignment)
mesh.point_data["adverse_pg"] = alignment

### Render
pv_mesh = to_pyvista(mesh)

alignment_np = alignment.numpy()
lim = np.percentile(np.abs(alignment_np[alignment_np != 0]), 95)

plotter = pv.Plotter(window_size=(1920, 1080))
plotter.add_mesh(
    pv_mesh,
    scalars="adverse_pg",
    cmap="RdBu_r",
    clim=(-lim, lim),
    show_edges=False,
    scalar_bar_args={
        "title": "dp/ds  (adverse ← → favorable)",
        "color": "black",
    },
)
plotter.set_background("white")

center = pv_mesh.center
eye = [center[0] - 5, center[1] - 4, center[2] + 2.5]
plotter.camera_position = [eye, center, (0, 0, 1)]
plotter.reset_camera()

plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
