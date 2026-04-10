"""Render the DrivAerML car as a surface mesh, its edge graph, and its cell-adjacency graph.

Loads the DrivAerML STL, remeshes to ~5k clusters, then shows three views:
1. Surface mesh (2D manifold in 3D)
2. Edge graph via facet extraction (1D manifold - best for point-centered data)
3. Cell-adjacency graph via centroid connectivity (1D manifold - best for cell-centered data)
"""

from pathlib import Path

import pyvista as pv
import torch

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.io import from_pyvista, to_pyvista
from physicsnemo.mesh.remeshing import remesh

pv.OFF_SCREEN = True

OUTPUT = Path(__file__).parent / "car_as_graph.png"
STL_PATH = Path.home() / "gh/aerodynamics_datasets/drivaerml/drivaer_data/run_1/drivaer_1.stl"

### Load and remesh the car
pv_car = pv.read(STL_PATH)
car_mesh = from_pyvista(pv_car)
car_remeshed = remesh(car_mesh, n_clusters=5000)

### Edge graph: extract facets (edges) - a 2D surface's facets are its 1D edges
edge_mesh = car_remeshed.get_facet_mesh()

### Cell-adjacency graph: connect centroids of adjacent cells
adj = car_remeshed.get_cell_to_cells_adjacency()
sources, targets = adj.expand_to_pairs()
keep = sources < targets
edges = torch.stack([sources[keep], targets[keep]], dim=1)
centroid_graph = Mesh(
    points=car_remeshed.cell_centroids,
    cells=edges,
    point_data=car_remeshed.cell_data,
)

### Convert to PyVista for rendering
pv_surface = to_pyvista(car_remeshed)
pv_edges = to_pyvista(edge_mesh)
pv_centroid = to_pyvista(centroid_graph)

### Front-driver-side camera, computed from mesh center
center = list(pv_surface.center)
CAMERA = [[center[0] - 5, center[1] - 4, center[2] + 2.5], center, (0, 0, 1)]

plotter = pv.Plotter(shape=(1, 3), window_size=(2400, 800))

### Panel 1: surface mesh (2D manifold in 3D)
plotter.subplot(0, 0)
plotter.add_mesh(pv_surface, color="steelblue", show_edges=True, edge_color="black", line_width=0.5)
plotter.add_text(
    f"Surface mesh (2D manifold)\n{car_remeshed.n_points:,} points, {car_remeshed.n_cells:,} triangles",
    font_size=9, color="black",
)
plotter.set_background("white")
plotter.camera_position = CAMERA
plotter.reset_camera()

### Panel 2: edge graph (1D manifold - point-centered)
plotter.subplot(0, 1)
plotter.add_mesh(pv_edges, color="darkorange", line_width=2)
plotter.add_mesh(
    pv.PolyData(pv_edges.points),
    color="darkorange", point_size=4, render_points_as_spheres=True,
)
plotter.add_text(
    f"Edge graph (point-centered)\n{edge_mesh.n_points:,} nodes, {edge_mesh.n_cells:,} edges",
    font_size=9, color="black",
)
plotter.set_background("white")
plotter.camera_position = CAMERA
plotter.reset_camera()

### Panel 3: cell-adjacency graph (1D manifold - cell-centered)
plotter.subplot(0, 2)
plotter.add_mesh(pv_centroid, color="#4CAF50", line_width=1.5)
plotter.add_mesh(
    pv.PolyData(pv_centroid.points),
    color="#2E7D32", point_size=3, render_points_as_spheres=True,
)
plotter.add_text(
    f"Cell-adjacency graph (cell-centered)\n{centroid_graph.n_points:,} nodes, {centroid_graph.n_cells:,} edges",
    font_size=9, color="black",
)
plotter.set_background("white")
plotter.camera_position = CAMERA
plotter.reset_camera()

plotter.screenshot(OUTPUT, transparent_background=False)
plotter.close()

print(f"Saved {OUTPUT}")
