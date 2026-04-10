"""Create a 3x2 panel showing meshes across different dimensional configurations."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.curves import helix_3d
from physicsnemo.mesh.primitives.planar import annulus_2d
from physicsnemo.mesh.primitives.surfaces import torus
from physicsnemo.mesh.primitives.volumes import cube_volume

OUTPUT = Path(__file__).parent / "dimensional_gallery.png"

AIRFRANS_DIR = (
    Path.home()
    / "gh/aerodynamics_datasets/airfrans/Dataset"
    / "airFoil2D_SST_31.468_13.713_3.339_3.51_6.993"
)

fig = plt.figure(figsize=(15, 19))

### Panel 1: 0D manifold in 3D (point cloud on a sphere)
ax1 = fig.add_subplot(3, 2, 1, projection="3d")
rng = np.random.default_rng(42)
n_cloud = 2000
gaussian_pts = rng.standard_normal((n_cloud, 3))
sphere_pts = gaussian_pts / np.linalg.norm(gaussian_pts, axis=1, keepdims=True)
ax1.scatter(
    sphere_pts[:, 0], sphere_pts[:, 1], sphere_pts[:, 2],
    s=2, c=sphere_pts[:, 2], cmap="viridis", alpha=0.7,
)
ax1.set_title("0D manifold in 3D\n(point cloud; no connectivity)", fontsize=12, fontweight="bold")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

### Panel 2: 1D manifold in 3D (helix, straight segments only)
ax2 = fig.add_subplot(3, 2, 2, projection="3d")
mesh_1d = helix_3d.load(n_points=400, n_turns=3)
pts_1d = mesh_1d.points.numpy()
cells_1d = mesh_1d.cells.numpy()
for c in cells_1d:
    p0, p1 = pts_1d[c[0]], pts_1d[c[1]]
    ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="#2196F3", linewidth=1.2)
ax2.scatter(pts_1d[:, 0], pts_1d[:, 1], pts_1d[:, 2], s=3, color="#1565C0", zorder=5)
ax2.set_title("1D manifold in 3D\n(edges in 3-space)", fontsize=12, fontweight="bold")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

### Panel 3: 2D manifold in 2D (annulus with radial coloring)
ax3 = fig.add_subplot(3, 2, 3)
mesh_2d = annulus_2d.load(inner_radius=0.4, outer_radius=1.0, n_radial=8, n_angular=40)
pts2 = mesh_2d.points.numpy()
cells2 = mesh_2d.cells.numpy()
cell_centroids = pts2[cells2].mean(axis=1)
radial_dist = np.sqrt(cell_centroids[:, 0] ** 2 + cell_centroids[:, 1] ** 2)
triang = mtri.Triangulation(pts2[:, 0], pts2[:, 1], cells2)
ax3.tripcolor(triang, radial_dist, cmap="viridis", edgecolors="black", linewidth=0.3)
ax3.set_title("2D manifold in 2D\n(triangles in 2-space)", fontsize=12, fontweight="bold")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_aspect("equal")

### Panel 4: 2D manifold in 3D (torus)
ax4 = fig.add_subplot(3, 2, 4, projection="3d")
mesh_surf = torus.load(major_radius=1.0, minor_radius=0.35, n_major=40, n_minor=20)
pts3 = mesh_surf.points.numpy()
cells3 = mesh_surf.cells.numpy()
z_vals = pts3[cells3].mean(axis=1)[:, 2]
norm_t = Normalize(vmin=z_vals.min(), vmax=z_vals.max())
sm_t = ScalarMappable(cmap="plasma", norm=norm_t)
face_colors_t = sm_t.to_rgba(z_vals)
ax4.plot_trisurf(
    pts3[:, 0], pts3[:, 1], pts3[:, 2],
    triangles=cells3, edgecolor="k", linewidth=0.1, alpha=0.9,
)
ax4.collections[0].set_facecolors(face_colors_t)
ax4.set_title("2D manifold in 3D\n(triangles in 3-space)", fontsize=12, fontweight="bold")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("z")

### Panel 5: 3D manifold in 3D (cube volume, boundary shown)
ax5 = fig.add_subplot(3, 2, 5, projection="3d")
mesh_vol = cube_volume.load(subdivisions=3)
boundary = mesh_vol.get_boundary_mesh()
bpts = boundary.points.numpy()
bcells = boundary.cells.numpy()
z_face = bpts[bcells].mean(axis=1)[:, 2]
norm5 = Normalize(vmin=z_face.min(), vmax=z_face.max())
sm5 = ScalarMappable(cmap="coolwarm", norm=norm5)
face_colors5 = sm5.to_rgba(z_face)
ax5.plot_trisurf(
    bpts[:, 0], bpts[:, 1], bpts[:, 2],
    triangles=bcells, edgecolor="k", linewidth=0.15, alpha=0.7,
)
ax5.collections[0].set_facecolors(face_colors5)
ax5.set_title(
    f"3D manifold in 3D\n(tetrahedra in 3-space; {mesh_vol.n_cells:,} tets)",
    fontsize=12, fontweight="bold",
)
ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.set_zlabel("z")

### Panel 6: Interior mesh + its boundary (AirFRANS airfoil)
ax6 = fig.add_subplot(3, 2, 6)

prefix = "airFoil2D_SST_31.468_13.713_3.339_3.51_6.993"
pv_internal = pv.read(AIRFRANS_DIR / f"{prefix}_internal.vtu")
pv_aerofoil = pv.read(AIRFRANS_DIR / f"{prefix}_aerofoil.vtp")

### Triangulate and clip to a region near the airfoil
pv_tri = pv_internal.triangulate()
pv_clipped = pv_tri.clip_box([-0.3, 1.5, -0.5, 0.5, 0, 1], invert=False)

### Project to 2D (drop z coordinate, which is constant at 0.5)
int_pts = np.column_stack([pv_clipped.points[:, 0], pv_clipped.points[:, 1]])
int_cells = pv_clipped.cells.reshape(-1, 4)[:, 1:]  # strip VTK cell-size prefix

### Draw interior mesh
triang_int = mtri.Triangulation(int_pts[:, 0], int_pts[:, 1], int_cells)
ax6.triplot(triang_int, color="#CCCCCC", linewidth=0.15)

### Draw boundary (aerofoil) on top - outline only, no fill
bnd_pts = pv_aerofoil.points[:, :2]
bnd_order = np.argsort(np.arctan2(
    bnd_pts[:, 1] - bnd_pts[:, 1].mean(),
    bnd_pts[:, 0] - bnd_pts[:, 0].mean(),
))
bnd_sorted = bnd_pts[bnd_order]
ax6.plot(
    np.append(bnd_sorted[:, 0], bnd_sorted[0, 0]),
    np.append(bnd_sorted[:, 1], bnd_sorted[0, 1]),
    color="#D84315", linewidth=2.5, zorder=4, label="Boundary (1D mesh)",
)

ax6.set_title(
    "Interior mesh + its boundary\n(both are Mesh objects)",
    fontsize=12, fontweight="bold",
)
ax6.set_xlabel("x")
ax6.set_ylabel("y")
ax6.set_aspect("equal")
ax6.legend(loc="upper right", fontsize=9)

fig.suptitle(
    "PhysicsNeMo-Mesh: One API for Any Dimension",
    fontsize=16, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT}")
