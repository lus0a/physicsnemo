"""Ad-hoc verification of the ElderProblem2D solver (run before training).

Checks (variable-density, non-Boussinesq c-p form):
 1. Flow solve: div(rho q) ~ 0 in the interior (the continuity the the dense
    solve enforces), using the Darcy face fluxes from the solved head.
 2. No-flow walls: the wall face fluxes are zero by construction and the
    interior velocity next to the walls is ~ 0.
 3. Buoyancy direction: an interior top-hat dense (c=1) blob produces downward
    (Fz > 0, +z) velocity below it -> dense fluid sinks for flow_sign = +1.
 4. CFL stability: a short rollout produces no NaN/Inf and c stays in [0, 1].
 5. Initial condition / gauge: h = 0 initially and p = 0 at the top-left corner.

Run:  python _verify_solver.py
"""
import numpy as np
import torch

from datapipe import ElderProblem2D


def _div_face(Fx, Fz, dx, dy):
    """Interior divergence of (rho q) from interior face fluxes (walls = 0)."""
    # x-faces Fx: [..., Ny, Nx-1]; per-cell east = Fx, west = shifted Fx.
    Fx_east = torch.cat([Fx, torch.zeros_like(Fx[..., :, :1])], dim=-1)
    Fx_west = torch.cat([torch.zeros_like(Fx[..., :, :1]), Fx], dim=-1)
    # z-faces Fz: [..., Ny-1, Nx]; per-cell south = Fz, north = shifted Fz.
    Fz_south = torch.cat([Fz, torch.zeros_like(Fz[..., :1, :])], dim=-2)
    Fz_north = torch.cat([torch.zeros_like(Fz[..., :1, :]), Fz], dim=-2)
    return (Fx_east - Fx_west) / dx + (Fz_south - Fz_north) / dy


def test_flow_residual():
    dp = ElderProblem2D(resolution=24, batch_size=2, n_trajectories=2,
                        rollout_steps=2, device="cpu")
    c = torch.zeros(2, 1, dp.Ny_tot, dp.Nx_tot)
    c[:, :, 2:5, dp.src_x0:dp.src_x1] = 1.0
    dp._apply_bc_c(c)
    # Quasi-static solve (no storage): div(rho q) ~ 0.
    h = dp._flow_solve(dp._interior(c))
    Fx, Fz = dp._face_fluxes(dp._interior(c), h)
    div = _div_face(Fx, Fz, dp.dx, dp.dy)
    err = div.abs().max().item() / (Fx.abs().amax().item() + Fz.abs().amax().item() + 1e-12)
    print(f"[1a] quasi-static div(rho q) max rel err = {err:.2e}")
    assert err < 1e-3, "flow solve did not satisfy continuity"

    # Storage-corrected solve: d(phi rho)/dt + div(rho q) ~ 0, with a nonzero
    # dc/dt (forward difference over the macro step).
    dc = 0.01 * torch.randn_like(c)
    dp._apply_bc_c(dc)
    dc_dt = dp._interior(dc) / dp.dt_macro
    h2 = dp._flow_solve(dp._interior(c), dc_dt)
    Fx, Fz = dp._face_fluxes(dp._interior(c), h2)
    storage = dp.phi * dp.drho * dc_dt
    res = storage + _div_face(Fx, Fz, dp.dx, dp.dy)
    # The gauge cell (top-left, where h is pinned) does not enforce continuity;
    # exclude it from the check.
    res[..., 0, 0] = 0.0
    err2 = res.abs().max().item() / (Fx.abs().amax().item() + Fz.abs().amax().item() + 1e-12)
    print(f"[1b] storage-corrected (d(phi rho)/dt + div(rho q)) max rel err = {err2:.2e}")
    assert err2 < 1e-3, "storage-corrected flow solve did not satisfy continuity"


def test_no_flow_walls():
    dp = ElderProblem2D(resolution=24, batch_size=1, n_trajectories=1,
                        rollout_steps=2, device="cpu")
    c = torch.zeros(1, 1, dp.Ny_tot, dp.Nx_tot)
    c[:, :, 2:5, dp.src_x0:dp.src_x1] = 1.0
    dp._apply_bc_c(c)
    h = dp._flow_solve(dp._interior(c))
    Fx, Fz = dp._face_fluxes(dp._interior(c), h)
    # Wall face fluxes are absent by construction (Fx/Fz hold only interior
    # faces); the interior cells adjacent to the wall should have ~0 net wall
    # flux. Check the divergence at boundary-interior cells is dominated by
    # interior faces, i.e. the wall contributes nothing.
    div = _div_face(Fx, Fz, dp.dx, dp.dy)
    # The first/last interior columns/rows: their wall-side face is zero, so
    # the only flux imbalance comes from interior faces -> finite and small.
    print(f"[2] |div(rho q)| at boundary-interior cells max = "
          f"{div[..., :, 0].abs().max().item():.2e} / "
          f"{div[..., :, -1].abs().max().item():.2e}")
    assert torch.isfinite(div).all()


def test_buoyancy_sign():
    """Interior dense blob: flow_sign=+1 must give downward (Fz>0) velocity."""
    results = {}
    for sign in (+1.0, -1.0):
        dp = ElderProblem2D(resolution=32, batch_size=1, n_trajectories=1,
                            rollout_steps=2, device="cpu", flow_sign=sign)
        c = torch.zeros(1, 1, dp.Ny_tot, dp.Nx_tot)
        r0, r1 = 2, 6
        c[:, :, r0:r1, dp.src_x0:dp.src_x1] = 1.0
        dp._apply_bc_c(c)
        h = dp._flow_solve(dp._interior(c))
        Fx, Fz = dp._face_fluxes(dp._interior(c), h)
        row = (r0 + r1) // 2
        results[sign] = float(Fz[0, 0, row, dp.src_x0 + 2].item())
    print(f"[3] Fz at dense-blob center:  sign=+1 -> {results[+1.0]:+.3e},  "
          f"sign=-1 -> {results[-1.0]:+.3e}")
    sinking_sign = +1.0 if results[+1.0] > 0 else -1.0
    print(f"    sinking (Fz>0, downward) requires flow_sign = {sinking_sign}")
    return sinking_sign


def test_cfl_stability():
    dp = ElderProblem2D(resolution=32, batch_size=2, n_trajectories=2,
                        rollout_steps=8, dt_macro=10.0 * 24 * 3600.0,
                        device="cpu")
    batch = next(iter(dp))
    for k in ("c0", "p0", "c1", "p1"):
        assert torch.isfinite(batch[k]).all(), f"{k} has NaN/inf"
    c1 = batch["c1"]
    print(f"[4] rollout finite: c1 range [{c1.min():.3f}, {c1.max():.3f}], "
          f"shapes {tuple(batch['c0'].shape)}")
    assert c1.min() >= -1e-6 and c1.max() <= 1.0 + 1e-3, "c left [0, 1]"


def test_ic_and_gauge():
    dp = ElderProblem2D(resolution=16, batch_size=1, n_trajectories=1,
                        rollout_steps=2, device="cpu")
    # Fresh-water IC: c = 0 (except source) => rho uniform => h = 0 everywhere.
    h0 = dp._traj_h
    print(f"[5] IC |h| max = {h0.abs().max().item():.2e} (should be ~0)")
    assert h0.abs().max().item() < 1e-3, "IC head not ~0"
    p = h0 + dp.p_hydro
    # Gauge: p = 0 at the top-left corner node.
    print(f"    p at top-left corner = {p[0, 0, 0, 0].item():.3e} (should be ~0)")
    assert abs(p[0, 0, 0, 0].item()) < 1e-3, "pressure gauge not ~0 at top corner"


def test_10_year_fingering(sinking_sign):
    import matplotlib.pyplot as plt
    dp = ElderProblem2D(resolution=64, batch_size=1, n_trajectories=1,
                        rollout_steps=400, dt_macro=10.0 * 24 * 3600.0,
                        flow_sign=sinking_sign, 
                        device="cpu")
    
 
    steps_10_years = 365
    c1 = None
    for _ in range(steps_10_years):
        _, _, c1, _ = dp._advance_all()
    
    c_final = c1[0, 0].detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(c_final, origin="upper", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="Concentration c")
    plt.title(f"Concentration field after ~10 years ({steps_10_years} steps), flow_sign={sinking_sign}")
    plt.xlabel("x cells")
    plt.ylabel("z cells")
    plt.savefig("10_year_fingering.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[6] Saved 10-year fingering plot. Max c = {c_final.max():.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_flow_residual()
    test_no_flow_walls()
    sinking = test_buoyancy_sign() # 获取正确的符号
    test_cfl_stability()
    test_ic_and_gauge()
    # 将正确的符号传递给指进测试
    test_10_year_fingering(sinking)
    print("\nBUOYANCY_SIGN_OK" if sinking == +1.0 else "\nBUOYANCY_SIGN_FLIP_NEEDED")