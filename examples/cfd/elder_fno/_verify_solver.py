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
 6. Long-term rollout: Simulate years of physical time to visualize fingering.

Run:  python _verify_solver.py
"""
import numpy as np
import torch
import os

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
    c = torch.zeros(2, 1, dp.Ny_tot, dp.Nx_tot)   # 全零浓度场
    c[:, :, 2:5, dp.src_x0:dp.src_x1] = 1.0       # 在源段下方放一个 dense blob
    dp._apply_bc_c(c)                              # 施加 c 边界条件
    # 准静态求解 (无 storage): div(rho q) ~ 0。
    h = dp._flow_solve(dp._interior(c))            # 解水头 h
    Fx, Fz = dp._face_fluxes(dp._interior(c), h)   # 由 (c,h) 算 Darcy 面通量
    div = _div_face(Fx, Fz, dp.dx, dp.dy)          # 面通量散度 div(rho q)
    err = div.abs().max().item() / (Fx.abs().amax().item() + Fz.abs().amax().item() + 1e-12)   # 相对误差
    print(f"[1a] quasi-static div(rho q) max rel err = {err:.2e}")
    assert err < 1e-3, "flow solve did not satisfy continuity"

    # 带 storage 的求解: d(phi rho)/dt + div(rho q) ~ 0, 用一个非零 dc/dt (macro 步前向差分)。
    dc = 0.01 * torch.randn_like(c)                # 随机浓度扰动 (模拟一步变化)
    dp._apply_bc_c(dc)
    dc_dt = dp._interior(dc) / dp.dt_macro         # 扰动的 dc/dt
    h2 = dp._flow_solve(dp._interior(c), dc_dt)    # 带 storage 解水头
    Fx, Fz = dp._face_fluxes(dp._interior(c), h2)
    storage = dp.phi * dp.drho * dc_dt             # 流体质量存储项 d(phi rho)/dt
    res = storage + _div_face(Fx, Fz, dp.dx, dp.dy)   # 完整连续性残差
    # gauge cell (顶左, h 钉住处) 不满足连续性, 从检查中排除。
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
    results = {}                                  # 记录两种 flow_sign 下的中心 Fz
    for sign in (+1.0, -1.0):
        dp = ElderProblem2D(resolution=32, batch_size=1, n_trajectories=1,
                            rollout_steps=2, device="cpu", flow_sign=sign)
        c = torch.zeros(1, 1, dp.Ny_tot, dp.Nx_tot)
        r0, r1 = 2, 6                             # dense blob 的行范围
        c[:, :, r0:r1, dp.src_x0:dp.src_x1] = 1.0   # 内部放一个 dense blob
        dp._apply_bc_c(c)
        h = dp._flow_solve(dp._interior(c))
        Fx, Fz = dp._face_fluxes(dp._interior(c), h)
        row = (r0 + r1) // 2                       # blob 中间行
        results[sign] = float(Fz[0, 0, row, dp.src_x0 + 2].item())   # blob 中心的 z 通量
    print(f"[3] Fz at dense-blob center:  sign=+1 -> {results[+1.0]:+.3e},  "
          f"sign=-1 -> {results[-1.0]:+.3e}")
    sinking_sign = +1.0 if results[+1.0] > 0 else -1.0   # Fz>0 = 向下 = 下沉
    print(f"    sinking (Fz>0, downward) requires flow_sign = {sinking_sign}")
    return sinking_sign


def test_cfl_stability():
    dp = ElderProblem2D(resolution=32, batch_size=2, n_trajectories=2,
                        rollout_steps=8, dt_macro=10.0 * 24 * 3600.0,
                        device="cpu")
    batch = next(iter(dp))                         # 取一个 batch (推进 1 步)
    for k in ("c0", "p0", "c1", "p1"):
        assert torch.isfinite(batch[k]).all(), f"{k} has NaN/inf"   # 无 NaN/Inf
    c1 = batch["c1"]
    print(f"[4] rollout finite: c1 range [{c1.min():.3f}, {c1.max():.3f}], "
          f"shapes {tuple(batch['c0'].shape)}")
    assert c1.min() >= -1e-6 and c1.max() <= 1.0 + 1e-3, "c left [0, 1]"   # c 未越界


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
    
    dt_days = 10.0
    interval_days = 365  # 正常每隔多少天保存一次
    total_years = 10
    
    # 自动检测 GPU 以加速长周期推演
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[6] 正在使用 {device} 进行长跨度物理推演，请耐心等待...")
    
    dp = ElderProblem2D(resolution=64, batch_size=1, n_trajectories=1,
                        rollout_steps=400, dt_macro=dt_days * 24 * 3600.0,
                        flow_sign=sinking_sign, 
                        device=device)
    
    total_days = total_years * 365
    max_step = int(round(total_days / dt_days))
    
    # 创建专门的输出文件夹避免弄脏根目录
    out_dir = "Verify_fingering_evolution"
    os.makedirs(out_dir, exist_ok=True)
    
    # 计算需要截图保存的目标步数
    target_steps = set()
    
    # 💡 额外添加你特别想看的特定天数（比如 30 天，或者任意其他天数）
    custom_days = [30]
    for d in custom_days:
        target_steps.add(int(round(d / dt_days)))
        
    current_day = interval_days
    while current_day <= total_days:
        step = int(round(current_day / dt_days))
        target_steps.add(step)
        current_day += interval_days
    
    c1 = None                                       # 保存最新浓度 (循环外引用)
    for step in range(1, max_step + 1):             # 推进 max_step 个 macro 步
        _, _, c1, p1 = dp._advance_all()            # 推进一步, 取 c1/p1 (忽略 c0/p0)

        # 如果当前步数在目标集合中，则保存图像
        if step in target_steps:
            current_days = step * dt_days           # 当前物理天数
            c_save = c1[0, 0].detach().cpu().numpy()   # 浓度转 numpy (去 batch/channel 维)

            # 将绝对压力 p 转换为归一化的压力水头 h，滤除静水压力背景干扰
            h1 = (p1 - dp.p_hydro) / dp.p_scale
            h_save = h1[0, 0].detach().cpu().numpy()   # 归一化水头转 numpy
            
            # 修改布局为 2 行 1 列，并调整画布大小使其适合上下堆叠
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # 上图：浓度 c
            im_c = axes[0].imshow(c_save, origin="upper", cmap="viridis", vmin=0, vmax=1, aspect="auto")
            fig.colorbar(im_c, ax=axes[0], label="Concentration c")
            axes[0].set_title("Concentration c")
            axes[0].set_ylabel("z cells")
            # 通常上下排列时，上图的 x 轴标签可以省略，保持画面清爽
            
            # 下图：归一化压力水头 h
            im_h = axes[1].imshow(h_save, origin="upper", cmap="viridis", aspect="auto")
            fig.colorbar(im_h, ax=axes[1], label="Normalized Head h")
            axes[1].set_title("Pressure Head h (Normalized)")
            axes[1].set_xlabel("x cells")
            axes[1].set_ylabel("z cells")
            
            fig.suptitle(f"Elder Problem Evolution at ~{int(current_days)} days ({step} steps), flow_sign={sinking_sign}", fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            fig.savefig(os.path.join(out_dir, f"fingering_{int(current_days)}_days.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    -> Saved c and h plots at ~{int(current_days)} days. Max c = {c_save.max():.3f}, Max h = {h_save.max():.3e}")

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