# Elder Problem FNO (PINO), variable-density (c, p) form

Physics-informed Fourier Neural Operator for the **Elder problem** in *primitive
variables* **(concentration `c`, pressure `p`)**, **without the Boussinesq
approximation** — the classic benchmark for density-driven (buoyancy-driven)
convection in a porous medium. A dense fluid (`c = 1`) is introduced through a
central segment of the top boundary of an initially fresh (`c = 0`) domain and
sinks under gravity, producing the characteristic descending convection fingers.

This mirrors the UG4 Lua setup (`elder` without Boussinesq): variable density
`rho(c) = rho_f + drho * c`, Darcy velocity, a variable-density Darcy flow
equation (with fluid-mass storage), and a conservative transport equation. The
model learns the **joint single-step
operator `(c_n, p_n) -> (c_{n+1}, p_{n+1})`** over a macro time step `dt`.
Training combines a **data loss** (MSE vs. a reference solution, on both `c` and
the pressure head) with a **PDE-residual loss** containing **two** residuals:
the conservative transport residual in `c` and the flow / continuity residual
`div(rho q)` in `p`. All training data is generated on the fly by a
pure-PyTorch reference solver — no external datasets are required.

## Governing equations (SI; non-Boussinesq)

`x` horizontal, `z` downward (row index increases downward), gravity
`g_vec = (0, +g)` downward, `rho(c) = rho_f + drho * c` (`c = 1` dense):

```
Darcy velocity :  q = -(k/mu)(grad p - rho g_vec)
flow (p)       :  d(phi rho)/dt + div(rho q) = 0     (mass_scale=0 drops dp/dt only;
                                                      storage d(phi rho)/dt = phi*drho*dc/dt)
transport (c)  :  d(phi rho c)/dt + div(rho q c) = div(rho phi Dm grad c)
```

with the conservative time term `d(phi rho c)/dt = phi (rho_f + 2 drho c) c_t`
(since `rho = rho_f + drho c`). The Darcy velocity `q` couples the two
equations: it is reconstructed from the predicted `(p, c)` for the physics
residual, and solved from `(c_n, p_n)` by the reference solver.

**Equivalent-freshwater-head gauge.** Pressure is worked in
`h = p - p_hydro` with `p_hydro(z) = rho_f * g * z` (fresh-water hydrostatic,
`h = 0` initially). In this gauge `q = -(k/mu) grad h + (k drho c / mu) g_vec`:
buoyancy appears as an explicit body force proportional to `c`, and the flow
equation becomes a variable-coefficient Poisson equation for `h`. The network
sees the normalized head `h / p_scale` (default `p_scale = drho * g * H`); real
pressure is recovered as `p = h + p_hydro` for the residual and for plotting.
The primary variables are still `c` and `p` — `h` is only a shifted/scaled
gauge of `p`.

## Boundary and initial conditions (match the UG4 Lua)

- Top wall: `c = 1` on the central source segment (`source_frac` of the width),
  zero-flux Neumann elsewhere.
- Bottom wall: `c = 0` (Dirichlet).
- Left/right walls: zero-flux Neumann (for both `c` and the flow).
- Pressure gauge: `h = 0` (i.e. `p = 0`) at the top-left cell — fixes the
  pressure datum (the Lua pins `p = 0` at the top corners).
- Initial condition: `p = p_hydro` (so `h = 0`), `c = 0` except the source.

The grid **includes** the boundary nodes (shape `[B, 1, Ny+2, Nx+2]`), so the
FNO sees the walls directly.

## Files

| File | Purpose |
|------|---------|
| `datapipe.py` | `ElderProblem2D` — online reference solver: variable-coefficient dense flow solve for the head `h`, non-periodic conservative finite-difference transport (full-upwind advection), trajectory sampling. |
| `train_elder_fno.py` | PINO training loop (data + transport/continuity residual loss), validation, plotting. |
| `config.yaml` | Hydra configuration. |
| `elder_pde.py` | Symbolic transport + flow operators for the optional `phy_informer` backend (not used by the default `own_fd` path; kept as a reference). |
| `_verify_solver.py` | Sanity checks: flow continuity residual, no-flow walls, buoyancy direction, CFL stability, IC/gauge. |

## Design notes

- **Joint operator.** The FNO predicts both `c_{n+1}` and `h_{n+1}` (hence
  `p_{n+1}`). The Darcy face fluxes are solved from `(c_n, h_n)` and frozen over
  one macro step while the transport equation is advanced; `h_{n+1}` is then
  re-solved from `c_{n+1}`. This makes `(c_{n+1}, p_{n+1})` consistent with the
  single-step operator the FNO learns; the buoyancy coupling is captured
  *across* macro steps via trajectory sampling. Both the transport residual
  (in `c`) and the continuity residual `d(phi rho)/dt + div(rho q)` (in `p`,
  via Darcy from the predictions) train the FNO.
- **Dense flow solve (float64).** The flow equation (with the fluid-mass
  storage `d(phi rho)/dt = phi*drho*dc/dt` moved to the RHS)
  `-div((rho k/mu) grad h) = -phi*drho*dc/dt - g * d/dz((rho k/mu) drho c)` is a
  variable-coefficient Poisson equation (variable `rho(c)`, no-flow walls, one
  Dirichlet gauge node). The storage rate uses the forward difference
  `(c_{n+1} - c_n)/dt` (known once the transport step is done). It is assembled
  as a batched dense matrix and solved with `torch.linalg.solve` **in float64**
  — the SI coefficients `alpha/dx^2 ~ 1e-9` make the matrix condition (~`N^2`)
  exceed float32
  precision. The result is cast back to float32. This is faster than a
  Python-loop iterative solver on both CPU and GPU and scales as
  `O((Nx*Ny)^3)`, so raise resolution with care. The buoyancy face value uses
  cell-product-then-average so the face-flux divergence is exactly consistent
  with the flow RHS (continuity satisfied to ~1e-8, verified).
- **Trajectory sampling (strategy B).** `n_trajectories` independent rollouts
  are advanced one macro step at a time (pre-rolled to staggered phases, in
  parallel batched rounds) and reset after `rollout_steps`; each batch thus
  spans different stages of the fingering. The flow solve is batched over
  trajectories, so one round costs a single `linalg.solve`.
- **Non-periodic PDE residual (own_fd).** The residuals use hand-written
  non-periodic central finite differences on the interior (correct all the way
  to the walls). Residuals are normalized by physical scales
  (`phi*rho_f/dt` for transport, `rho_f*q_ref/H` with `q_ref = k*drho*g/mu`
  for continuity) so they are O(1) and `physics_weight` is easy to tune.
- **Residual masking.** The top interior rows (the imposed source and the stiff
  diffusion front just below it) are excluded from the residual via
  `mask_top_rows`; the fingering region deeper in the interior provides the
  physics signal.
- **Explicit-Euler reference.** The reference solver sub-steps the macro step
  to honor the CFL condition (min of diffusion and advection bounds),
  re-imposing the boundary conditions every sub-step. It is a demo-grade
  solver; the FNO matches it via the data loss and the residual uses the same
  spatial discretization, so the setup is self-consistent.
- **Flow storage term.** The flow equation keeps the fluid-mass storage
  `d(phi rho)/dt = phi*drho*dc/dt` (the Lua's `set_mass(phi*rho)` with
  `mass_scale=0` drops only the pressure-storage `dp/dt`, i.e. `Ss=0`, not the
  density storage). It is treated with a forward difference `(c_{n+1}-c_n)/dt`
  as a known RHS in the (still elliptic) head solve. For Elder's slow evolution
  and 20% density contrast its effect on `h` is ~0.03%, but it is retained for
  faithfulness to the UG4 reference.
- **Conservative transport time term.** The transport time term is the
  conservative `d(phi rho c)/dt = phi (rho_f + 2 drho c) c_t`. UG4's
  `ConvectionDiffusionFV1` discretization effectively uses `phi rho c_t`; the
  difference is the small `phi c d rho/dt` term, negligible for Elder's 20%
  density contrast but noted for exact UG4 comparison.

## Run

```bash
cd examples/cfd/elder_fno
python train_elder_fno.py
```

Outputs (Hydra redirects the working dir to `./outputs_elder_fno/`):
- `checkpoints/` — model + optimizer + scheduler state
- `val_XXXX.png` — true / predicted / error comparison for **both** `c` and `h`
  every `val_every` epochs

### Quick smoke test

```bash
python train_elder_fno.py \
  data.resolution=32 data.batch_size=8 data.steps_per_epoch=4 \
  data.rollout_steps=32 data.n_trajectories=8 \
  training.max_epochs=2 training.val_every=1
```

### Sanity-check the solver

```bash
python _verify_solver.py
```

### Pure data-driven (disable physics loss)

```bash
python train_elder_fno.py training.physics_weight=0
```

## Tuning guide

- **`physics_weight`**: residuals are normalized to O(1), so start with
  `physics_weight=0` (pure data-driven) to confirm the joint `(c, h)` operator
  learns, then increase from `1e-2` upward. `~0.1` makes the physics term
  comparable to the data term. `continuity_weight` rebalances the
  `div(rho q)` residual relative to the transport residual.
- **`p_data_weight`**: weight of the pressure-head data loss relative to the
  concentration data loss.
- **`dt_macro`**: time span of the learned operator. With the SI parameters the
  buoyancy Darcy speed is `~1e-6 m/s`, so `dt_macro = 10 days` (the Lua value)
  advects well under one cell — a mild operator. Increase `dt_macro` (e.g. to
  `~100 days`, `q*dt/dx ~ 1`) for a more non-trivial single-step operator at
  the cost of more CFL sub-steps and a more diffused reference.
- **`rollout_steps`**: how long each trajectory runs before resetting. Increase
  to sample later, more-developed finger stages (the fingers take many macro
  steps to reach the bottom).
- **`num_fno_modes`**: a per-axis list `[modes_y, modes_x]` for the 4:1 grid
  (e.g. `[8, 12]`); increase for sharper / higher-frequency solutions.
- **`flow_sign`**: dense fluid should sink (fingers descend from the top
  source). If they rise instead, flip `flow_sign` (and re-run
  `_verify_solver.py`, which reports the correct value).
- **Resolution**: the dense flow solve is `O((Nx*Ny)^3)`; `resolution=64`
  (`Nx*Ny = 1024`) is comfortable on GPU. Higher resolutions are slower.

## GPU performance

- **`training.use_amp`** (default `false`): mixed precision via `torch.autocast`
  + `GradScaler`. The PDE residual is kept in fp32 (autocast disabled inside)
  to preserve finite-difference precision (the flow *solve* is always fp64).
  Auto-disabled on CPU.
- **`training.tf32`** (default `true`): TF32 matmul/conv + `cudnn.benchmark`.
  Free win on Ampere+; no-op on CPU.
- **`training.compile`** (default `false`): `torch.compile` the FNO forward
  only (residuals stay eager). Best on CUDA/Linux for longer runs.
- **Multi-GPU (DDP)**: launch with
  `torchrun --nproc_per_node=<N> train_elder_fno.py ...`. Each rank generates
  its own on-the-fly batches and gradients are all-reduced automatically.

```bash
# single GPU, AMP + compile
python train_elder_fno.py training.use_amp=true training.compile=true seed=42

# multi-GPU via torchrun
torchrun --nproc_per_node=4 train_elder_fno.py training.use_amp=true training.compile=true seed=42
```

## References

- Elder, J. W. (1967). Transient convection in a porous medium.
- Ackerer, P., et al. — reformulated (concentration-Dirichlet) Elder benchmark.
- Johannsen, K. / Kolditz, O. — variable-density Darcy flow forms of the Elder
  problem.
- UG4 `elder` (without Boussinesq) Lua script — the reference setup this example
  reproduces.
