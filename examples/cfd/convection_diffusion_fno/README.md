# Convection-Diffusion FNO (PINO)

Physics-informed Fourier Neural Operator for the 2-D time-dependent
convection-diffusion (CD) equation:

```
dT/dt + u * dT/dx + v * dT/dy - D * (d2T/dx2 + d2T/dy2) = 0
```

The model learns the single-step operator `T_n, u, v -> T_{n+1}` over a macro
time step `dt`. Training combines a **data loss** (MSE vs. a reference
solution) with a **PDE-residual loss** (the full CD residual evaluated on the
prediction). All training data is generated on the fly by a pure-PyTorch
pseudo-spectral solver — no external datasets are required.

## Files

| File | Purpose |
|------|---------|
| `cd_pde.py` | Symbolic spatial operator `adv_diff` for `PhysicsInformer`. |
| `datapipe.py` | `ConvectionDiffusion2D` — online pseudo-spectral solver yielding `{T0, T1, u, v, dt}`. |
| `train_cd_fno.py` | PINO training loop (data + PDE-residual loss), validation, plotting. |
| `config.yaml` | Hydra configuration. |

## Design notes

- **Time derivative**: `PhysicsInformer` only computes spatial (x/y) derivatives,
  so the PDE class defines the spatial operator
  `u*T_x + v*T_y - D*(T_xx + T_yy)`. The time term `T_t = (T_{n+1} - T_n)/dt`
  is added manually in the training script to form the residual
  `T_t + adv_diff`.
- **Velocity field**: divergence-free by construction — derived from a random
  streamfunction `psi` as `u = d psi/dy`, `v = -d psi/dx` in spectral space, so
  `div(u, v) = 0` exactly.
- **Reference solver**: pseudo-spectral derivatives + explicit-Euler sub-steps
  on a periodic domain. Sub-step count is auto-increased to honor the CFL
  condition `dt < min(0.25*dx^2/D, dx/u_max)`. The k=0 (mean) mode is pinned to
  conserve mass.
- **PDE-residual boundary**: the finite-difference stencil is unreliable at the
  domain edge, so 2 cells are cropped from the residual (then zero-padded back)
  before the L1-to-zero loss, mirroring the Darcy PINO example.

## Run

```bash
cd examples/cfd/convection_diffusion_fno
python train_cd_fno.py
```

Outputs (Hydra redirects the working dir to `./outputs_cd_fno/`):
- `checkpoints/` — model + optimizer + scheduler state
- `val_XXXX.png` — true / predicted / error comparison every `val_every` epochs

### Quick smoke test

Override config to run a fast end-to-end pass before a real training run:

```bash
python train_cd_fno.py \
  data.resolution=32 data.batch_size=8 data.steps_per_epoch=8 \
  training.max_epochs=2 training.val_every=1
```

### Pure data-driven (disable physics loss)

```bash
python train_cd_fno.py training.physics_weight=0
```

## Tuning guide

- **`physics_weight`**: the PDE-residual and data loss often differ in scale.
  Start with `physics_weight=0` (pure data-driven) to confirm the model learns,
  then increase from `1e-2` upward. If training destabilizes, lower it.
- **`D`, `u_max`, `dt_macro`**: set the PDE regime. Larger `dt_macro` or
  stronger advection makes the single-step map harder; the solver auto-increases
  sub-steps for stability, but the operator-learning task gets harder.
- **`num_fno_modes`, `latent_channels`**: increase for sharper / higher-frequency
  solutions.
- **`resolution`**: 64 is a good default; the FNO is resolution-invariant so a
  model trained at 64 can be evaluated at higher resolution (adjust `fd_dx`
  accordingly: `fd_dx = length / resolution`).

## GPU performance

The script is multi-GPU ready and exposes precision knobs for CUDA servers:

- **`training.use_amp`** (default `false`): mixed precision via `torch.autocast`
  + `GradScaler`. Enable on GPU for a sizable speedup. The PDE residual is kept
  in fp32 (autocast disabled inside) to preserve finite-difference precision,
  while the FNO forward runs in AMP. Auto-disabled on CPU.
- **`training.tf32`** (default `true`): allows TF32 matmul/conv and enables
  `cudnn.benchmark`. Free win on Ampere+; no-op on CPU.
- **`seed`** (default `0`): when > 0, seeds torch/numpy per-rank
  (`seed + dist.rank`) so DDP ranks generate different data — set it for
  reproducibility.
- **`training.compile`** (default `false`): `torch.compile` the FNO forward
  (only the model — `PhysicsInformer` stays eager). Best on CUDA/Linux for
  longer runs; the first iterations pay a one-time compile cost. Use
  `compile_mode: "reduce-overhead"` (CUDA-graphs based) or `"max-autotune"` for
  more speed at higher compile cost. Verified to trace cleanly (forward +
  backward) under dynamo. Note: on a Windows CPU box the inductor backend
  needs MSVC (`cl`); use this flag on the GPU server instead.
- **Multi-GPU (DDP)**: `DistributedManager.initialize()` is called at startup.
  Launch with `torchrun --nproc_per_per_node=<N> train_cd_fno.py ...`. Each rank
  generates its own on-the-fly batches (different RNG) and gradients are
  all-reduced automatically.

Typical GPU run:

```bash
# single GPU, AMP + compile
python train_cd_fno.py training.use_amp=true training.compile=true seed=42

# multi-GPU via torchrun
torchrun --nproc_per_node=4 train_cd_fno.py training.use_amp=true training.compile=true seed=42
```

The reference solver is also optimized: the diffusion term is added directly in
spectral space, so each integration sub-step costs 3 FFTs instead of 4.

## Sanity-checking the datapipe

```bash
python -c "from datapipe import ConvectionDiffusion2D; \
d = ConvectionDiffusion2D(batch_size=2, device='cpu'); \
b = next(iter(d)); print({k: (v.shape if hasattr(v,'shape') else v) for k,v in b.items()})"
# expect: {'T0': [2,1,64,64], 'T1': [2,1,64,64], 'u': [2,1,64,64], 'v': [2,1,64,64], 'dt': 0.1}
```
