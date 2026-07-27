"""验证 model.load(.mdlus) 能否替代 load_checkpoint (跳过 1.5GB .pt).
跑通后 fno_step.py 每步只读 .mdlus, 快很多."""
import glob
import os

import torch
from omegaconf import OmegaConf

from physicsnemo.distributed import DistributedManager

from ufno import build_model
from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes, _resolve_in_channels, build_invar

cfg = OmegaConf.load("config.yaml")
phy = cfg.physics
device = torch.device("cpu")
DistributedManager.initialize()

dp = VtuElderDataset(
    "DataSet",
    1,
    device,
    phy.phi,
    phy.Dm,
    phy.permeability,
    phy.viscosity,
    phy.g,
    phy.rho_f,
    phy.drho,
    phy.W,
    phy.H,
    phy.dt_macro,
    file_stride=int(cfg.data.get("file_stride", 1)),
)
mdl = cfg.model
modes = _resolve_fno_modes(
    OmegaConf.to_container(mdl, resolve=True)["num_fno_modes"], dp, mdl.padding
)
model = build_model(
    mdl, num_fno_modes=modes, in_channels=_resolve_in_channels(mdl)
).to(device)

ckpt_dirs = [
    "outputs_elder_ufno/checkpoints",
    "outputs_elder_fno/checkpoints",
    "outputs_baseline30days/checkpoints",
]
mdlus = []
for d in ckpt_dirs:
    mdlus = sorted(glob.glob(os.path.join(d, "*.mdlus")))
    if mdlus:
        break
if not mdlus:
    raise SystemExit("no *.mdlus found under outputs_*/checkpoints")

mdlus.sort(key=lambda f: int(os.path.basename(f).rsplit(".", 2)[-2]))
print("loading", mdlus[-1], "arch=", mdl.get("arch", "fno"))
model.load(mdlus[-1])
model.eval()

c_n = dp.data[0:1, 0:1]
P_n = dp.data[0:1, 1:2]
h_n = (P_n - dp.p_hydro) / dp.p_scale
_dt_aware = bool(cfg.model.get("dt_channel", False))
_dt_ref = float(cfg.model.get("dt_ref_s", 2.592e6))
with torch.no_grad():
    raw = model(build_invar(c_n, h_n, dp.dt_macro, _dt_aware, _dt_ref))
print("forward OK, out shape", tuple(raw.shape), "c_delta range",
      float(raw[0, 0].min()), float(raw[0, 0].max()))
