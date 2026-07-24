"""验证 model.load(.mdlus) 能否替代 load_checkpoint (跳过 1.5GB .pt).
跑通后 fno_step.py 每步只读 .mdlus, 快很多. 对比 infer.py 的 pred_step.npz."""
import glob, os
import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.fno import FNO
from vtu_dataset import VtuElderDataset
from train_elder_fno import _resolve_fno_modes, _resolve_in_channels, build_invar

cfg = OmegaConf.load("config.yaml")
phy = cfg.physics
device = torch.device("cpu")
DistributedManager.initialize()

dp = VtuElderDataset("DataSet", 1, device, phy.phi, phy.Dm, phy.permeability,
                     phy.viscosity, phy.g, phy.rho_f, phy.drho, phy.W, phy.H,
                     phy.dt_macro, file_stride=int(cfg.data.get("file_stride", 1)))
mdl = cfg.model
modes = _resolve_fno_modes(
    OmegaConf.to_container(mdl, resolve=True)["num_fno_modes"], dp, mdl.padding)
model = FNO(
    in_channels=_resolve_in_channels(mdl), out_channels=mdl.out_channels,
    decoder_layers=mdl.decoder_layers, decoder_layer_size=mdl.decoder_layer_size,
    dimension=mdl.dimension, latent_channels=mdl.latent_channels,
    num_fno_layers=mdl.num_fno_layers, num_fno_modes=modes, padding=mdl.padding,
).to(device)

mdlus = sorted(glob.glob("outputs_baseline30days/checkpoints/*.mdlus"))
mdlus.sort(key=lambda f: int(os.path.basename(f).rsplit(".", 2)[-2]))
print("loading", mdlus[-1])
model.load(mdlus[-1])          # <-- 关键: 直接 .mdlus, 不读 1.5GB .pt
model.eval()

c_n = dp.data[0:1, 0:1]
P_n = dp.data[0:1, 1:2]
h_n = (P_n - dp.p_hydro) / dp.p_scale
_dt_aware = bool(cfg.model.get("dt_channel", False))
_dt_ref = float(cfg.model.get("dt_ref_s", 2.592e6))
with torch.no_grad():
    raw = model(build_invar(c_n, h_n, dp.dt_macro, _dt_aware, _dt_ref))
c_pred = raw[0, 0].numpy()
print("model.load c_pred range:", float(c_pred.min()), float(c_pred.max()))

ref = np.load("pred_step.npz")["c"]
print("infer.py  c_pred range:", float(ref.min()), float(ref.max()))
print("max|diff| vs infer.py:", float(np.max(np.abs(c_pred - ref))))
print("OK" if float(np.max(np.abs(c_pred - ref))) < 1e-3 else "MISMATCH")
