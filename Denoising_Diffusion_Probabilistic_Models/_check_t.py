"""Is the per-timestep loss curve real, or is my reconstruction wrong?"""
import torch
import torch.nn.functional as F

import diffusion_models as dm
from diffusion_models import DiffusionModel, device, testX
from evaluate import upscale32
from run_simple_unet import build_model

T = 1000
ckpt = torch.load("checkpoints/simple_unet_100.pt", map_location=device)
model = build_model("simple").to(device)
model.load_state_dict(ckpt["model"])
diff = DiffusionModel(T, model).to(device)
model.eval()

x0 = upscale32(testX[:64]).to(device)
print(f"{'t':>5} {'model':>9} {'predict0':>9} {'predict_xt':>10}")
for t in (1, 2, 5, 10, 100, 500, 900, 1000):
    ab = diff.alpha_bar[t - 1]
    b = torch.sqrt(1 - ab)
    m_model = m_zero = m_id = 0.0
    draws = 8
    with torch.no_grad():
        for _ in range(draws):
            eps = torch.randn_like(x0)
            xt = torch.sqrt(ab) * x0 + b * eps
            pred = model(xt, torch.full((len(x0),), t - 1, dtype=torch.long, device=device))
            m_model += F.mse_loss(pred, eps).item() / draws
            m_zero += F.mse_loss(torch.zeros_like(eps), eps).item() / draws
            m_id += F.mse_loss(xt, eps).item() / draws
    print(f"{t:5d} {m_model:9.4f} {m_zero:9.4f} {m_id:10.4f}   (b={float(b):.4f})")
