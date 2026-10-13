"""Bounded conditioning of Mamba's positive discretization step."""
import torch
from torch import nn
import torch.nn.functional as F


class BoundedDeltaConditioner(nn.Module):
    def __init__(self, eeg_dim=32, inner_dim=64, rank=4, eta=0.2):
        super().__init__()
        if not 0 <= eta <= 0.5:
            raise ValueError("eta out of registered range")
        self.eta = float(eta)
        self.net = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, rank, bias=False), nn.Tanh(), nn.Linear(rank, inner_dim, bias=False))
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, dt_pre_bias, dt_bias, eeg, profile):
        batch, inner, length = dt_pre_bias.shape
        if dt_bias.shape != (inner,) or profile.shape != (batch, length):
            raise ValueError("delta/profile shape mismatch")
        if not torch.isfinite(profile).all() or (profile < 0).any():
            raise ValueError("profile must be finite and nonnegative")
        q = profile.float() / profile.float().amax(-1, keepdim=True).clamp_min(1e-8)
        modulation = torch.tanh(self.net(eeg.detach())).float().unsqueeze(-1)
        factor = torch.exp(self.eta * modulation * q.unsqueeze(1))
        positive_base = F.softplus(dt_pre_bias.float() + dt_bias.float()[None, :, None])
        return positive_base * factor, factor


def conditioned_selective_scan(x, dt_pre_bias, state_b, state_c, z, mamba_core, conditioner, eeg, profile):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    positive_dt, factor = conditioner(dt_pre_bias, mamba_core.dt_proj.bias, eeg, profile)
    output = selective_scan_fn(x.contiguous(), positive_dt.to(x.dtype).contiguous(), -torch.exp(mamba_core.A_log.float()), state_b.contiguous(), state_c.contiguous(), mamba_core.D.float(), z=z.contiguous(), delta_bias=None, delta_softplus=False, return_last_state=False)
    return output, {"dt_factor": factor, "positive_dt": positive_dt}


def split_half_second_tokens(segments):
    if segments.ndim != 6 or segments.shape[1] != 13 or segments.shape[4] != 10:
        raise ValueError("expected [B,13,H,W,10,C]")
    batch, length, height, width, _, channels = segments.shape
    result = segments.reshape(batch, length, height, width, 2, 5, channels)
    result = result.permute(0, 1, 4, 2, 3, 5, 6).contiguous()
    centers = -3.0 + (torch.arange(26, device=segments.device).float() + 0.5) * 0.5
    return result.reshape(batch, 26, height, width, 5, channels), centers
