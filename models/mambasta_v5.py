"""V5 fusion candidates built on the audited V4 region backbone."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from models.lightweight_baseline import LightweightEncoder
from models.mamba_sta_v2 import PhysicalTimeEmbedding
from models.mambasta_v4 import DelayPriorReadout, frozen_eeg_forward
from models.mambasta_v3 import GatedTemporalConv
from models.mambasta_v4_fusion import Mamba, V4AnchoredFusion, causal_conv1d_fn
from models.v5_delta import BoundedDeltaConditioner, conditioned_selective_scan, split_half_second_tokens
from models.v5_heads import ConservativeResidualHead, ProbabilityFusionHead


class ConditionalDeltaMamba(nn.Module):
    """Mamba-1 slow scan with one bounded, positive dt modification."""
    def __init__(self, dim=32, state=8, conv_kernel=4, expand=2):
        super().__init__()
        self.core = Mamba(d_model=dim, d_state=state, d_conv=conv_kernel, expand=expand, use_fast_path=False)
        self.conditioner = BoundedDeltaConditioner(dim, self.core.d_inner, rank=4, eta=0.2)

    def forward(self, hidden_states, eeg, profile):
        module = self.core; batch, seqlen, _ = hidden_states.shape
        xz = rearrange(module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if module.in_proj.bias is not None: xz = xz + rearrange(module.in_proj.bias.to(xz.dtype), "d -> d 1")
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None: x = module.act(module.conv1d(x)[..., :seqlen])
        else: x = causal_conv1d_fn(x=x, weight=rearrange(module.conv1d.weight, "d 1 w -> d w"), bias=module.conv1d.bias, activation=module.activation)
        projected = module.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, state_b, state_c = torch.split(projected, [module.dt_rank, module.d_state, module.d_state], dim=-1)
        dt = module.dt_proj.weight @ dt.t(); dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        state_b = rearrange(state_b, "(b l) n -> b n l", l=seqlen).contiguous(); state_c = rearrange(state_c, "(b l) n -> b n l", l=seqlen).contiguous()
        y, diagnostics = conditioned_selective_scan(x, dt, state_b, state_c, z, module, self.conditioner, eeg, profile)
        return module.out_proj(rearrange(y, "b d l -> b l d")), diagnostics


def replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            groups = max(group for group in range(1, 5) if child.num_features % group == 0)
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_batch_norm(child)


class MambaSTAV5(V4AnchoredFusion):
    """Final-search model. W1 families are implemented first and share one backbone."""

    def __init__(self, family, mixer, *, bound=0.5, alpha=0.25, profile="uniform", **kwargs):
        if mixer not in {"mamba", "conv"}:
            raise ValueError(mixer)
        super().__init__(f"r_region_{mixer}", **kwargs)
        self.v5_family = family
        self.mixer_kind = mixer
        self.profile_kind = profile
        if family == "conservative_residual":
            self.v5_head = ConservativeResidualHead(kwargs.get("dim", 32), bound=bound, penalty=0.01)
            self.output_kind = "residual"
        elif family == "probability":
            self.v5_head = ProbabilityFusionHead(kwargs.get("dim", 32), alpha=alpha)
            self.output_kind = "probability"
        elif family in {"region_mean", "conditional_delta", "half_second", "group_norm", "trial_loss"}:
            self.v5_head = ConservativeResidualHead(kwargs.get("dim", 32), bound=1.0 if family == "conditional_delta" else 2.0, penalty=0.01 if family == "conditional_delta" else 0.0)
            self.output_kind = "residual"
            if family == "region_mean":
                nn.init.zeros_(self.region_score.weight); nn.init.zeros_(self.region_score.bias); self.region_score.requires_grad_(False)
            if family == "conditional_delta":
                self.mixer = ConditionalDeltaMamba(kwargs.get("dim",32), kwargs.get("state",8), kwargs.get("conv_kernel",4), kwargs.get("expand",2))
                self.register_buffer("constant_eeg_feature", torch.zeros(kwargs.get("dim",32)), persistent=True)
            if family == "half_second":
                centers = -3.0 + (torch.arange(26).float() + 0.5) * 0.5
                self.readout = DelayPriorReadout(centers, dim=kwargs.get("dim",32), rank=kwargs.get("rank",8), beta=0.5, uniform_floor=0.2)
                if mixer == "conv": self.mixer = GatedTemporalConv(kwargs.get("dim",32), self.mixer.inner, dilations=(2,4,8))
            if family == "group_norm": replace_batch_norm(self.patch_encoder)
        else: raise ValueError(f"unsupported V5 family: {family}")

    def set_constant_eeg(self, value):
        if self.v5_family != "conditional_delta" or self.profile_kind != "eeg_constant": return
        if value.shape != self.constant_eeg_feature.shape: raise ValueError("constant EEG feature shape mismatch")
        self.constant_eeg_feature.copy_(value.detach())

    def encode_v5_regions(self, fnirs):
        segments = self.adapter(fnirs)
        if self.v5_family == "half_second": segments, centers = split_half_second_tokens(segments)
        else: centers = self.adapter.center_seconds.to(fnirs.device, fnirs.dtype)
        batch, length, _, _, width, _ = segments.shape
        regions=(segments[:,:,:8,:8],segments[:,:,:8,8:],segments[:,:,8:,:8],segments[:,:,8:,8:])
        regional=torch.stack(regions,1)
        flat=regional.permute(0,1,2,6,3,4,5).reshape(batch*4*length,2,8,8,width)
        tokens=self.patch_encoder(flat).reshape(batch,4,length,-1)
        tokens=tokens+self.time_embedding(centers.to(fnirs.device,fnirs.dtype))[None,None]
        if self.v5_family == "conditional_delta":
            eeg = self._teacher_feature
            if self.profile_kind == "eeg_constant": eeg = self.constant_eeg_feature[None].expand(batch,-1)
            if self.profile_kind == "uniform" or self.profile_kind == "eeg_constant": profile=torch.ones(batch,length,device=fnirs.device,dtype=fnirs.dtype)
            else:
                profile=self.readout.bases.mean(0)
                if self.profile_kind == "reverse": profile=profile.flip(0)
                profile=profile[None].expand(batch,-1)
            mixed, scan = self.mixer(tokens.reshape(batch*4,length,-1), eeg[:,None].expand(-1,4,-1).reshape(batch*4,-1), profile[:,None].expand(-1,4,-1).reshape(batch*4,length))
            mixed=mixed.reshape_as(tokens)
        else:
            mixed=self.mixer(tokens.reshape(batch*4,length,-1)).reshape_as(tokens); scan={}
        memory=tokens+mixed; values=[]; weights=[]
        for region in range(4):
            value,item=self.readout(self._teacher_feature,memory[:,region]);values.append(value);weights.append(item["weights"])
        region_values=torch.stack(values,1)+self.region_embedding[None]
        if self.v5_family == "region_mean": region_weights=torch.full((batch,4),0.25,device=fnirs.device,dtype=fnirs.dtype)
        else: region_weights=self.region_score(region_values).squeeze(-1).softmax(-1)
        feature=torch.einsum("br,brd->bd",region_weights,region_values)
        return feature,{"weights":torch.stack(weights,1),"region_weights":region_weights,"memory":memory,**scan}

    def forward(self, eeg, fnirs, *, intervention=None, return_diagnostics=False):
        del intervention
        teacher_logits, teacher_feature = frozen_eeg_forward(self.teacher, eeg)
        self._eeg_grid = eeg
        if self.v5_family == "probability":
            # A zero query makes the temporal readout prior-only. Therefore the
            # auxiliary branch is independent of EEG and can legitimately be N-only.
            self._teacher_feature = torch.zeros_like(teacher_feature)
            n_feature, path = self.encode_v5_regions(fnirs)
            output, head = self.v5_head(teacher_logits, n_feature)
        else:
            self._teacher_feature = teacher_feature
            n_feature, path = self.encode_v5_regions(fnirs)
            output, head = self.v5_head(teacher_logits, teacher_feature, n_feature)
        diagnostics = {"teacher_logits": teacher_logits, "teacher_feature": teacher_feature, "n_feature": n_feature, **path, **head} if return_diagnostics else {}
        return output, diagnostics


def make_mambasta_v5(config):
    return MambaSTAV5(
        config["family"], config["mixer"], bound=config.get("bound", 0.5), alpha=config.get("alpha", 0.25), profile=config.get("profile", "uniform"),
        dim=config.get("d_model", 32), state=config.get("d_state", 8), conv_kernel=config.get("d_conv", 4),
        expand=config.get("expand", 2), rank=config.get("query_rank", 8), dropout=config.get("dropout", 0.2),
    )
