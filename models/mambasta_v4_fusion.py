"""Frozen-EEG anchored residual fusion models for V4 S2."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn, selective_scan_fn
except ImportError as exc:
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None

from models.encoder_audit import EEGNetStyleEncoder
from models.lightweight_baseline import LightweightEncoder
from models.mambasta_v3 import GatedTemporalConv, active_parameters, parameter_matched_inner
from models.mambasta_v4 import AnchoredResidualHead, DelayPriorReadout, LagConditionedWrite, freeze_teacher, frozen_eeg_forward, positive_hrf_bank
from models.mambasta_v3 import EEG32
from models.mamba_sta_v2 import PhysicalTimeEmbedding
from models.time_provenance_adapter import ChronologicalFNIRSAdapter


class LagWriteMamba(nn.Module):
    """Mamba-1 slow path with only input-dependent B conditionally modulated."""
    def __init__(self, dim=32, state=8, conv_kernel=4, expand=2):
        super().__init__()
        self.core = Mamba(d_model=dim, d_state=state, d_conv=conv_kernel, expand=expand, use_fast_path=False)
        self.condition = LagConditionedWrite(dim=dim, state=state, rank=4, rho=0.5)

    def forward(self, hidden_states, eeg, lag_prior):
        module=self.core;batch,seqlen,_=hidden_states.shape
        xz=rearrange(module.in_proj.weight @ rearrange(hidden_states,"b l d -> d (b l)"),"d (b l) -> b d l",l=seqlen)
        if module.in_proj.bias is not None:xz=xz+rearrange(module.in_proj.bias.to(xz.dtype),"d -> d 1")
        x,z=xz.chunk(2,dim=1)
        if causal_conv1d_fn is None:x=module.act(module.conv1d(x)[...,:seqlen])
        else:x=causal_conv1d_fn(x=x,weight=rearrange(module.conv1d.weight,"d 1 w -> d w"),bias=module.conv1d.bias,activation=module.activation)
        x_dbl=module.x_proj(rearrange(x,"b d l -> (b l) d"));dt,state_b,state_c=torch.split(x_dbl,[module.dt_rank,module.d_state,module.d_state],dim=-1)
        dt=module.dt_proj.weight @ dt.t();dt=rearrange(dt,"d (b l) -> b d l",l=seqlen)
        state_b=rearrange(state_b,"(b l) n -> b n l",l=seqlen).contiguous();state_c=rearrange(state_c,"(b l) n -> b n l",l=seqlen).contiguous()
        state_b=self.condition(state_b,eeg,lag_prior)
        y=selective_scan_fn(x,dt,-torch.exp(module.A_log.float()),state_b.contiguous(),state_c,module.D.float(),z=z,delta_bias=module.dt_proj.bias.float(),delta_softplus=True,return_last_state=False)
        return module.out_proj(rearrange(y,"b d l -> b l d"))


class SlowFastMixer(nn.Module):
    """Fixed-width 16+16 slow/fast mixer with Mamba or dilated-conv slow path."""
    def __init__(self, kind, dim=32):
        super().__init__();self.kind=kind;self.input=nn.Linear(dim,dim,bias=False)
        self.fast=nn.Conv1d(16,16,3,groups=16,bias=False)
        if kind=="mamba":self.slow=Mamba(d_model=16,d_state=8,d_conv=4,expand=2,use_fast_path=False)
        elif kind=="conv":self.slow=GatedTemporalConv(16,40)
        else:raise ValueError(kind)
        self.output=nn.Linear(dim,dim,bias=False)

    def forward(self,tokens):
        fast,slow=self.input(tokens).split(16,-1)
        fast=self.fast(F.pad(fast.transpose(1,2),(2,0))).transpose(1,2)
        # A three-point smooth trend is the slow input; sequence length stays 13.
        trend=F.avg_pool1d(F.pad(slow.transpose(1,2),(2,0),mode="replicate"),3,stride=1).transpose(1,2)
        return self.output(torch.cat((F.silu(fast),self.slow(trend)),-1))


class V4AnchoredFusion(nn.Module):
    VALID = {"r_e", "r_u", "r_q", "r_h", "r_hm", "r_hc",
             "r_h_adaptive", "r_h_wrong", "r_3q", "r_3h",
             "r_region_conv", "r_region_mamba", "r_gate_fixed", "r_gate_learned"}
    VALID |= {"r_lagwrite", "r_lagwrite_uniform", "r_slowfast_mamba", "r_slowfast_conv"}

    def __init__(self, architecture, dim=32, state=8, conv_kernel=4, expand=2, rank=8, dropout=0.2):
        super().__init__()
        if architecture not in self.VALID:
            raise ValueError(f"unknown V4 fusion architecture: {architecture}")
        self.architecture = architecture
        self.teacher = freeze_teacher(EEG32(dim=dim, dropout=dropout))
        self.adapter = ChronologicalFNIRSAdapter()
        if architecture == "r_e":
            self.eeg_control = EEGNetStyleEncoder(dim)
            self.patch_encoder = None
            self.time_embedding = None
            self.mixer = None
            self.readout = None
        else:
            self.eeg_control = None
            self.patch_encoder = LightweightEncoder(2, (5, 3), (2, 1), (8, dim))
            self.time_embedding = PhysicalTimeEmbedding(dim)
            if architecture in {"r_hm", "r_region_mamba"}:
                if Mamba is None:
                    raise ImportError("R-HM requires mamba_ssm") from MAMBA_IMPORT_ERROR
                self.mixer = Mamba(d_model=dim, d_state=state, d_conv=conv_kernel, expand=expand, use_fast_path=False)
            elif architecture in {"r_hc", "r_region_conv", "r_gate_fixed", "r_gate_learned"}:
                if Mamba is None:
                    raise ImportError("R-HC matching requires mamba_ssm") from MAMBA_IMPORT_ERROR
                target = active_parameters(Mamba(d_model=dim, d_state=state, d_conv=conv_kernel, expand=expand, use_fast_path=False))
                width = parameter_matched_inner("conv", target, dim=dim)
                self.mixer = GatedTemporalConv(dim, width)
            elif architecture in {"r_lagwrite", "r_lagwrite_uniform"}:
                self.mixer = LagWriteMamba(dim, state, conv_kernel, expand)
            elif architecture == "r_slowfast_mamba":
                self.mixer = SlowFastMixer("mamba", dim)
            elif architecture == "r_slowfast_conv":
                self.mixer = SlowFastMixer("conv", dim)
            else:
                self.mixer = nn.Identity()
            beta = 0.0 if architecture in {"r_q", "r_3q"} else 0.5
            self.readout = None if architecture == "r_u" else DelayPriorReadout(
                self.adapter.center_seconds, dim=dim, rank=rank, beta=beta,
                uniform_floor=0.2, adaptive=architecture == "r_h_adaptive"
            )
            if architecture in {"r_3q", "r_3h"}:
                centers = self.adapter.center_seconds
                priors = []
                for interval in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)):
                    basis, _ = positive_hrf_bank(centers, eeg_interval=interval)
                    prior = basis.mean(0)
                    priors.append(0.8 * prior + 0.2 / len(prior))
                self.register_buffer("multi_priors", torch.stack(priors), persistent=True)
            if architecture in {"r_region_conv", "r_region_mamba"}:
                self.region_embedding = nn.Parameter(torch.zeros(4, dim))
                self.region_score = nn.Linear(dim, 1)
                nn.init.normal_(self.region_embedding, std=0.02)
            if architecture == "r_gate_learned":
                self.gate_layer = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 1))
        self.residual = AnchoredResidualHead(dim, dim, dim, max_correction=2.0)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def load_teacher_state(self, state):
        self.teacher.load_state_dict(state, strict=True)
        freeze_teacher(self.teacher)

    def eeg_time_tokens(self, eeg):
        encoder = self.teacher.eeg_encoder
        with torch.no_grad():
            value = encoder.spatial(encoder.temporal(eeg.permute(0, 4, 1, 2, 3).contiguous()))
            value = encoder.separable[:-1](value).squeeze(2).squeeze(2)
            return torch.nn.functional.adaptive_avg_pool1d(value, 3).transpose(1, 2)

    def encode_fnirs(self, fnirs):
        segments = self.adapter(fnirs)
        batch = segments.shape[0]
        if self.architecture in {"r_region_conv", "r_region_mamba"}:
            # Four fixed geometric quadrants; each region has its own recurrent
            # state because regions are folded into the batch dimension.
            regions = (
                segments[:, :, :8, :8], segments[:, :, :8, 8:],
                segments[:, :, 8:, :8], segments[:, :, 8:, 8:],
            )
            regional = torch.stack(regions, 1)  # [B,R,L,H,W,T,C]
            flat = regional.permute(0, 1, 2, 6, 3, 4, 5).reshape(batch * 4 * 13, 2, 8, 8, 10)
            tokens = self.patch_encoder(flat).reshape(batch, 4, 13, -1)
            centers = self.adapter.center_seconds.to(fnirs.device, fnirs.dtype)
            tokens = tokens + self.time_embedding(centers)[None, None]
            mixed = self.mixer(tokens.reshape(batch * 4, 13, -1)).reshape_as(tokens)
            memory = tokens + mixed
            values, weights = [], []
            for region in range(4):
                value, item = self.readout(self._teacher_feature, memory[:, region])
                values.append(value); weights.append(item["weights"])
            region_values = torch.stack(values, 1) + self.region_embedding[None]
            region_weights = self.region_score(region_values).squeeze(-1).softmax(-1)
            feature = torch.einsum("br,brd->bd", region_weights, region_values)
            return feature, {"weights": torch.stack(weights, 1), "region_weights": region_weights, "memory": memory}
        flat = segments.permute(0, 1, 5, 2, 3, 4).reshape(batch * 13, 2, 16, 16, 10)
        content = self.patch_encoder(flat).reshape(batch, 13, -1)
        centers = self.adapter.center_seconds.to(fnirs.device, fnirs.dtype)
        tokens = content + self.time_embedding(centers).unsqueeze(0)
        if self.architecture in {"r_lagwrite", "r_lagwrite_uniform"}:
            if self.architecture == "r_lagwrite":
                prior = self.readout.bases.mean(0)[None].expand(batch, -1)
            else:
                prior = torch.ones(batch, 13, device=tokens.device, dtype=tokens.dtype)
            mixed = self.mixer(tokens, self._teacher_feature, prior)
        else:
            mixed = self.mixer(tokens)
        memory = tokens + mixed
        if self.architecture == "r_u":
            return memory.mean(1), {"weights": torch.full((batch, 13), 1/13, device=fnirs.device, dtype=fnirs.dtype), "memory": memory}
        if self.architecture in {"r_3q", "r_3h"}:
            query_tokens = self.eeg_time_tokens(self._eeg_grid)
            outputs, weights = [], []
            for index in range(3):
                override = None
                if self.architecture == "r_3h":
                    override = self.multi_priors[index][None].expand(batch, -1)
                value, item = self.readout(query_tokens[:, index], memory, prior_override=override)
                outputs.append(value); weights.append(item["weights"])
            feature = torch.stack(outputs, 1).mean(1)
            diagnostic = {"weights": torch.stack(weights, 1), "prior": self.multi_priors if self.architecture == "r_3h" else None}
        elif self.architecture == "r_h_wrong":
            prior = self.readout.bases.mean(0).flip(0)
            prior = 0.8 * prior + 0.2 / len(prior)
            feature, diagnostic = self.readout(self._teacher_feature, memory, prior_override=prior[None].expand(batch, -1))
        else:
            feature, diagnostic = self.readout(self._teacher_feature, memory)
        diagnostic["memory"] = memory
        return feature, diagnostic

    def forward(self, eeg, fnirs, *, intervention=None, return_diagnostics=False):
        del intervention
        teacher_logits, teacher_feature = frozen_eeg_forward(self.teacher, eeg)
        self._teacher_feature = teacher_feature
        self._eeg_grid = eeg
        if self.architecture == "r_e":
            complement = self.eeg_control(eeg.permute(0, 4, 1, 2, 3).contiguous())
            path_diagnostic = {}
        else:
            complement, path_diagnostic = self.encode_fnirs(fnirs)
        logits, delta = self.residual(teacher_logits, teacher_feature, complement)
        gate = None
        if self.architecture == "r_gate_fixed":
            gate = torch.full_like(delta, 0.5)
        elif self.architecture == "r_gate_learned":
            gate = torch.sigmoid(self.gate_layer(torch.cat((teacher_feature, complement), -1)))
        if gate is not None:
            used = gate * delta
            logits = teacher_logits.detach() + torch.cat((-0.5 * used, 0.5 * used), -1)
        diagnostics = {"teacher_logits": teacher_logits, "teacher_feature": teacher_feature, "complement": complement, "delta": delta, "gate": gate, **path_diagnostic} if return_diagnostics else {}
        return logits, diagnostics


def make_v4_fusion(name, **kwargs):
    return V4AnchoredFusion(name, **kwargs)
