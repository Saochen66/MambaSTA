"""Fair EEG-query temporal-memory models for MambaSTA V3."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError as exc:
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None

from models.encoder_audit import EEGNetStyleEncoder
from models.lightweight_baseline import LightweightEncoder
from models.mamba_sta_v2 import PhysicalTimeEmbedding
from models.time_provenance_adapter import ChronologicalFNIRSAdapter


class QueryReadout(nn.Module):
    def __init__(self, dim=32, rank=8):
        super().__init__()
        self.qnorm = nn.LayerNorm(dim)
        self.knorm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, rank, bias=False)
        self.k = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = rank ** -0.5

    def forward(self, eeg, memory, *, query_override=None):
        if memory.ndim not in (3, 4):
            raise ValueError("memory must be [B,L,D] or [B,R,L,D]")
        batch, dim = memory.shape[0], memory.shape[-1]
        if eeg.shape != (batch, dim):
            raise ValueError("EEG shape mismatch")
        shape = memory.shape[1:-1]
        flat = memory.reshape(batch, -1, dim)
        source = eeg if query_override is None else query_override
        if source.shape != eeg.shape:
            raise ValueError("query override shape mismatch")
        query = self.q(self.qnorm(source))
        key = self.k(self.knorm(flat))
        score = torch.einsum("bd,btd->bt", query, key) * self.scale
        weights = score.softmax(dim=-1)
        readout = torch.einsum("bt,btd->bd", weights, self.v(flat))
        return readout, weights.reshape(batch, *shape)


class ZeroMixer(nn.Module):
    def forward(self, tokens):
        return torch.zeros_like(tokens)


class GatedTemporalConv(nn.Module):
    def __init__(self, dim=32, inner=64, kernel=3, dilations=(1, 2, 4)):
        super().__init__()
        self.inner = inner
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(inner, inner, kernel, groups=inner, dilation=d)
            for d in dilations
        ])
        self.pads = [(kernel - 1) * d for d in dilations]
        self.out_proj = nn.Linear(inner, dim, bias=False)

    def forward(self, tokens):
        x, gate = self.in_proj(tokens).chunk(2, dim=-1)
        x = x.transpose(1, 2)
        for conv, pad in zip(self.convs, self.pads):
            x = x + F.silu(conv(F.pad(x, (pad, 0))))
        return self.out_proj(x.transpose(1, 2) * F.silu(gate))


class PointwiseGatedControl(nn.Module):
    def __init__(self, dim=32, inner=64):
        super().__init__()
        self.inner = inner
        self.up = nn.Linear(dim, 2 * inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, tokens):
        x, gate = self.up(tokens).chunk(2, dim=-1)
        return self.down(F.silu(x) * F.silu(gate))


def active_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_matched_inner(kind, target, choices=(32, 40, 48, 56, 64, 72, 80, 96), dim=32):
    constructors = {
        "point": lambda width: PointwiseGatedControl(dim, width),
        "conv": lambda width: GatedTemporalConv(dim, width),
    }
    scored = [(abs(active_parameters(constructors[kind](width)) - target), width) for width in choices]
    return min(scored)[1]


class QueryMemoryHead(nn.Module):
    def __init__(self, mixer, dim=32, rank=8, dropout=0.2):
        super().__init__()
        self.mixer = mixer
        self.pre = nn.LayerNorm(dim)
        self.post = nn.LayerNorm(dim)
        self.readout = QueryReadout(dim, rank)
        self.eeg_norm = nn.LayerNorm(dim)
        self.fnirs_norm = nn.LayerNorm(dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(dim), nn.Linear(dim, 2),
        )

    def forward(self, eeg, tokens, *, query_override=None):
        normalized = self.pre(tokens)
        mixed = self.mixer(normalized)
        memory = self.post(tokens + mixed)
        fnirs, weights = self.readout(eeg, memory, query_override=query_override)
        logits = self.fusion(torch.cat((self.eeg_norm(eeg), self.fnirs_norm(fnirs)), dim=-1))
        return {
            "logits": logits, "readout_weights": weights,
            "memory": memory, "mixer_output": mixed,
        }


class EEG32(nn.Module):
    def __init__(self, dim=32, dropout=0.2):
        super().__init__()
        self.eeg_encoder = EEGNetStyleEncoder(dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(dim), nn.Linear(dim, 2),
        )

    def forward(self, eeg, fnirs, *, intervention=None, return_diagnostics=False):
        del fnirs, intervention
        feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())
        logits = self.classifier(feature)
        diagnostics = {"eeg_feature": feature} if return_diagnostics else {}
        return logits, diagnostics


class QueryMemoryModel(nn.Module):
    VALID = {"q_raw", "q_point", "q_conv", "q_mamba"}

    def __init__(
        self, architecture, dim=32, state=8, conv_kernel=4, expand=2,
        query_rank=8, dropout=0.2,
    ):
        super().__init__()
        if architecture not in self.VALID:
            raise ValueError(f"Unknown V3 query model: {architecture}")
        if Mamba is None:
            raise ImportError("Q-Mamba controls require mamba_ssm") from MAMBA_IMPORT_ERROR
        self.architecture = architecture
        self.adapter = ChronologicalFNIRSAdapter()
        self.eeg_encoder = EEGNetStyleEncoder(dim)
        self.patch_encoder = LightweightEncoder(2, (5, 3), (2, 1), (8, dim))
        self.time_embedding = PhysicalTimeEmbedding(dim)
        reference = Mamba(
            d_model=dim, d_state=state, d_conv=conv_kernel,
            expand=expand, use_fast_path=False,
        )
        target = active_parameters(reference)
        if architecture == "q_raw":
            mixer = ZeroMixer()
            self.mixer_inner = 0
        elif architecture == "q_point":
            self.mixer_inner = parameter_matched_inner("point", target, dim=dim)
            mixer = PointwiseGatedControl(dim, self.mixer_inner)
        elif architecture == "q_conv":
            self.mixer_inner = parameter_matched_inner("conv", target, dim=dim)
            mixer = GatedTemporalConv(dim, self.mixer_inner)
        else:
            mixer = reference
            self.mixer_inner = getattr(reference, "d_inner", dim * expand)
        self.reference_mamba_parameters = target
        self.head = QueryMemoryHead(mixer, dim=dim, rank=query_rank, dropout=dropout)

    def encode_tokens(self, fnirs, intervention=None):
        intervention = intervention or {}
        segments = self.adapter(fnirs)
        if "physical_delete" in intervention:
            segments = self.adapter.replace_physical_segments(
                segments,
                intervention["physical_delete"],
                intervention["physical_replacement"],
            )
        batch = segments.shape[0]
        patches = segments.permute(0, 1, 5, 2, 3, 4).reshape(
            batch * 13, 2, 16, 16, 10
        )
        content = self.patch_encoder(patches).reshape(batch, 13, -1)
        centers = self.adapter.center_seconds.to(device=fnirs.device, dtype=fnirs.dtype)
        return content + self.time_embedding(centers).unsqueeze(0), segments

    def forward(self, eeg, fnirs, *, intervention=None, return_diagnostics=False):
        intervention = intervention or {}
        eeg_feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())
        tokens, segments = self.encode_tokens(fnirs, intervention)
        result = self.head(
            eeg_feature, tokens,
            query_override=intervention.get("query_override"),
        )
        diagnostics = {
            "readout_weights": result["readout_weights"],
            "mixer_output": result["mixer_output"],
            "memory": result["memory"],
            "eeg_feature": eeg_feature,
        } if return_diagnostics else {}
        if return_diagnostics:
            diagnostics["physical_segments"] = segments
        return result["logits"], diagnostics


def make_mambasta_v3(name, **kwargs):
    if name == "e32":
        allowed = {key: value for key, value in kwargs.items() if key in {"dim", "dropout"}}
        return EEG32(**allowed)
    return QueryMemoryModel(name, **kwargs)
