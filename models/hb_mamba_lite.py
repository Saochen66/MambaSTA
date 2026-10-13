"""HB-Mamba-Lite and parameter-matched sequence-mixer controls."""

import torch
import torch.nn as nn

from models.architecture_tracks import DelayEmbedding, LAG_SECONDS
from models.encoder_audit import EEGNetStyleEncoder
from models.lightweight_baseline import LightweightEncoder

try:
    import mamba_ssm
    from mamba_ssm import Mamba
except ImportError as exc:
    mamba_ssm = None
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


class GatedConvSequenceMixer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        expanded = d_model * 3
        self.in_projection = nn.Linear(d_model, expanded * 2)
        self.depthwise = nn.Conv1d(
            expanded * 2,
            expanded * 2,
            kernel_size=3,
            padding=1,
            groups=expanded * 2,
            bias=False,
        )
        self.out_projection = nn.Linear(expanded, d_model)
        self.post = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, x):
        projected = self.depthwise(self.in_projection(x).transpose(1, 2)).transpose(1, 2)
        value, gate = projected.chunk(2, dim=-1)
        return self.out_projection(value * torch.sigmoid(gate)) + self.post(x)


class MLPSequenceMixer(nn.Module):
    def __init__(self, d_model, sequence_length=33, hidden=None):
        super().__init__()
        hidden = hidden or d_model * 2
        self.token_mlp = nn.Sequential(
            nn.Linear(sequence_length, hidden),
            nn.GELU(),
            nn.Linear(hidden, sequence_length),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        token_mixed = self.token_mlp(x.transpose(1, 2)).transpose(1, 2)
        return token_mixed + self.channel_mlp(x)


class LightAttentionSequenceMixer(nn.Module):
    def __init__(self, d_model, heads=4, hidden=None):
        super().__init__()
        hidden = hidden or d_model * 2
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        attended, _ = self.attention(x, x, x, need_weights=False)
        return attended + self.feed_forward(x)


def make_mixer(name, d_model, d_state, d_conv):
    if name == "mamba":
        if Mamba is None:
            raise ImportError("HB-Mamba-Lite requires official mamba_ssm") from MAMBA_IMPORT_ERROR
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)
    if name == "gatedconv":
        return GatedConvSequenceMixer(d_model)
    if name == "mlp":
        return MLPSequenceMixer(d_model)
    if name == "attention":
        return LightAttentionSequenceMixer(d_model)
    raise ValueError(f"Unknown sequence mixer: {name}")


class HBMambaLite(nn.Module):
    """EEGNet anchor + 33 ordered fNIRS patch tokens + isolated sequence mixer."""

    def __init__(
        self,
        mixer="mamba",
        d_model=32,
        d_state=8,
        d_conv=4,
        dropout=0.2,
        fusion="concat",
        patch_width=8,
    ):
        super().__init__()
        if fusion not in {"concat", "weighted", "eeg_anchor"}:
            raise ValueError(f"Unknown fusion: {fusion}")
        self.mixer_name = mixer
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.fusion_name = fusion
        self.patch_width = patch_width
        self.eeg_encoder = EEGNetStyleEncoder(d_model)
        self.patch_encoder = LightweightEncoder(
            2, (5, 3), (2, 1), (patch_width, d_model)
        )
        self.delay_embedding = DelayEmbedding(d_model)
        self.local_time_embedding = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.local_time_embedding, std=0.02)
        self.input_norm = nn.LayerNorm(d_model)
        self.lag_mixer = make_mixer(mixer, d_model, d_state, d_conv)
        self.output_norm = nn.LayerNorm(d_model)
        self.relevance = nn.Linear(d_model, 1)
        self.eeg_norm = nn.LayerNorm(d_model)
        self.fnirs_norm = nn.LayerNorm(d_model)
        if fusion == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion == "weighted":
            self.modality_gate = nn.Linear(d_model * 2, 2)
            self.fusion_dropout = nn.Dropout(dropout)
        else:
            self.fnirs_projection = nn.Linear(d_model, d_model)
            self.residual_gate = nn.Linear(d_model * 2, 1)
            self.fusion_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.mamba_version = getattr(mamba_ssm, "__version__", None) if mixer == "mamba" else None

    def forward(self, eeg, fnirs, intervention=None):
        intervention = intervention or {}
        if eeg.ndim != 5 or eeg.shape[1:] != (16, 16, 600, 1):
            raise ValueError(f"Expected EEG [B,16,16,600,1], got {tuple(eeg.shape)}")
        if fnirs.ndim != 6 or fnirs.shape[1:] != (11, 16, 16, 30, 2):
            raise ValueError(f"Expected fNIRS [B,11,16,16,30,2], got {tuple(fnirs.shape)}")
        batch = eeg.shape[0]
        eeg_feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())

        if "lag_permutation" in intervention:
            fnirs = fnirs[:, intervention["lag_permutation"]]
        patches = fnirs.reshape(batch, 11, 16, 16, 3, 10, 2)
        patches = patches.permute(0, 1, 4, 6, 2, 3, 5).reshape(batch * 33, 2, 16, 16, 10)
        tokens = self.patch_encoder(patches).reshape(batch, 11, 3, self.d_model)
        tokens = tokens + self.local_time_embedding.view(1, 1, 3, self.d_model)
        tokens = tokens.reshape(batch, 33, self.d_model)
        if "token_permutation" in intervention:
            tokens = tokens[:, intervention["token_permutation"]]

        delays = torch.tensor(LAG_SECONDS, device=fnirs.device, dtype=fnirs.dtype)
        if "delay_permutation" in intervention:
            delays = delays[intervention["delay_permutation"]]
        tokens = tokens + self.delay_embedding(delays.repeat_interleave(3)).unsqueeze(0)
        normalized = self.input_norm(tokens)
        mixed = self.output_norm(tokens + self.lag_mixer(normalized))
        token_relevance = torch.softmax(self.relevance(mixed).squeeze(-1), dim=1)
        occlusion = intervention.get("occlusion")
        if occlusion in {"top", "bottom"}:
            k = int(intervention.get("occlusion_k", 1))
            if not 1 <= k < token_relevance.shape[1]:
                raise ValueError(f"occlusion_k must be in [1, 32], got {k}")
            indices = torch.topk(
                token_relevance, k, dim=1, largest=occlusion == "top"
            ).indices
            keep = torch.ones_like(token_relevance)
            keep.scatter_(1, indices, 0.0)
            mixed = mixed * keep.unsqueeze(-1)
            token_relevance = token_relevance * keep
            token_relevance = token_relevance / token_relevance.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
        if intervention.get("uniform_relevance", False):
            token_relevance = torch.full_like(token_relevance, 1.0 / 33.0)
        fnirs_feature = torch.sum(token_relevance.unsqueeze(-1) * mixed, dim=1)
        lag_relevance = token_relevance.reshape(batch, 11, 3).sum(dim=2)

        eeg_feature = self.eeg_norm(eeg_feature)
        fnirs_feature = self.fnirs_norm(fnirs_feature)
        paired = torch.cat((eeg_feature, fnirs_feature), dim=-1)
        if self.fusion_name == "concat":
            fused = self.fusion(paired)
            modality_weights = torch.full(
                (batch, 2), 0.5, device=eeg.device, dtype=eeg.dtype
            )
        elif self.fusion_name == "weighted":
            modality_weights = torch.softmax(self.modality_gate(paired), dim=-1)
            fused = (
                modality_weights[:, :1] * eeg_feature
                + modality_weights[:, 1:] * fnirs_feature
            )
            fused = self.fusion_dropout(fused)
        else:
            gate = torch.sigmoid(self.residual_gate(paired))
            fused = eeg_feature + gate * self.fnirs_projection(fnirs_feature)
            fused = self.fusion_dropout(fused)
            modality_weights = torch.cat(
                (1.0 / (1.0 + gate), gate / (1.0 + gate)), dim=-1
            )
        return self.classifier(fused), {
            "modality_weights": modality_weights,
            "lag_relevance": lag_relevance,
            "token_relevance": token_relevance,
        }


def make_hb_mamba_lite(name, **kwargs):
    mixer = {
        "hb_mamba": "mamba",
        "hb_gatedconv": "gatedconv",
        "hb_mlp": "mlp",
        "hb_attention": "attention",
    }[name]
    return HBMambaLite(mixer=mixer, **kwargs)
