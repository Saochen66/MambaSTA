"""Lightweight STA bridge controls on the audited 13-segment timeline."""

import torch
import torch.nn as nn

from models.conditional_selective_mamba import ConditionalSelectiveMamba
from models.encoder_audit import EEGNetStyleEncoder
from models.lightweight_baseline import LightweightEncoder
from models.mamba_sta_v2 import PhysicalTimeEmbedding
from models.time_provenance_adapter import ChronologicalFNIRSAdapter


# Locations used by preprocessing.pipeline_lit for the 28 observed EEG sensors.
EEG_GRID_LOCATIONS = (
    (0, 6), (2, 5), (2, 8), (3, 7), (5, 2), (5, 6), (7, 1),
    (7, 4), (7, 8), (9, 2), (9, 6), (11, 2), (11, 5), (11, 8),
    (13, 8), (14, 6), (0, 10), (2, 11), (3, 9), (5, 10),
    (5, 14), (7, 12), (7, 15), (9, 10), (9, 14), (11, 11),
    (11, 14), (14, 10),
)


class LiteSTAAlignment(nn.Module):
    """One spatial alignment path plus attention or conditional-Mamba time path.

    ``b0_lite_sta_attention`` uses an EEG-query cross-attention readout.
    ``b1_lite_sta_cond_mamba`` replaces that time path with Delta-conditioned
    official Mamba while leaving the encoder and spatial path unchanged.
    """

    def __init__(
        self,
        architecture="b0_lite_sta_attention",
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        condition_rank=4,
        condition_scale=0.5,
        dropout=0.2,
    ):
        super().__init__()
        if architecture not in {"b0_lite_sta_attention", "b1_lite_sta_cond_mamba"}:
            raise ValueError(f"Unknown Lite-STA architecture: {architecture}")
        self.architecture = architecture
        self.d_model = d_model
        self.adapter = ChronologicalFNIRSAdapter()
        self.eeg_encoder = EEGNetStyleEncoder(d_model)
        self.patch_encoder = LightweightEncoder(2, (5, 3), (2, 1), (8, d_model))
        self.time_embedding = PhysicalTimeEmbedding(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # A single residual spatial path.  The final projection is zero at
        # initialization, so the gated EEG branch initially equals EEGNet.
        self.fnirs_spatial = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 1, 1, bias=True),
        )
        nn.init.zeros_(self.fnirs_spatial[-1].weight)
        nn.init.zeros_(self.fnirs_spatial[-1].bias)
        mask = torch.zeros(1, 1, 16, 16, 1)
        for row, column in EEG_GRID_LOCATIONS:
            mask[0, 0, row, column, 0] = 1.0
        self.register_buffer("eeg_observed_mask", mask, persistent=True)

        if architecture == "b0_lite_sta_attention":
            self.temporal_path = nn.MultiheadAttention(
                d_model, num_heads=1, dropout=dropout, batch_first=True
            )
            self.attention_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.output_norm = nn.LayerNorm(d_model)
            self.relevance = None
            self.condition_mode = "attention_readout"
        else:
            self.temporal_path = ConditionalSelectiveMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                condition_rank=condition_rank,
                condition_scale=condition_scale,
                condition_mode="delta",
            )
            self.output_norm = nn.LayerNorm(d_model)
            self.relevance = nn.Linear(d_model, 1)
            self.condition_mode = "delta"
        self.eeg_norm = nn.LayerNorm(d_model)
        self.fnirs_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def _tokens_and_spatial_map(self, fnirs, intervention=None):
        intervention = intervention or {}
        segments = self.adapter(fnirs)
        if "physical_delete" in intervention:
            if "physical_replacement" not in intervention:
                raise ValueError("physical_delete requires physical_replacement")
            segments = self.adapter.replace_physical_segments(
                segments,
                intervention["physical_delete"],
                intervention["physical_replacement"],
            )
        batch = segments.shape[0]
        patches = segments.permute(0, 1, 5, 2, 3, 4).reshape(
            batch * 13, 2, 16, 16, 10
        )
        content = self.patch_encoder(patches).reshape(batch, 13, self.d_model)
        centers = self.adapter.center_seconds.to(device=fnirs.device, dtype=fnirs.dtype)
        tokens = content + self.time_embedding(centers).unsqueeze(0)
        # Mean only forms the conditioning map; it does not replace temporal tokens.
        spatial_source = segments.mean(dim=(1, 4)).permute(0, 3, 1, 2).contiguous()
        spatial_gate = self.fnirs_spatial(spatial_source)
        return tokens, segments, spatial_gate

    def _gated_eeg(self, eeg, spatial_gate, spatial_off=False, spatial_override=None):
        encoder = self.eeg_encoder
        temporal = encoder.temporal(eeg.permute(0, 4, 1, 2, 3).contiguous())
        if spatial_override is not None:
            spatial_gate = spatial_override
        if spatial_off:
            spatial_gate = torch.zeros_like(spatial_gate)
        gate = 1.0 + 0.1 * torch.tanh(spatial_gate).unsqueeze(-1) * self.eeg_observed_mask
        gated = temporal * gate
        return encoder.separable(encoder.spatial(gated)).flatten(1), gate

    def forward(self, eeg, fnirs, *, time_metadata=None, intervention=None, return_diagnostics=False):
        del time_metadata
        intervention = intervention or {}
        tokens, segments, spatial_gate = self._tokens_and_spatial_map(fnirs, intervention)
        eeg_feature, applied_gate = self._gated_eeg(
            eeg,
            spatial_gate,
            spatial_off=bool(intervention.get("spatial_path_off", False)),
            spatial_override=intervention.get("spatial_gate_override"),
        )
        normalized = self.input_norm(tokens)
        if self.architecture == "b0_lite_sta_attention":
            attended, attention = self.temporal_path(
                eeg_feature.unsqueeze(1), normalized, normalized,
                need_weights=True, average_attn_weights=True,
            )
            attended = attended.squeeze(1)
            fnirs_feature = self.output_norm(attended + self.attention_ffn(attended))
            mixer_diagnostics = {"attention_weights": attention.squeeze(1)}
        else:
            condition = intervention.get("e_condition_override", eeg_feature)
            mixed_output = self.temporal_path(
                normalized,
                condition,
                condition_off=bool(intervention.get("condition_off", False)),
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                mixed_delta, mixer_diagnostics = mixed_output
            else:
                mixed_delta, mixer_diagnostics = mixed_output, {}
            mixed = self.output_norm(tokens + mixed_delta)
            attention = torch.softmax(self.relevance(mixed).squeeze(-1), dim=1)
            fnirs_feature = torch.sum(attention.unsqueeze(-1) * mixed, dim=1)
            mixer_diagnostics["attention_weights"] = attention
        eeg_feature = self.eeg_norm(eeg_feature)
        fnirs_feature = self.fnirs_norm(fnirs_feature)
        logits = self.classifier(self.fusion(torch.cat((eeg_feature, fnirs_feature), dim=-1)))
        diagnostics = {
            "condition_mode": self.condition_mode,
            "modality_weights": None,
            "spatial_gate_abs_mean": (applied_gate - 1.0).abs().mean().detach(),
            **mixer_diagnostics,
        }
        if return_diagnostics:
            diagnostics["physical_segments"] = segments
            diagnostics["spatial_gate"] = applied_gate
        return logits, diagnostics


def make_lite_sta_alignment(name, **kwargs):
    return LiteSTAAlignment(architecture=name, **kwargs)
