"""MambaSTA-v2 chronological and conditional-state development scaffold."""

import torch
import torch.nn as nn

from models.conditional_selective_mamba import ConditionalSelectiveMamba
from models.encoder_audit import EEGNetStyleEncoder
from models.hb_mamba_lite import MLPSequenceMixer
from models.lightweight_baseline import LightweightEncoder
from models.time_provenance_adapter import ChronologicalFNIRSAdapter


class PhysicalTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model)
        )

    def forward(self, center_seconds):
        scale = center_seconds.abs().max().clamp_min(1.0)
        return self.network((center_seconds / scale).unsqueeze(-1))


class MambaSTAV2(nn.Module):
    """EEGNet anchor plus 13 unique physical fNIRS patches.

    Architectures:
      - ``a0_chrono_mamba``: official Mamba scan, no state condition.
      - ``a0_chrono_mlp``: parameter-near MLP mixer.
      - ``a1_delta_cond``: low-rank EEG-token interaction changes Delta.
      - ``a1_c_cond``: identical interaction changes C readout only.
      - ``chrono_fnirs_only``: strong chronological fNIRS-only baseline.
    """

    def __init__(
        self,
        architecture="a0_chrono_mamba",
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        condition_rank=4,
        condition_scale=0.5,
        dropout=0.2,
        verify_duplicates=False,
    ):
        super().__init__()
        valid = {
            "a0_chrono_mamba", "a0_chrono_mlp", "a1_delta_cond",
            "a1_c_cond", "chrono_fnirs_only",
        }
        if architecture not in valid:
            raise ValueError(f"Unknown MambaSTA-v2 architecture: {architecture}")
        self.architecture = architecture
        self.d_model = d_model
        self.adapter = ChronologicalFNIRSAdapter(verify_duplicates=verify_duplicates)
        self.eeg_encoder = None if architecture == "chrono_fnirs_only" else EEGNetStyleEncoder(d_model)
        self.patch_encoder = LightweightEncoder(2, (5, 3), (2, 1), (8, d_model))
        self.time_embedding = PhysicalTimeEmbedding(d_model)
        self.input_norm = nn.LayerNorm(d_model)
        if architecture == "a0_chrono_mlp":
            self.mixer = MLPSequenceMixer(d_model, sequence_length=13, hidden=d_model * 2)
            self.condition_mode = "none"
            self.mamba_version = None
        else:
            condition_mode = {
                "a0_chrono_mamba": "none",
                "a1_delta_cond": "delta",
                "a1_c_cond": "c",
                "chrono_fnirs_only": "none",
            }[architecture]
            self.mixer = ConditionalSelectiveMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                condition_rank=condition_rank, condition_scale=condition_scale,
                condition_mode=condition_mode,
            )
            self.condition_mode = condition_mode
            self.mamba_version = self.mixer.mamba_version
        self.output_norm = nn.LayerNorm(d_model)
        self.relevance = nn.Linear(d_model, 1)
        self.fnirs_norm = nn.LayerNorm(d_model)
        if architecture == "chrono_fnirs_only":
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2)
            )
        else:
            self.eeg_norm = nn.LayerNorm(d_model)
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout)
            )
            self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def _encode_tokens(self, fnirs, intervention):
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
        labels = centers
        if "time_label_permutation" in intervention:
            labels = labels[intervention["time_label_permutation"]]
        tokens = content + self.time_embedding(labels).unsqueeze(0)
        if "scan_order_permutation" in intervention:
            # Content and its already attached physical-time metadata move together.
            tokens = tokens[:, intervention["scan_order_permutation"]]
        return tokens, segments

    def forward(
        self,
        eeg,
        fnirs,
        *,
        time_metadata=None,
        intervention=None,
        return_diagnostics=False,
    ):
        del time_metadata  # Physical coordinates are frozen by the audited adapter.
        intervention = intervention or {}
        tokens, physical_segments = self._encode_tokens(fnirs, intervention)
        normalized = self.input_norm(tokens)
        eeg_feature = None
        if self.eeg_encoder is not None:
            eeg_feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())

        if isinstance(self.mixer, ConditionalSelectiveMamba):
            condition = intervention.get("e_condition_override", eeg_feature)
            mixed_output = self.mixer(
                normalized,
                condition,
                condition_off=bool(intervention.get("condition_off", False)),
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                mixed_delta, mixer_diagnostics = mixed_output
            else:
                mixed_delta = mixed_output
                mixer_diagnostics = {}
        else:
            mixed_delta = self.mixer(normalized)
            mixer_diagnostics = {}
        mixed = self.output_norm(tokens + mixed_delta)
        token_relevance = torch.softmax(self.relevance(mixed).squeeze(-1), dim=1)
        fnirs_feature = torch.sum(token_relevance.unsqueeze(-1) * mixed, dim=1)
        fnirs_feature = self.fnirs_norm(fnirs_feature)

        if self.architecture == "chrono_fnirs_only":
            logits = self.classifier(fnirs_feature)
        else:
            eeg_feature = self.eeg_norm(eeg_feature)
            logits = self.classifier(self.fusion(torch.cat((eeg_feature, fnirs_feature), dim=-1)))
        diagnostics = {
            "physical_center_seconds": self.adapter.center_seconds,
            "token_relevance": token_relevance,
            "condition_mode": self.condition_mode,
            "modality_weights": None,
            **mixer_diagnostics,
        }
        if return_diagnostics:
            diagnostics["physical_segments"] = physical_segments
            diagnostics["mixed_tokens"] = mixed
        return logits, diagnostics


def make_mambasta_v2(name, **kwargs):
    return MambaSTAV2(architecture=name, **kwargs)
