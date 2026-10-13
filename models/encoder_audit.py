"""Controlled lightweight encoder variants for the Phase-2 audit."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightweight_baseline import LightweightEncoder


class StandardConv3dEncoder(nn.Module):
    """Standard-Conv counterpart with the same output widths as E0/F0."""

    def __init__(self, in_channels, temporal_kernel, temporal_stride, channels=(8, 16)):
        super().__init__()
        k1, k2 = temporal_kernel
        s1, s2 = temporal_stride
        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels, channels[0], (3, 3, k1),
                stride=(2, 2, s1), padding=(1, 1, k1 // 2), bias=False,
            ),
            nn.BatchNorm3d(channels[0]),
            nn.ELU(),
            nn.Conv3d(
                channels[0], channels[1], (3, 3, k2),
                stride=(2, 2, s2), padding=(1, 1, k2 // 2), bias=False,
            ),
            nn.BatchNorm3d(channels[1]),
            nn.ELU(),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        return self.layers(x).flatten(1)


class WiderDSConv3dEncoder(nn.Module):
    """Wider DSConv features projected to the fixed 16-D comparison interface."""

    def __init__(self, in_channels, temporal_kernel, temporal_stride):
        super().__init__()
        self.encoder = LightweightEncoder(
            in_channels=in_channels,
            temporal_kernel=temporal_kernel,
            temporal_stride=temporal_stride,
            channels=(16, 32),
        )
        self.projection = nn.Linear(32, 16)

    def forward(self, x):
        return self.projection(self.encoder(x))


class EEGNetStyleEncoder(nn.Module):
    """Temporal -> depthwise full-grid spatial -> separable temporal encoder."""

    def __init__(self, feature_dim=16):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv3d(1, 8, (1, 1, 31), padding=(0, 0, 15), bias=False),
            nn.BatchNorm3d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(8, 16, (16, 16, 1), groups=8, bias=False),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.AvgPool3d((1, 1, 4)),
        )
        self.separable = nn.Sequential(
            nn.Conv3d(16, 16, (1, 1, 15), padding=(0, 0, 7), groups=16, bias=False),
            nn.Conv3d(16, feature_dim, 1, bias=False),
            nn.BatchNorm3d(feature_dim),
            nn.ELU(),
            nn.AvgPool3d((1, 1, 4)),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        return self.separable(self.spatial(self.temporal(x))).flatten(1)


class HypersignalFNIRSEncoder(nn.Module):
    """LoBrA-inspired HbO/HbR/HbT/HbD input ablation."""

    def __init__(self):
        super().__init__()
        self.encoder = LightweightEncoder(
            in_channels=4,
            temporal_kernel=(5, 3),
            temporal_stride=(2, 2),
            channels=(8, 16),
        )

    def forward(self, x):
        hbo, hbr = x[:, :1], x[:, 1:2]
        augmented = torch.cat((hbo, hbr, hbo + hbr, hbo - hbr), dim=1)
        return self.encoder(augmented)


class RegionFNIRSEncoder(nn.Module):
    """Four fixed non-overlapping spatial quadrants with learned region pooling."""

    def __init__(self, feature_dim=16):
        super().__init__()
        self.local_encoder = LightweightEncoder(
            in_channels=2,
            temporal_kernel=(5, 3),
            temporal_stride=(2, 2),
            channels=(8, feature_dim),
        )
        self.region_embedding = nn.Parameter(torch.zeros(4, feature_dim))
        self.region_score = nn.Linear(feature_dim, 1)
        nn.init.normal_(self.region_embedding, std=0.02)

    def forward(self, x):
        regions = (
            x[:, :, :8, :8], x[:, :, :8, 8:],
            x[:, :, 8:, :8], x[:, :, 8:, 8:],
        )
        tokens = torch.stack(
            [self.local_encoder(region) for region in regions], dim=1
        )
        tokens = tokens + self.region_embedding.unsqueeze(0)
        weights = F.softmax(self.region_score(tokens).squeeze(-1), dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class EncoderAuditModel(nn.Module):
    """Unimodal wrapper that holds classifier and evaluation protocol fixed."""

    def __init__(self, modality, variant, feature_dim=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.modality = modality
        if modality == "eeg":
            encoders = {
                "dsconv": lambda: LightweightEncoder(1, (15, 5), (6, 2), (8, feature_dim)),
                "conv": lambda: StandardConv3dEncoder(1, (15, 5), (6, 2), (8, feature_dim)),
                "eegnet": lambda: EEGNetStyleEncoder(feature_dim),
                "dsconv_wide": lambda: WiderDSConv3dEncoder(1, (15, 5), (6, 2)),
            }
        elif modality == "fnirs":
            encoders = {
                "dsconv": lambda: LightweightEncoder(2, (5, 3), (2, 2), (8, feature_dim)),
                "conv": lambda: StandardConv3dEncoder(2, (5, 3), (2, 2), (8, feature_dim)),
                "dsconv_wide": lambda: WiderDSConv3dEncoder(2, (5, 3), (2, 2)),
                "hypersignal": HypersignalFNIRSEncoder,
                "region": lambda: RegionFNIRSEncoder(feature_dim),
            }
        else:
            raise ValueError(f"Unknown modality: {modality}")
        if variant not in encoders:
            raise ValueError(f"Unknown {modality} encoder variant: {variant}")
        self.encoder = encoders[variant]()
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, eeg, fnirs):
        batch_size = eeg.shape[0]
        weights = torch.zeros((batch_size, 2), device=eeg.device, dtype=eeg.dtype)
        if self.modality == "eeg":
            feature = self.encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())
            weights[:, 0] = 1.0
        else:
            lag_count = fnirs.shape[1]
            flattened = fnirs.reshape(batch_size * lag_count, 16, 16, 30, 2)
            flattened = flattened.permute(0, 4, 1, 2, 3).contiguous()
            tokens = self.encoder(flattened).reshape(batch_size, lag_count, -1)
            feature = tokens.mean(dim=1)
            weights[:, 1] = 1.0
        return self.classifier(feature), {"modality_weights": weights}


def make_encoder_audit_model(name):
    definitions = {
        "eeg_dsconv": ("eeg", "dsconv"),
        "eeg_conv": ("eeg", "conv"),
        "eeg_eegnet": ("eeg", "eegnet"),
        "eeg_dsconv_wide": ("eeg", "dsconv_wide"),
        "fnirs_dsconv": ("fnirs", "dsconv"),
        "fnirs_conv": ("fnirs", "conv"),
        "fnirs_dsconv_wide": ("fnirs", "dsconv_wide"),
        "fnirs_hypersignal": ("fnirs", "hypersignal"),
        "fnirs_region": ("fnirs", "region"),
    }
    modality, variant = definitions[name]
    return EncoderAuditModel(modality, variant)
