import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.elu(self.norm(self.pointwise(self.depthwise(x))))


class LightweightEncoder(nn.Module):
    def __init__(self, in_channels, temporal_kernel, temporal_stride, channels=(8, 16)):
        super().__init__()
        k1, k2 = temporal_kernel
        s1, s2 = temporal_stride
        self.layers = nn.Sequential(
            DepthwiseSeparableConv3d(
                in_channels, channels[0], (3, 3, k1), (2, 2, s1), (1, 1, k1 // 2)
            ),
            DepthwiseSeparableConv3d(
                channels[0], channels[1], (3, 3, k2), (2, 2, s2), (1, 1, k2 // 2)
            ),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x):
        return self.layers(x).flatten(1)


class SpectralEEGEncoder(nn.Module):
    """Preserve task-relevant EEG band power and coarse spatial location."""

    def __init__(self, feature_dim=16, sample_rate=200.0, samples=600):
        super().__init__()
        frequencies = torch.fft.rfftfreq(samples, d=1.0 / sample_rate)
        bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
        masks = torch.stack([(frequencies >= low) & (frequencies < high) for low, high in bands])
        self.register_buffer("band_masks", masks, persistent=False)
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, feature_dim),
        )

    def forward(self, eeg):
        spectrum = torch.fft.rfft(eeg.squeeze(1), dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        band_power = torch.stack(
            [power[..., mask].mean(dim=-1) for mask in self.band_masks], dim=1
        )
        return self.spatial(torch.log(band_power.clamp_min(1e-8)))


class M0LightweightBaseline(nn.Module):
    """No-Mamba sanity baseline for validating the repaired data pipeline."""

    def __init__(self, feature_dim=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.eeg_encoder = LightweightEncoder(
            in_channels=1,
            temporal_kernel=(15, 5),
            temporal_stride=(6, 2),
            channels=(8, feature_dim),
        )
        self.fnirs_encoder = LightweightEncoder(
            in_channels=2,
            temporal_kernel=(5, 3),
            temporal_stride=(2, 2),
            channels=(8, feature_dim),
        )
        self.eeg_norm = nn.LayerNorm(feature_dim)
        self.fnirs_norm = nn.LayerNorm(feature_dim)
        self.modality_gate = nn.Linear(feature_dim * 2, 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, eeg, fnirs):
        batch_size = eeg.shape[0]
        eeg_feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())

        fnirs_lags = fnirs.shape[1]
        fnirs_flat = fnirs.reshape(batch_size * fnirs_lags, 16, 16, 30, 2)
        fnirs_flat = fnirs_flat.permute(0, 4, 1, 2, 3).contiguous()
        fnirs_tokens = self.fnirs_encoder(fnirs_flat).reshape(batch_size, fnirs_lags, -1)
        fnirs_feature = fnirs_tokens.mean(dim=1)

        eeg_feature = self.eeg_norm(eeg_feature)
        fnirs_feature = self.fnirs_norm(fnirs_feature)
        modality_weights = torch.softmax(
            self.modality_gate(torch.cat([eeg_feature, fnirs_feature], dim=-1)), dim=-1
        )
        fused = (
            modality_weights[:, :1] * eeg_feature
            + modality_weights[:, 1:] * fnirs_feature
        )
        logits = self.classifier(fused)
        diagnostics = {
            "modality_weights": modality_weights,
            "fnirs_tokens": fnirs_tokens,
        }
        return logits, diagnostics


class M0EEGOnly(nn.Module):
    """EEG-only diagnostic with the same encoder and classifier as M0."""

    def __init__(self, feature_dim=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.eeg_encoder = LightweightEncoder(
            in_channels=1,
            temporal_kernel=(15, 5),
            temporal_stride=(6, 2),
            channels=(8, feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, eeg, fnirs):
        del fnirs
        feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())
        logits = self.classifier(feature)
        weights = torch.zeros((eeg.shape[0], 2), device=eeg.device, dtype=eeg.dtype)
        weights[:, 0] = 1.0
        return logits, {"modality_weights": weights}


class M0FNIRSOnly(nn.Module):
    """fNIRS-only diagnostic using M0's encoder and uniform lag pooling."""

    def __init__(self, feature_dim=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.fnirs_encoder = LightweightEncoder(
            in_channels=2,
            temporal_kernel=(5, 3),
            temporal_stride=(2, 2),
            channels=(8, feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, eeg, fnirs):
        batch_size, lag_count = fnirs.shape[:2]
        flat = fnirs.reshape(batch_size * lag_count, 16, 16, 30, 2)
        tokens = self.fnirs_encoder(flat.permute(0, 4, 1, 2, 3).contiguous())
        feature = tokens.reshape(batch_size, lag_count, -1).mean(dim=1)
        logits = self.classifier(feature)
        weights = torch.zeros((batch_size, 2), device=fnirs.device, dtype=fnirs.dtype)
        weights[:, 1] = 1.0
        return logits, {"modality_weights": weights, "fnirs_tokens": tokens}


class M3SpectralEEGOnly(nn.Module):
    """Frequency-aware EEG diagnostic shared unchanged across all three tasks."""

    def __init__(self, feature_dim=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.eeg_encoder = SpectralEEGEncoder(feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, eeg, fnirs):
        del fnirs
        feature = self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())
        logits = self.classifier(feature)
        weights = torch.zeros((eeg.shape[0], 2), device=eeg.device, dtype=eeg.dtype)
        weights[:, 0] = 1.0
        return logits, {"modality_weights": weights}


class M4SpectralFusion(nn.Module):
    """Frequency-aware EEG branch with M0's lightweight fNIRS fusion."""

    def __init__(self, feature_dim=16, dropout=0.2, num_classes=2):
        super().__init__()
        self.eeg_encoder = SpectralEEGEncoder(feature_dim=feature_dim)
        self.fnirs_encoder = LightweightEncoder(
            in_channels=2,
            temporal_kernel=(5, 3),
            temporal_stride=(2, 2),
            channels=(8, feature_dim),
        )
        self.eeg_norm = nn.LayerNorm(feature_dim)
        self.fnirs_norm = nn.LayerNorm(feature_dim)
        self.modality_gate = nn.Linear(feature_dim * 2, 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, num_classes)
        )

    def forward(self, eeg, fnirs):
        batch_size, lag_count = fnirs.shape[:2]
        eeg_feature = self.eeg_norm(
            self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())
        )
        flat = fnirs.reshape(batch_size * lag_count, 16, 16, 30, 2)
        tokens = self.fnirs_encoder(flat.permute(0, 4, 1, 2, 3).contiguous())
        fnirs_feature = self.fnirs_norm(tokens.reshape(batch_size, lag_count, -1).mean(dim=1))
        weights = torch.softmax(
            self.modality_gate(torch.cat([eeg_feature, fnirs_feature], dim=-1)), dim=-1
        )
        fused = weights[:, :1] * eeg_feature + weights[:, 1:] * fnirs_feature
        return self.classifier(fused), {
            "modality_weights": weights,
            "fnirs_tokens": tokens.reshape(batch_size, lag_count, -1),
        }
