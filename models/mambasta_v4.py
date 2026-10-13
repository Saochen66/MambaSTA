"""Mechanism-first components for MambaSTA V4 multimodal alignment."""

from __future__ import annotations

import math

import torch
from torch import nn


def positive_hrf_bank(
    centers,
    eeg_interval=(0.0, 3.0),
    peaks=(3.0, 5.0, 7.0),
    shape=4.0,
    samples=24,
    tail_seconds=12.0,
):
    """Integrate positive gamma response bases over one EEG support interval."""
    if shape <= 1 or samples < 2:
        raise ValueError("shape > 1 and samples >= 2 are required")
    begin, end = map(float, eeg_interval)
    if end <= begin or not peaks or min(peaks) <= 0:
        raise ValueError("invalid interval or peak")
    centers = torch.as_tensor(centers, dtype=torch.float32)
    source = begin + (torch.arange(samples, device=centers.device) + 0.5) * (
        end - begin
    ) / samples
    lag = centers[:, None] - source[None, :]
    valid = (lag > 0) & (lag <= tail_seconds)
    positive_lag = lag.clamp_min(1e-6)
    profiles = []
    for peak in peaks:
        scale = float(peak) / (shape - 1.0)
        log_pdf = (
            (shape - 1.0) * positive_lag.log()
            - positive_lag / scale
            - shape * math.log(scale)
            - math.lgamma(shape)
        )
        curve = (log_pdf.exp() * valid).mean(-1) * (end - begin)
        if curve.sum() <= 0:
            raise ValueError("no observed support for this basis")
        profiles.append(curve)
    raw = torch.stack(profiles)
    return raw / raw.sum(-1, keepdim=True), raw


class DelayPriorReadout(nn.Module):
    """Content readout with an optional weak prior on physical fNIRS time."""

    def __init__(
        self, centers, dim=32, rank=8, beta=0.5, uniform_floor=0.2, adaptive=False
    ):
        super().__init__()
        if not 0 < uniform_floor <= 1 or not 0 <= beta <= 2:
            raise ValueError("invalid prior settings")
        bases, raw = positive_hrf_bank(centers)
        self.register_buffer("bases", bases)
        self.register_buffer("basis_raw_profiles", raw)
        self.beta = float(beta)
        self.floor = float(uniform_floor)
        self.eeg_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, rank, bias=False)
        self.key = nn.Linear(dim, rank, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = rank**-0.5
        self.mix = nn.Linear(dim, bases.shape[0]) if adaptive else None
        if self.mix is not None:
            nn.init.zeros_(self.mix.weight)
            nn.init.zeros_(self.mix.bias)

    def forward(self, eeg, tokens, *, query_override=None, prior_override=None):
        if tokens.ndim != 3 or eeg.shape != (tokens.shape[0], tokens.shape[2]):
            raise ValueError("expected eeg [B,D] and tokens [B,L,D]")
        source = eeg if query_override is None else query_override
        source = self.eeg_norm(source)
        score = torch.einsum(
            "bd,btd->bt", self.query(source), self.key(self.token_norm(tokens))
        ) * self.scale
        if self.mix is None:
            prior = self.bases.mean(0)[None].expand(eeg.shape[0], -1)
        else:
            prior = self.mix(source).softmax(-1) @ self.bases
        prior = (1.0 - self.floor) * prior + self.floor / tokens.shape[1]
        if prior_override is not None:
            if prior_override.shape != prior.shape or (prior_override <= 0).any():
                raise ValueError("override must be positive and shaped [B,L]")
            prior = prior_override / prior_override.sum(-1, keepdim=True)
        weights = (score + self.beta * prior.log()).softmax(-1)
        output = torch.einsum("bt,btd->bd", weights, self.value(tokens))
        return output, {"weights": weights, "prior": prior, "content_score": score}


class AnchoredResidualHead(nn.Module):
    """A bounded residual whose zero initialization exactly preserves a teacher."""

    def __init__(self, eeg_dim=32, fnirs_dim=32, hidden=32, max_correction=2.0):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(eeg_dim + fnirs_dim),
            nn.Linear(eeg_dim + fnirs_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.max_correction = float(max_correction)

    def forward(self, eeg_logits, eeg_feature, fnirs_feature):
        raw = self.out(self.feature(torch.cat((eeg_feature, fnirs_feature), -1)))
        delta = self.max_correction * torch.tanh(raw)
        residual = torch.cat((-0.5 * delta, 0.5 * delta), -1)
        return eeg_logits.detach() + residual, delta


def freeze_teacher(teacher):
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def frozen_eeg_forward(teacher, eeg_grid):
    teacher.eval()
    with torch.no_grad():
        feature = teacher.eeg_encoder(eeg_grid.permute(0, 4, 1, 2, 3).contiguous())
        logits = teacher.classifier(feature)
    return logits, feature


class LagConditionedWrite(nn.Module):
    """Low-rank, zero-initialized multiplicative modulation of Mamba B."""

    def __init__(self, dim=32, state=8, rank=4, rho=0.5):
        super().__init__()
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0,1)")
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, rank, bias=False)
        self.down = nn.Linear(rank, state, bias=False)
        nn.init.zeros_(self.down.weight)
        self.rho = float(rho)

    def forward(self, base_b, eeg, lag_prior, *, off=False):
        if off:
            return base_b
        if lag_prior.shape != (base_b.shape[0], base_b.shape[2]):
            raise ValueError("physical lag profile must match scan length")
        if (lag_prior < 0).any():
            raise ValueError("lag profile must be nonnegative")
        profile = lag_prior / lag_prior.amax(-1, keepdim=True).clamp_min(1e-8)
        modulation = torch.tanh(self.down(torch.tanh(self.up(self.norm(eeg)))))
        ratio = 1.0 + self.rho * modulation[:, :, None] * profile[:, None, :]
        return base_b * ratio.to(base_b.dtype)

