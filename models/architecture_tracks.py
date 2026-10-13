"""Five isolated Phase-3 architecture tracks from the long-iteration plan."""

import torch
import torch.nn as nn

from models.encoder_audit import EEGNetStyleEncoder
from models.lightweight_baseline import LightweightEncoder, SpectralEEGEncoder

try:
    import mamba_ssm
    from mamba_ssm import Mamba
except ImportError as exc:
    mamba_ssm = None
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


LAG_SECONDS = tuple(range(-3, 8))


class DelayEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))

    def forward(self, values):
        return self.network(values.unsqueeze(-1) / 7.0)


class GatedConvMixer(nn.Module):
    """MambaOut-style local gated convolution negative control."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_projection = nn.Linear(d_model, d_model * 2)
        self.depthwise = nn.Conv1d(
            d_model * 2, d_model * 2, kernel_size=3, padding=1,
            groups=d_model * 2, bias=False,
        )
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        projected = self.in_projection(self.norm(x)).transpose(1, 2)
        projected = self.depthwise(projected).transpose(1, 2)
        value, gate = projected.chunk(2, dim=-1)
        return self.out_projection(value * torch.sigmoid(gate))


class TrackBase(nn.Module):
    def __init__(self, eeg_encoder, d_model=32, dropout=0.2, mixer="mamba"):
        super().__init__()
        if mixer == "mamba":
            if Mamba is None:
                raise ImportError("Architecture tracks require official mamba_ssm") from MAMBA_IMPORT_ERROR
            self.token_mixer = Mamba(d_model=d_model, d_state=8, d_conv=4, expand=2)
        elif mixer == "gated_conv":
            self.token_mixer = GatedConvMixer(d_model)
        else:
            raise ValueError(mixer)
        self.eeg_encoder = eeg_encoder
        self.condition_projection = nn.Linear(d_model, d_model)
        self.delay_embedding = DelayEmbedding(d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.relevance = nn.Linear(d_model, 1)
        self.eeg_norm = nn.LayerNorm(d_model)
        self.fnirs_norm = nn.LayerNorm(d_model)
        self.modality_gate = nn.Linear(d_model * 2, 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2)
        )
        self.mamba_version = getattr(mamba_ssm, "__version__", None) if mixer == "mamba" else None

    def encode_eeg(self, eeg):
        return self.eeg_encoder(eeg.permute(0, 4, 1, 2, 3).contiguous())

    def mix_conditioned(self, tokens, eeg_feature, delay_values, condition_feature=None):
        delay = self.delay_embedding(delay_values.to(tokens.device, tokens.dtype))
        if condition_feature is None:
            condition_feature = eeg_feature
        conditioned = tokens + self.condition_projection(condition_feature).unsqueeze(1) + delay.unsqueeze(0)
        mixed = self.token_mixer(self.input_norm(conditioned))
        return self.output_norm(conditioned + mixed)

    def finish(self, eeg_feature, mixed, lag_relevance, token_relevance=None):
        relevance = lag_relevance if token_relevance is None else token_relevance
        fnirs_feature = torch.sum(relevance.unsqueeze(-1) * mixed, dim=1)
        eeg_feature = self.eeg_norm(eeg_feature)
        fnirs_feature = self.fnirs_norm(fnirs_feature)
        modality_weights = torch.softmax(
            self.modality_gate(torch.cat((eeg_feature, fnirs_feature), dim=-1)), dim=-1
        )
        fused = modality_weights[:, :1] * eeg_feature + modality_weights[:, 1:] * fnirs_feature
        return self.classifier(fused), {
            "modality_weights": modality_weights,
            "lag_relevance": lag_relevance,
            "token_relevance": relevance,
        }


class CrossMambaLite(TrackBase):
    """Track A: EEG condition is added before the selective state update."""

    def __init__(self, d_model=32, eeg_encoder=None, mixer="mamba"):
        super().__init__(eeg_encoder or EEGNetStyleEncoder(d_model), d_model, mixer=mixer)
        self.fnirs_encoder = LightweightEncoder(2, (5, 3), (2, 2), (8, d_model))

    def compute_relevance(self, mixed, condition_feature):
        return torch.softmax(self.relevance(mixed).squeeze(-1), dim=1)

    def forward(self, eeg, fnirs, intervention=None):
        intervention = intervention or {}
        batch, lag_count = fnirs.shape[:2]
        eeg_feature = self.encode_eeg(eeg)
        if "lag_permutation" in intervention:
            fnirs = fnirs[:, intervention["lag_permutation"]]
        flat = fnirs.reshape(batch * lag_count, 16, 16, 30, 2).permute(0, 4, 1, 2, 3).contiguous()
        tokens = self.fnirs_encoder(flat).reshape(batch, lag_count, -1)
        delay = torch.tensor(LAG_SECONDS, device=fnirs.device, dtype=fnirs.dtype)
        if "delay_permutation" in intervention:
            delay = delay[intervention["delay_permutation"]]
        condition_feature = eeg_feature
        if "condition_permutation" in intervention:
            condition_feature = eeg_feature[intervention["condition_permutation"]]
        mixed = self.mix_conditioned(tokens, eeg_feature, delay, condition_feature)
        relevance = self.compute_relevance(mixed, condition_feature)
        if intervention.get("uniform_relevance", False):
            relevance = torch.full_like(relevance, 1.0 / lag_count)
        elif intervention.get("fixed_central_relevance", False):
            relevance = torch.zeros_like(relevance)
            relevance[:, 3] = 1.0
        return self.finish(eeg_feature, mixed, relevance)


class HierarchicalLagMamba(TrackBase):
    """Track B: 11 lags x 3 within-lag temporal patches = 33 tokens."""

    def __init__(self, d_model=32):
        super().__init__(EEGNetStyleEncoder(d_model), d_model)
        self.patch_encoder = LightweightEncoder(2, (5, 3), (2, 1), (8, d_model))
        self.local_time_embedding = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.local_time_embedding, std=0.02)

    def forward(self, eeg, fnirs, intervention=None):
        intervention = intervention or {}
        batch = fnirs.shape[0]
        eeg_feature = self.encode_eeg(eeg)
        if "lag_permutation" in intervention:
            fnirs = fnirs[:, intervention["lag_permutation"]]
        patches = fnirs.reshape(batch, 11, 16, 16, 3, 10, 2)
        patches = patches.permute(0, 1, 4, 6, 2, 3, 5).reshape(batch * 33, 2, 16, 16, 10)
        tokens = self.patch_encoder(patches).reshape(batch, 11, 3, -1)
        tokens = tokens + self.local_time_embedding.view(1, 1, 3, -1)
        tokens = tokens.reshape(batch, 33, -1)
        delay = torch.tensor(LAG_SECONDS, device=fnirs.device, dtype=fnirs.dtype).repeat_interleave(3)
        if "delay_permutation" in intervention:
            base_delay = torch.tensor(LAG_SECONDS, device=fnirs.device, dtype=fnirs.dtype)
            delay = base_delay[intervention["delay_permutation"]].repeat_interleave(3)
        condition_feature = eeg_feature
        if "condition_permutation" in intervention:
            condition_feature = eeg_feature[intervention["condition_permutation"]]
        mixed = self.mix_conditioned(tokens, eeg_feature, delay, condition_feature)
        token_relevance = torch.softmax(self.relevance(mixed).squeeze(-1), dim=1)
        if intervention.get("uniform_relevance", False):
            token_relevance = torch.full_like(token_relevance, 1.0 / token_relevance.shape[1])
        elif intervention.get("fixed_central_relevance", False):
            token_relevance = torch.zeros_like(token_relevance)
            token_relevance[:, 9:12] = 1.0 / 3.0
        lag_relevance = token_relevance.reshape(batch, 11, 3).sum(dim=2)
        return self.finish(eeg_feature, mixed, lag_relevance, token_relevance)


class RegionLagMamba(TrackBase):
    """Track C: four fixed regions x 11 lags = 44 region-major tokens."""

    def __init__(self, d_model=32):
        super().__init__(EEGNetStyleEncoder(d_model), d_model)
        self.region_encoder = LightweightEncoder(2, (5, 3), (2, 2), (8, d_model))
        self.region_embedding = nn.Parameter(torch.zeros(4, d_model))
        nn.init.normal_(self.region_embedding, std=0.02)

    def forward(self, eeg, fnirs, intervention=None):
        intervention = intervention or {}
        batch = fnirs.shape[0]
        eeg_feature = self.encode_eeg(eeg)
        if "lag_permutation" in intervention:
            fnirs = fnirs[:, intervention["lag_permutation"]]
        regions = torch.stack(
            (
                fnirs[:, :, :8, :8], fnirs[:, :, :8, 8:],
                fnirs[:, :, 8:, :8], fnirs[:, :, 8:, 8:],
            ), dim=2,
        )
        regions = regions.permute(0, 2, 1, 6, 3, 4, 5).reshape(batch * 44, 2, 8, 8, 30)
        tokens = self.region_encoder(regions).reshape(batch, 4, 11, -1)
        tokens = tokens + self.region_embedding.view(1, 4, 1, -1)
        tokens = tokens.reshape(batch, 44, -1)
        delay = torch.tensor(LAG_SECONDS, device=fnirs.device, dtype=fnirs.dtype).repeat(4)
        if "delay_permutation" in intervention:
            base_delay = torch.tensor(LAG_SECONDS, device=fnirs.device, dtype=fnirs.dtype)
            delay = base_delay[intervention["delay_permutation"]].repeat(4)
        condition_feature = eeg_feature
        if "condition_permutation" in intervention:
            condition_feature = eeg_feature[intervention["condition_permutation"]]
        mixed = self.mix_conditioned(tokens, eeg_feature, delay, condition_feature)
        token_relevance = torch.softmax(self.relevance(mixed).squeeze(-1), dim=1)
        if intervention.get("uniform_relevance", False):
            token_relevance = torch.full_like(token_relevance, 1.0 / token_relevance.shape[1])
        elif intervention.get("fixed_central_relevance", False):
            token_relevance = torch.zeros_like(token_relevance)
            token_relevance[:, (3, 14, 25, 36)] = 1.0 / 4.0
        lag_relevance = token_relevance.reshape(batch, 4, 11).sum(dim=1)
        return self.finish(eeg_feature, mixed, lag_relevance, token_relevance)


class SpectralConditionedMamba(CrossMambaLite):
    """Track D: spectral EEG condition plus an explicit residual EEG path."""

    def __init__(self, d_model=32):
        super().__init__(d_model=d_model, eeg_encoder=SpectralEEGEncoder(d_model))


class MambaOutControl(CrossMambaLite):
    """Track E: Track-A inputs and fusion with the SSM replaced by gated CNN."""

    def __init__(self, d_model=32):
        super().__init__(d_model=d_model, mixer="gated_conv")


class FiLMConditionedMamba(CrossMambaLite):
    """Track F: EEG applies feature-wise scale and shift before the Mamba update."""

    def __init__(self, d_model=32):
        super().__init__(d_model=d_model)
        self.condition_film = nn.Linear(d_model, d_model * 2)

    def mix_conditioned(self, tokens, eeg_feature, delay_values, condition_feature=None):
        if condition_feature is None:
            condition_feature = eeg_feature
        delay = self.delay_embedding(delay_values.to(tokens.device, tokens.dtype)).unsqueeze(0)
        scale, shift = self.condition_film(condition_feature).chunk(2, dim=-1)
        base = tokens + delay
        conditioned = base * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        mixed = self.token_mixer(self.input_norm(conditioned))
        return self.output_norm(conditioned + mixed)


class QueryRelevanceMamba(CrossMambaLite):
    """Track G: EEG condition also queries lag states when computing relevance."""

    def __init__(self, d_model=32):
        super().__init__(d_model=d_model)
        self.condition_query = nn.Linear(d_model, d_model, bias=False)
        self.lag_key = nn.Linear(d_model, d_model, bias=False)
        self.query_scale = d_model ** -0.5

    def compute_relevance(self, mixed, condition_feature):
        base_logits = self.relevance(mixed).squeeze(-1)
        query = self.condition_query(condition_feature).unsqueeze(1)
        conditional_logits = (query * self.lag_key(mixed)).sum(dim=-1) * self.query_scale
        return torch.softmax(base_logits + conditional_logits, dim=1)


class ConditionOnlyCrossMamba(CrossMambaLite):
    """Track H: removes the direct EEG residual so EEG can act only through conditioning."""

    def finish(self, eeg_feature, mixed, lag_relevance, token_relevance=None):
        relevance = lag_relevance if token_relevance is None else token_relevance
        fnirs_feature = self.fnirs_norm(torch.sum(relevance.unsqueeze(-1) * mixed, dim=1))
        modality_weights = torch.zeros(
            (mixed.shape[0], 2), device=mixed.device, dtype=mixed.dtype
        )
        modality_weights[:, 1] = 1.0
        return self.classifier(fnirs_feature), {
            "modality_weights": modality_weights,
            "lag_relevance": lag_relevance,
            "token_relevance": relevance,
        }


def make_architecture_track(name):
    return {
        "track_a": CrossMambaLite,
        "track_b": HierarchicalLagMamba,
        "track_c": RegionLagMamba,
        "track_d": SpectralConditionedMamba,
        "track_e": MambaOutControl,
        "track_f": FiLMConditionedMamba,
        "track_g": QueryRelevanceMamba,
        "track_h": ConditionOnlyCrossMamba,
    }[name]()
