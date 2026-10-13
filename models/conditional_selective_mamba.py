"""Official-Mamba-based low-rank EEG-conditioned selective scan."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import mamba_ssm
    from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError as exc:
    mamba_ssm = None
    Mamba = None
    selective_scan_fn = None
    causal_conv1d_fn = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


class ConditionalSelectiveMamba(nn.Module):
    """Mamba-1 with low-rank EEG-token interaction in Delta or C.

    The wrapped official module owns every Mamba parameter and its special
    initialization.  The scan path is expanded only so the condition can be
    inserted before ``selective_scan_fn``.  Delta bias and softplus remain in
    the official scan call and are each applied exactly once.
    """

    def __init__(
        self,
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        condition_rank=4,
        condition_scale=0.5,
        condition_mode="delta",
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("ConditionalSelectiveMamba requires mamba_ssm") from MAMBA_IMPORT_ERROR
        if condition_mode not in {"none", "delta", "c"}:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")
        self.d_model = d_model
        self.condition_rank = condition_rank
        self.condition_scale = condition_scale
        self.condition_mode = condition_mode
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.eeg_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.eeg_projection = nn.Linear(d_model, condition_rank, bias=False)
        self.token_projection = nn.Linear(d_model, condition_rank, bias=False)
        if condition_mode == "delta":
            self.condition_projection = nn.Linear(
                condition_rank, self.mamba.d_inner, bias=False
            )
        elif condition_mode == "c":
            self.condition_projection = nn.Linear(
                condition_rank, self.mamba.d_state, bias=False
            )
        else:
            self.register_parameter("condition_projection", None)
        if condition_mode != "none":
            nn.init.zeros_(self.condition_projection.weight)
        self.mamba_version = getattr(mamba_ssm, "__version__", None)

    def copy_mamba_weights_from(self, source):
        """Copy an official Mamba state for numerical equivalence tests."""
        self.mamba.load_state_dict(source.state_dict())

    def _interaction(self, hidden_states, eeg_condition, condition_off):
        if self.condition_mode == "none" or condition_off:
            return None
        if eeg_condition is None or eeg_condition.shape != (
            hidden_states.shape[0], self.d_model
        ):
            raise ValueError(
                f"Expected EEG condition [B,{self.d_model}], got "
                f"{None if eeg_condition is None else tuple(eeg_condition.shape)}"
            )
        eeg_rank = torch.tanh(self.eeg_projection(self.eeg_norm(eeg_condition)))
        token_rank = torch.tanh(self.token_projection(self.token_norm(hidden_states)))
        return token_rank * eeg_rank.unsqueeze(1)

    def forward(
        self,
        hidden_states,
        eeg_condition=None,
        *,
        condition_off=False,
        return_diagnostics=False,
    ):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected tokens [B,L,{self.d_model}], got {tuple(hidden_states.shape)}"
            )
        batch, sequence_length, _ = hidden_states.shape
        module = self.mamba
        interaction = self._interaction(hidden_states, eeg_condition, condition_off)

        xz = rearrange(
            module.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=sequence_length,
        )
        if module.in_proj.bias is not None:
            xz = xz + rearrange(module.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            x = module.act(module.conv1d(x)[..., :sequence_length])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(module.conv1d.weight, "d 1 w -> d w"),
                bias=module.conv1d.bias,
                activation=module.activation,
            )

        projected = module.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt_rank, state_b, state_c = torch.split(
            projected, [module.dt_rank, module.d_state, module.d_state], dim=-1
        )
        delta_logits = F.linear(dt_rank, module.dt_proj.weight)
        delta_logits = rearrange(
            delta_logits, "(b l) d -> b d l", b=batch, l=sequence_length
        )
        state_b = rearrange(
            state_b, "(b l) n -> b n l", b=batch, l=sequence_length
        ).contiguous()
        state_c = rearrange(
            state_c, "(b l) n -> b n l", b=batch, l=sequence_length
        ).contiguous()

        condition_term = None
        if interaction is not None:
            condition_term = self.condition_scale * torch.tanh(
                self.condition_projection(interaction)
            )
            if self.condition_mode == "delta":
                delta_logits = delta_logits + rearrange(condition_term, "b l d -> b d l")
            elif self.condition_mode == "c":
                state_c = state_c + rearrange(condition_term, "b l n -> b n l")

        transition = -torch.exp(module.A_log.float())
        scan_output = selective_scan_fn(
            x,
            delta_logits,
            transition,
            state_b,
            state_c,
            module.D.float(),
            z=z,
            delta_bias=module.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        output = module.out_proj(rearrange(scan_output, "b d l -> b l d"))
        if not return_diagnostics:
            return output

        with torch.no_grad():
            state_scan = selective_scan_fn(
                x,
                delta_logits,
                transition,
                state_b,
                state_c,
                None,
                z=z,
                delta_bias=module.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            direct_scan = scan_output - state_scan
            state_sequence = rearrange(state_scan, "b d l -> b l d")
            direct_sequence = rearrange(direct_scan, "b d l -> b l d")
            full_sequence = rearrange(scan_output, "b d l -> b l d")
            state_projected = F.linear(state_sequence, module.out_proj.weight, bias=None)
            direct_projected = F.linear(direct_sequence, module.out_proj.weight, bias=None)

            def rms(value):
                return value.float().square().mean().sqrt()

            def cosine(left, right):
                left = left.float().reshape(left.shape[0], -1)
                right = right.float().reshape(right.shape[0], -1)
                numerator = (left * right).sum(dim=1)
                denominator = left.norm(dim=1) * right.norm(dim=1)
                return (numerator / denominator.clamp_min(1e-12)).mean()

            actual_delta = F.softplus(
                delta_logits + module.dt_proj.bias.view(1, -1, 1).to(delta_logits.dtype)
            )
            decay = torch.exp(
                actual_delta.float().unsqueeze(2)
                * transition.float().unsqueeze(0).unsqueeze(-1)
            )
            diagnostics = {
                "condition_mode": self.condition_mode,
                "condition_off": bool(condition_off),
                "delta_mean": actual_delta.mean().detach(),
                "delta_std": actual_delta.std(unbiased=False).detach(),
                "condition_term_abs_mean": (
                    torch.zeros((), device=output.device, dtype=output.dtype)
                    if condition_term is None
                    else condition_term.abs().mean().detach()
                ),
                "condition_term_abs_max": (
                    torch.zeros((), device=output.device, dtype=output.dtype)
                    if condition_term is None
                    else condition_term.abs().max().detach()
                ),
                "scan_state_rms": rms(state_scan).detach(),
                "scan_direct_rms": rms(direct_scan).detach(),
                "scan_full_rms": rms(scan_output).detach(),
                "scan_state_to_full_rms": (
                    rms(state_scan) / rms(scan_output).clamp_min(1e-12)
                ).detach(),
                "scan_direct_to_full_rms": (
                    rms(direct_scan) / rms(scan_output).clamp_min(1e-12)
                ).detach(),
                "scan_state_direct_cosine": cosine(state_scan, direct_scan).detach(),
                "projected_state_rms": rms(state_projected).detach(),
                "projected_direct_rms": rms(direct_projected).detach(),
                "projected_full_no_bias_rms": rms(
                    state_projected + direct_projected
                ).detach(),
                "projected_state_direct_cosine": cosine(
                    state_projected, direct_projected
                ).detach(),
                "decay_mean": decay.mean().detach(),
                "decay_std": decay.std(unbiased=False).detach(),
            }
        return output, diagnostics
