"""Finite inference interventions for the released MambaSTA head.

These outputs are component interventions on one trained checkpoint.  They are
not substitutes for independently retrained unimodal controls.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def inference_ablations(head, eeg_feature, fnirs_feature):
    full, _ = head(eeg_feature, fnirs_feature, interaction=True)
    no_cross, _ = head(eeg_feature, fnirs_feature, interaction=False)
    eeg_only_like, _ = head(
        eeg_feature, head.n_ref[None].expand_as(fnirs_feature), interaction=True
    )
    fnirs_only_like, _ = head(
        head.e_ref[None].expand_as(eeg_feature), fnirs_feature, interaction=True
    )
    return {
        "FULL": full,
        "NO_CROSS": no_cross,
        "EEG_REFERENCE": eeg_only_like,
        "FNIRS_REFERENCE": fnirs_only_like,
    }


__all__ = ["inference_ablations"]
