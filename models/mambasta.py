"""Stable public entry point for the released MambaSTA model.

The frozen implementation is the internally audited J1_R4 graph.  The public
name deliberately omits development-version numbering.
"""
from __future__ import annotations

from models.mambasta_v8 import V8JointModel, joint_training_loss


class MambaSTA(V8JointModel):
    """Released rank-4 reference-centred EEG-fNIRS joint model."""

    def __init__(self, config):
        values = dict(config)
        values["config_id"] = "J1_R4"
        super().__init__(values)


def build_mambasta(config):
    return MambaSTA(config)


__all__ = ["MambaSTA", "build_mambasta", "joint_training_loss"]
