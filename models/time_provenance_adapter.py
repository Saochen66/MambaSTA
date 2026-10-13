"""Source-index-preserving conversion from 11 overlapping lags to physical time."""

import torch
import torch.nn as nn


PHYSICAL_START_SECONDS = tuple(range(-3, 10))
PHYSICAL_CENTER_SECONDS = tuple(value + 0.5 for value in PHYSICAL_START_SECONDS)


def physical_copy_map():
    """Return all (lag, within-lag patch) copies for each physical second."""
    mapping = []
    for physical_index in range(13):
        copies = []
        for lag in range(11):
            patch = physical_index - lag
            if 0 <= patch < 3:
                copies.append((lag, patch))
        mapping.append(tuple(copies))
    return tuple(mapping)


PHYSICAL_COPY_MAP = physical_copy_map()
CANONICAL_COPY_MAP = tuple(copies[0] for copies in PHYSICAL_COPY_MAP)


class ChronologicalFNIRSAdapter(nn.Module):
    """Select one fixed source copy for each of 13 unique one-second segments.

    No averaging, filtering, or interpolation is performed.  The adapter only
    indexes exact source copies established by the provenance audit.
    """

    def __init__(self, verify_duplicates=False, atol=0.0, rtol=0.0):
        super().__init__()
        self.verify_duplicates = verify_duplicates
        self.atol = atol
        self.rtol = rtol
        self.register_buffer(
            "center_seconds",
            torch.tensor(PHYSICAL_CENTER_SECONDS, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, fnirs):
        if fnirs.ndim != 6 or fnirs.shape[1:] != (11, 16, 16, 30, 2):
            raise ValueError(f"Expected fNIRS [B,11,16,16,30,2], got {tuple(fnirs.shape)}")
        selected = []
        for physical_index, (lag, patch) in enumerate(CANONICAL_COPY_MAP):
            value = fnirs[:, lag, :, :, patch * 10 : (patch + 1) * 10, :]
            if self.verify_duplicates:
                for other_lag, other_patch in PHYSICAL_COPY_MAP[physical_index][1:]:
                    other = fnirs[
                        :, other_lag, :, :, other_patch * 10 : (other_patch + 1) * 10, :
                    ]
                    if not torch.allclose(value, other, atol=self.atol, rtol=self.rtol):
                        raise ValueError(
                            f"Physical segment {physical_index} has non-equivalent source copies"
                        )
            selected.append(value)
        return torch.stack(selected, dim=1)

    @staticmethod
    def replace_physical_segments(segments, indices, replacement):
        """Replace physical segments before the patch encoder.

        ``replacement`` may be shaped like one sample [13,16,16,10,2], one
        segment [16,16,10,2], or the full batch.  Replacement indices are
        physical seconds, so every duplicate in the original 33-token view is
        covered by construction.
        """
        if not indices:
            return segments
        result = segments.clone()
        index = torch.as_tensor(indices, device=segments.device, dtype=torch.long)
        replacement = torch.as_tensor(
            replacement, device=segments.device, dtype=segments.dtype
        )
        if replacement.ndim == 4:
            result[:, index] = replacement
        elif replacement.ndim == 5:
            result[:, index] = replacement[index].unsqueeze(0)
        elif replacement.ndim == 6:
            result[:, index] = replacement[:, index]
        else:
            raise ValueError(f"Unsupported replacement shape: {tuple(replacement.shape)}")
        return result


def expand_physical_mask_to_candidate_patches(physical_indices):
    """Map physical deletions back to every corresponding 33-token copy."""
    result = []
    for physical_index in physical_indices:
        for lag, patch in PHYSICAL_COPY_MAP[int(physical_index)]:
            result.append(lag * 3 + patch)
    return tuple(sorted(set(result)))
