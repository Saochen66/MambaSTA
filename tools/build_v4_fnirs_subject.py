"""Build one strongly signed V4 fNIRS preprocessing version for one subject."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import mne
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.pipeline_lit import interpolate_layout_to_grid, interpolate_to_grid
from preprocessing.windowing import build_model_windows
from tools.build_strict_v3_fold import ALL_POINTS, KNOWN_FNIRS, extract_epochs, grid_epochs
from tools.v4_common import cache_identity, sha256_file

CONFIG = ROOT / "configs/mambasta_v4.yaml"


def filter_epochs(epochs, version):
    values = mne.filter.filter_data(
        epochs,
        10.0,
        version["highpass_hz"],
        version["lowpass_hz"],
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        phase="zero",
        verbose="ERROR",
    )
    start, end = version["baseline_seconds"]
    first = int(round((start + 5.0) * 10))
    last = int(round((end + 5.0) * 10))
    if first < 0 or last > values.shape[-1] or last <= first:
        raise ValueError(f"invalid baseline interval: {version['baseline_seconds']}")
    return values - values[..., first:last].mean(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("wg", "mi", "ma"), required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--version", choices=("F0", "F1", "F2"), required=True)
    args = parser.parse_args()
    study = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw_dir = {"wg": "mat2array", "mi": "mat2array_multitask/mi", "ma": "mat2array_multitask/ma"}[args.task]
    raw_path = ROOT / raw_dir / f"{args.subject}.npz"
    output = ROOT / study["paths"]["v4_cache"] / args.version / args.task / f"{args.subject}.npz"
    manifest_path = output.with_suffix(".manifest.json")
    version = study["preprocessing"]["versions"][args.version]
    with np.load(raw_path, allow_pickle=False) as data:
        hbo = data["hbo"]
        hbr = data["hbr"]
        fnirs_time = np.asarray(data["fnirs_time"]).reshape(-1)
        labels = data["label"]
        fnirs_xy = np.asarray(data["fnirs_xy"], dtype=float) if "fnirs_xy" in data.files else None
    hbo_epochs = filter_epochs(extract_epochs(hbo, fnirs_time, 10, 5, 22), version)
    hbr_epochs = filter_epochs(extract_epochs(hbr, fnirs_time, 10, 5, 22), version)
    hbo_grid = grid_epochs(hbo_epochs, fnirs_xy, KNOWN_FNIRS)
    hbr_grid = grid_epochs(hbr_epochs, fnirs_xy, KNOWN_FNIRS)
    # The window builder also validates lag provenance. EEG is a zero placeholder
    # because this artifact owns fNIRS only; actual EEG remains in the signed V3 fold cache.
    eeg_placeholder = np.zeros((60, 16, 16, 3000), dtype=np.float32)
    _, fnirs_windows, label_windows, window_metadata = build_model_windows(
        eeg_placeholder, hbo_grid, hbr_grid, labels
    )
    fnirs_windows = fnirs_windows.astype(np.float32, copy=False)
    channel_epochs = np.stack((hbo_epochs, hbr_epochs), axis=-1).astype(np.float32)
    label_windows = label_windows.astype(np.float32, copy=False)
    payload = {
        "schema": "mambasta_v4_fnirs_subject_v1",
        "task": args.task,
        "subject": args.subject,
        "version": args.version,
        "source_sha256": sha256_file(raw_path),
        "event_semantics": study["preprocessing"]["event_semantics"],
        "physical_support": {"epoch_seconds": [-5, 22], "chrono13_centers_relative_to_eeg_window_center": [-4, 8]},
        "filter_config": {**study["preprocessing"]["filter"], **version},
        "baseline_config": {"seconds_relative_to_stored_event": version["baseline_seconds"]},
        "code_sha256": {
            "builder": sha256_file(Path(__file__)),
            "windowing": sha256_file(ROOT / "preprocessing/windowing.py"),
            "interpolation": sha256_file(ROOT / "preprocessing/pipeline_lit.py"),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "mne": mne.__version__},
    }
    identity = cache_identity(payload, {"fnirs": fnirs_windows, "channel_epochs": channel_epochs, "label": label_windows})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        fnirs=fnirs_windows,
        channel_epochs=channel_epochs,
        label=label_windows,
        complete_sha256=np.asarray(identity["complete_sha256"]),
    )
    manifest = {
        **identity,
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_file_sha256": sha256_file(output),
        "array_shapes": {"fnirs": list(fnirs_windows.shape), "channel_epochs": list(channel_epochs.shape), "label": list(label_windows.shape)},
        "window_metadata": window_metadata,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "success", "output": str(output), "complete_sha256": identity["complete_sha256"], "bytes": output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
