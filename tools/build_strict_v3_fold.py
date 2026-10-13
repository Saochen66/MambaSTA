"""Build one leakage-controlled strict_v3 preprocessing cache.

The cache is fold specific because ICA is fitted on training trials only. All
architecture candidates in that fold consume the exact same cached arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import mne
import numpy as np
from mne_icalabel import label_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.auto_preprocessing_literature import (
    EEG_CHN_NAMES,
    EOG_CHN_NAMES,
    auto_exclude_components,
    enable_onnxruntime_path_compatibility,
)
from preprocessing.pipeline_lit import interpolate_layout_to_grid, interpolate_to_grid
from preprocessing.windowing import build_model_windows
from training.splits import subject_session_split


KNOWN_EEG = np.asarray([
    [0, 6], [2, 5], [2, 8], [3, 7], [5, 2], [5, 6], [7, 1], [7, 4],
    [7, 8], [9, 2], [9, 6], [11, 2], [11, 5], [11, 8], [13, 8], [14, 6],
    [0, 10], [2, 11], [3, 9], [5, 10], [5, 14], [7, 12], [7, 15], [9, 10],
    [9, 14], [11, 11], [11, 14], [14, 10],
], dtype=float)
KNOWN_FNIRS = np.asarray([
    [2, 4], [3, 4], [1, 5], [2, 5], [1, 7], [3, 6], [2, 7], [3, 8],
    [1, 8], [2, 9], [1, 9], [6, 4], [7, 5], [7, 3], [8, 4], [10, 5],
    [11, 6], [11, 4], [12, 5], [3, 10], [2, 11], [3, 12], [1, 11], [2, 12],
    [6, 12], [7, 13], [7, 11], [8, 12], [10, 11], [11, 12], [11, 10],
    [12, 11], [12, 8], [13, 7], [13, 9], [14, 8],
], dtype=float)
ALL_POINTS = np.column_stack(np.meshgrid(np.arange(16), np.arange(16))).reshape(-1, 2)
# The expression above interleaves axes; use the historical row/column ordering.
_XX, _YY = np.meshgrid(np.arange(16), np.arange(16))
ALL_POINTS = np.column_stack((_XX.ravel(), _YY.ravel()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_epochs(signal: np.ndarray, onsets_ms: np.ndarray, rate: int,
                   pre_seconds: int, post_seconds: int) -> np.ndarray:
    length = (pre_seconds + post_seconds) * rate
    result = np.empty((len(onsets_ms), signal.shape[0], length), dtype=np.float64)
    for trial, onset_ms in enumerate(onsets_ms):
        start = int((float(onset_ms) / 1000.0 - pre_seconds) * rate)
        end = start + length
        if start < 0 or end > signal.shape[1]:
            raise ValueError(f"trial {trial} epoch [{start}:{end}] outside signal")
        result[trial] = signal[:, start:end]
    return result


def make_eeg_info(names: list[str]) -> tuple[mne.Info, list[str], list[str]]:
    if len(names) < 3:
        raise ValueError("EEG must include at least one EEG and two EOG channels")
    eeg_names, eog_names = names[:-2], names[-2:]
    info = mne.create_info(names, 200, ["eeg"] * len(eeg_names) + ["eog"] * 2)
    info.set_montage("standard_1005", on_missing="warn")
    return info, eeg_names, eog_names


def filter_eeg_epochs(epochs: np.ndarray, info: mne.Info) -> mne.EpochsArray:
    # Array filters operate only on the last axis, so trials remain independent.
    filtered = mne.filter.notch_filter(
        epochs, 200.0, [50.0], method="iir", verbose="ERROR"
    )
    filtered = mne.filter.filter_data(
        filtered, 200.0, 0.5, 45.0, method="iir",
        iir_params=dict(order=6, ftype="butter"), verbose="ERROR",
    )
    instance = mne.EpochsArray(filtered, info, tmin=-5.0, baseline=None, verbose="ERROR")
    instance.set_eeg_reference("average", verbose="ERROR")
    return instance


def filter_fnirs_epochs(epochs: np.ndarray) -> np.ndarray:
    values = mne.filter.filter_data(
        epochs, 10.0, 0.01, 0.1, method="iir",
        iir_params=dict(order=4, ftype="butter"), verbose="ERROR",
    )
    # First 8 seconds of the stored epoch are -5..+3 s relative to onset.
    values -= values[..., :80].mean(axis=-1, keepdims=True)
    return values


def grid_epochs(values: np.ndarray, xy: np.ndarray | None, known: np.ndarray) -> np.ndarray:
    result = np.empty((values.shape[0], 16, 16, values.shape[-1]), dtype=np.float32)
    for trial in range(values.shape[0]):
        if xy is None:
            result[trial] = interpolate_to_grid(values[trial], known, ALL_POINTS).astype(np.float32)
        else:
            result[trial] = interpolate_layout_to_grid(values[trial], xy).astype(np.float32)
    return result


def build_fold(raw_path: Path, output_path: Path, test_session: int) -> dict:
    with np.load(raw_path, allow_pickle=False) as data:
        eeg = data["eeg"]
        hbo = data["hbo"]
        hbr = data["hbr"]
        eeg_time = np.asarray(data["eeg_time"]).reshape(-1)
        fnirs_time = np.asarray(data["fnirs_time"]).reshape(-1)
        labels = data["label"]
        eeg_names = ([str(item) for item in data["eeg_ch_names"]]
                     if "eeg_ch_names" in data.files else EEG_CHN_NAMES + EOG_CHN_NAMES)
        eeg_xy = np.asarray(data["eeg_xy"], dtype=float) if "eeg_xy" in data.files else None
        fnirs_xy = np.asarray(data["fnirs_xy"], dtype=float) if "fnirs_xy" in data.files else None

    trial_labels = labels.T if labels.shape[0] == 2 else labels
    repeated_labels = np.repeat(trial_labels, 10, axis=0)
    split = subject_session_split(repeated_labels, test_session=test_session, seed=42)
    train_trials = np.asarray(split["train_trial_ids"], dtype=int)

    eeg_epochs = extract_epochs(eeg, eeg_time, 200, 5, 10)
    hbo_epochs = extract_epochs(hbo, fnirs_time, 10, 5, 22)
    hbr_epochs = extract_epochs(hbr, fnirs_time, 10, 5, 22)
    info, eeg_channel_names, eog_names = make_eeg_info(eeg_names)
    filtered = filter_eeg_epochs(eeg_epochs, info)

    fit_epochs = filtered[train_trials].copy().filter(
        1.0, 45.0, method="iir", iir_params=dict(order=6, ftype="butter"), verbose="ERROR"
    )
    ica = mne.preprocessing.ICA(
        n_components=min(20, len(eeg_channel_names)), method="infomax",
        fit_params=dict(extended=True), random_state=42, max_iter="auto",
    )
    ica.fit(fit_epochs, picks="eeg", verbose="ERROR")
    enable_onnxruntime_path_compatibility()
    ic_labels = label_components(fit_epochs, ica, method="iclabel")
    eog_candidates, eog_scores = ica.find_bads_eog(
        fit_epochs, ch_name=eog_names, verbose="ERROR"
    )
    excluded = auto_exclude_components(
        ic_labels, prob_threshold=0.8, eog_candidates=eog_candidates
    )
    ica.exclude = excluded
    cleaned = ica.apply(filtered.copy(), verbose="ERROR").get_data(picks="eeg", copy=True)

    hbo_clean = filter_fnirs_epochs(hbo_epochs)
    hbr_clean = filter_fnirs_epochs(hbr_epochs)
    eeg_grid = grid_epochs(cleaned, eeg_xy, KNOWN_EEG)
    hbo_grid = grid_epochs(hbo_clean, fnirs_xy, KNOWN_FNIRS)
    hbr_grid = grid_epochs(hbr_clean, fnirs_xy, KNOWN_FNIRS)
    eeg_input, fnirs_input, label_input, window_meta = build_model_windows(
        eeg_grid, hbo_grid, hbr_grid, labels
    )
    eeg_input = eeg_input.astype(np.float32, copy=False)
    fnirs_input = fnirs_input.astype(np.float32, copy=False)
    label_input = label_input.astype(np.float32, copy=False)

    source_hash = sha256_file(raw_path)
    semantic = {
        "schema": "strict_v3_fold_v1",
        "source_sha256": source_hash,
        "test_session": int(test_session),
        "split_seed": 42,
        "filter_scope": "independent_trial_epoch",
        "ica_fit_trial_ids": train_trials.tolist(),
        "ica_components": int(ica.n_components_),
        "ica_excluded": [int(item) for item in excluded],
    }
    preprocess_hash = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, eeg=eeg_input, fnirs=fnirs_input, label=label_input,
        train_indices=split["train_indices"], val_indices=split["val_indices"],
        test_indices=split["test_indices"],
        train_trial_ids=split["train_trial_ids"], val_trial_ids=split["val_trial_ids"],
        test_trial_ids=split["test_trial_ids"], preprocess_hash=np.asarray(preprocess_hash),
    )
    probabilities = np.asarray(ic_labels["y_pred_proba"])
    if probabilities.ndim > 1:
        probabilities = probabilities.max(axis=1)
    manifest = {
        **semantic,
        "preprocess_hash": preprocess_hash,
        "source": str(raw_path.resolve()),
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "eeg_channel_names": eeg_channel_names,
        "eog_channel_names": eog_names,
        "iclabel_labels": list(ic_labels["labels"]),
        "iclabel_probabilities": [float(value) for value in probabilities],
        "eog_candidates": [int(value) for value in eog_candidates],
        "eog_scores": np.asarray(eog_scores).astype(float).reshape(-1).tolist(),
        "val_trial_ids": split["val_trial_ids"].tolist(),
        "test_trial_ids": split["test_trial_ids"].tolist(),
        "array_shapes": {"eeg": list(eeg_input.shape), "fnirs": list(fnirs_input.shape),
                         "label": list(label_input.shape)},
        "window_metadata": window_meta,
        "versions": {"python": sys.version, "mne": mne.__version__},
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("wg", "mi", "ma"), required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--test-session", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--base-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    raw_dir = {"wg": "mat2array", "mi": "mat2array_multitask/mi",
               "ma": "mat2array_multitask/ma"}[args.task]
    raw_path = args.base_dir / raw_dir / f"{args.subject}.npz"
    output_path = (args.base_dir / "experiments/mambasta_v3_20261004/strict_cache" /
                   args.task / args.subject / f"session{args.test_session}.npz")
    result = build_fold(raw_path, output_path, args.test_session)
    print(json.dumps({"status": "success", "output": result["output"],
                      "preprocess_hash": result["preprocess_hash"],
                      "array_shapes": result["array_shapes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
