"""One-fold trainer for the strictly cached MambaSTA V3 comparison."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mambasta_v3 import make_mambasta_v3
from training.train_v2 import (
    make_loader, normalization_hash, parameter_table, run_epoch,
    scalar_stats, set_model_seed, write_predictions,
)


def load_cache(config):
    data = np.load(config["cache_path"], allow_pickle=False)
    embedded = str(data["preprocess_hash"].item())
    if embedded != config["preprocess_hash"]:
        data.close()
        raise RuntimeError("Strict cache preprocess hash mismatch")
    split = {
        name: np.asarray(data[name], dtype=int)
        for name in (
            "train_indices", "val_indices", "test_indices",
            "train_trial_ids", "val_trial_ids", "test_trial_ids",
        )
    }
    return data, data["eeg"], data["fnirs"], data["label"], split


def make_model(config):
    return make_mambasta_v3(
        config["architecture"], dim=config.get("d_model", 32),
        state=config.get("d_state", 8), conv_kernel=config.get("d_conv", 4),
        expand=config.get("expand", 2), query_rank=config.get("query_rank", 8),
        dropout=config.get("dropout", 0.2),
    )


def split_json(split):
    return {name: [int(value) for value in values] for name, values in split.items()}


def measure_latency(model, loader, device):
    model.eval()
    eeg, fnirs, _, _ = next(iter(loader))
    eeg, fnirs = eeg.to(device), fnirs.to(device)
    with torch.no_grad():
        for _ in range(5):
            model(eeg, fnirs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model(eeg, fnirs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / 20.0 / eeg.shape[0]


def fit(config, run_dir):
    started = time.time()
    set_model_seed(config["model_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handle, eeg, fnirs, labels, split = load_cache(config)
    stats = {
        "eeg": scalar_stats(eeg, split["train_indices"]),
        "fnirs": scalar_stats(fnirs, split["train_indices"]),
    }
    train_loader = make_loader(eeg, fnirs, labels, split, "train", stats, config, True)
    val_loader = make_loader(eeg, fnirs, labels, split, "val", stats, config, False)
    model = make_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_loss, best_state, best_epoch, best_val = float("inf"), None, None, None
    history, stale = [], 0
    for epoch in range(1, config["max_epochs"] + 1):
        train_metrics, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, config["gradient_clip"]
        )
        with torch.no_grad():
            val_metrics, _ = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[V3] epoch={epoch}/{config['max_epochs']} "
            f"train={train_metrics['loss']:.4f}/{train_metrics['accuracy']:.4f} "
            f"val={val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f}", flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss, best_epoch, best_val = val_metrics["loss"], epoch, val_metrics
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        selected_val, val_predictions = run_epoch(model, val_loader, criterion, device)
    latency = measure_latency(model, val_loader, device)
    normalization = {
        "eeg_mean": stats["eeg"][0], "eeg_std": stats["eeg"][1],
        "fnirs_mean": stats["fnirs"][0], "fnirs_std": stats["fnirs"][1],
    }
    normalization["sha256"] = normalization_hash(normalization)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": parameters,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_ms_per_sample": latency, "elapsed_seconds": time.time() - started,
    }
    metrics = {
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_val": best_val, "selected_checkpoint_val": selected_val, "test": None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "config.json": config, "environment.json": environment,
        "split.json": split_json(split), "normalization.json": normalization,
        "history.json": history, "metrics.json": metrics,
        "module_parameters.json": parameter_table(model),
    }
    for name, payload in artifacts.items():
        (run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    write_predictions(run_dir / "validation_predictions.csv", val_predictions)
    torch.save({"model_state": best_state, "config": config, "metrics": metrics},
               run_dir / "best_model.pt")
    (run_dir / "success.json").write_text(json.dumps({
        "signature": config["signature"], "fit_complete": True,
        "test_evaluated": False, "failure": None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    handle.close()


def evaluate(config, run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"]["signature"] != config["signature"]:
        raise RuntimeError("Checkpoint signature mismatch")
    handle, eeg, fnirs, labels, split = load_cache(config)
    normalization = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
    stats = {"eeg": (normalization["eeg_mean"], normalization["eeg_std"]),
             "fnirs": (normalization["fnirs_mean"], normalization["fnirs_std"])}
    loader = make_loader(eeg, fnirs, labels, split, "test", stats, config, False)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        metrics, predictions = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    saved["test"] = metrics
    (run_dir / "metrics.json").write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_predictions(run_dir / "test_predictions.csv", predictions)
    success = json.loads((run_dir / "success.json").read_text(encoding="utf-8"))
    success["test_evaluated"] = True
    (run_dir / "success.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    handle.close()
    print(f"[V3] test={metrics['loss']:.4f}/{metrics['accuracy']:.4f} "
          f"trial={metrics['trial']['accuracy']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "fit":
        fit(config, Path(args.run_dir))
    else:
        evaluate(config, Path(args.run_dir))


if __name__ == "__main__":
    main()
