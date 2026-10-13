"""One-fold trainer/evaluator for the validation-selected MambaSTA-v2 study."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.encoder_audit import make_encoder_audit_model
from models.hb_mamba_lite import make_hb_mamba_lite
from models.mamba_sta_v2 import make_mambasta_v2
from models.lite_sta_alignment import make_lite_sta_alignment
from training.splits import subject_session_split


V2_NAMES = {
    "a0_chrono_mamba", "a0_chrono_mlp", "a1_delta_cond",
    "a1_c_cond", "chrono_fnirs_only",
}
LITE_STA_NAMES = {"b0_lite_sta_attention", "b1_lite_sta_cond_mamba"}


def set_model_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scalar_stats(array, indices):
    total = 0.0
    count = 0
    for index in indices:
        sample = array[index]
        total += float(np.sum(sample, dtype=np.float64))
        count += sample.size
    mean = total / count
    squared = 0.0
    for index in indices:
        sample = np.asarray(array[index], dtype=np.float64)
        squared += float(np.sum(np.square(sample - mean), dtype=np.float64))
    std = float(np.sqrt(squared / count))
    if not np.isfinite(std) or std < 1e-12:
        raise ValueError(f"Invalid training standard deviation: {std}")
    return mean, std


class IndexedDataset(Dataset):
    def __init__(self, eeg, fnirs, labels, indices, eeg_stats, fnirs_stats):
        self.eeg = eeg
        self.fnirs = fnirs
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.eeg_stats = eeg_stats
        self.fnirs_stats = fnirs_stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        eeg = np.asarray(
            (self.eeg[index] - self.eeg_stats[0]) / self.eeg_stats[1], dtype=np.float32
        )
        fnirs = np.asarray(
            (self.fnirs[index] - self.fnirs_stats[0]) / self.fnirs_stats[1], dtype=np.float32
        )
        target = int(np.argmax(self.labels[index]))
        return torch.from_numpy(eeg), torch.from_numpy(fnirs), target, index


def confusion_metrics(confusion):
    confusion = np.asarray(confusion, dtype=float)
    total = confusion.sum()
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, support, out=np.zeros(2), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(2), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(2), where=(precision + recall) > 0,
    )
    accuracy = diagonal.sum() / total
    expected = float(np.dot(support, predicted) / (total * total))
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0
    return {
        "accuracy": float(accuracy), "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()), "kappa": float(kappa),
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def aggregate_trials(probabilities, targets, indices):
    rows = []
    confusion = np.zeros((2, 2), dtype=np.int64)
    for trial in sorted(set(int(index) // 10 for index in indices)):
        mask = np.asarray([int(index) // 10 == trial for index in indices])
        probability = probabilities[mask].mean(axis=0)
        target_values = np.unique(targets[mask])
        if len(target_values) != 1 or int(mask.sum()) != 10:
            raise AssertionError(f"Trial {trial} is incomplete or has mixed labels")
        target = int(target_values[0])
        prediction = int(np.argmax(probability))
        confusion[target, prediction] += 1
        rows.append(
            {
                "trial_global": trial, "session": trial // 20,
                "trial_in_session": trial % 20, "target": target,
                "probability_0": float(probability[0]),
                "probability_1": float(probability[1]), "prediction": prediction,
            }
        )
    return {**confusion_metrics(confusion), "trials": rows}


def run_epoch(model, loader, criterion, device, optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    probabilities = []
    raw_logits = []
    targets = []
    indices = []
    gradient_sum = 0.0
    gradient_steps = 0
    for eeg, fnirs, target, index in loader:
        eeg = eeg.to(device, non_blocking=True)
        fnirs = fnirs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, _ = model(eeg, fnirs)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                gradient_sum += float(norm.detach())
                gradient_steps += 1
                optimizer.step()
        loss_sum += float(loss.detach()) * target.shape[0]
        probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        raw_logits.append(logits.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        indices.append(index.numpy())
    probabilities = np.concatenate(probabilities)
    raw_logits = np.concatenate(raw_logits)
    targets = np.concatenate(targets)
    indices = np.concatenate(indices)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(targets, predictions):
        confusion[int(actual), int(predicted)] += 1
    metrics = {
        "loss": loss_sum / len(targets),
        **confusion_metrics(confusion),
        "trial": aggregate_trials(probabilities, targets, indices),
    }
    if training:
        metrics["gradient_norm_mean_pre_clip"] = gradient_sum / max(gradient_steps, 1)
    prediction_rows = [
        {
            "model_index": int(index), "trial_global": int(index) // 10,
            "session": int(index) // 200, "window_in_trial": int(index) % 10,
            "target": int(target), "probability_0": float(probability[0]),
            "probability_1": float(probability[1]), "prediction": int(prediction),
            "raw_0": float(raw[0]), "raw_1": float(raw[1]), "raw_type": "logit",
        }
        for index, target, probability, prediction, raw in zip(
            indices, targets, probabilities, predictions, raw_logits
        )
    ]
    return metrics, prediction_rows


def make_model(config):
    architecture = config["architecture"]
    if architecture == "eeg_eegnet":
        return make_encoder_audit_model(architecture)
    if architecture in {"hb_mamba", "hb_mlp"}:
        return make_hb_mamba_lite(
            architecture, d_model=32, d_state=8, d_conv=4,
            fusion="concat", patch_width=8,
        )
    if architecture in V2_NAMES:
        return make_mambasta_v2(
            architecture,
            d_model=config.get("d_model", 32), d_state=config.get("d_state", 8),
            d_conv=config.get("d_conv", 4), expand=config.get("expand", 2),
            condition_rank=config.get("condition_rank", 4),
            condition_scale=config.get("condition_scale", 0.5),
        )
    if architecture in LITE_STA_NAMES:
        return make_lite_sta_alignment(
            architecture,
            d_model=config.get("d_model", 32), d_state=config.get("d_state", 8),
            d_conv=config.get("d_conv", 4), expand=config.get("expand", 2),
            condition_rank=config.get("condition_rank", 4),
            condition_scale=config.get("condition_scale", 0.5),
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def load_arrays(config):
    with np.load(config["subject_path"]) as data:
        return data["eeg"], data["fnirs"], data["label"]


def make_loader(eeg, fnirs, labels, split, name, stats, config, shuffle):
    dataset = IndexedDataset(
        eeg, fnirs, labels, split[f"{name}_indices"], stats["eeg"], stats["fnirs"]
    )
    generator = torch.Generator().manual_seed(config["model_seed"])
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle,
        generator=generator if shuffle else None, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameter_table(model):
    rows = []
    for name, module in model.named_children():
        rows.append(
            {
                "module": name,
                "parameters": sum(parameter.numel() for parameter in module.parameters()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in module.parameters() if parameter.requires_grad
                ),
            }
        )
    return rows


def normalization_hash(stats):
    payload = json.dumps(stats, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit(config, run_dir):
    start_time = time.time()
    set_model_seed(config["model_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg, fnirs, labels = load_arrays(config)
    split = subject_session_split(labels, config["test_session"], seed=config["split_seed"])
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
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    best_val = None
    history = []
    stale = 0
    for epoch in range(1, config["max_epochs"] + 1):
        train_metrics, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, config["gradient_clip"]
        )
        with torch.no_grad():
            val_metrics, _ = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[V2] epoch={epoch}/{config['max_epochs']} "
            f"train={train_metrics['loss']:.4f}/{train_metrics['accuracy']:.4f} "
            f"val={val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f}", flush=True,
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val = val_metrics
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        selected_val, val_predictions = run_epoch(model, val_loader, criterion, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    split_json = {
        key: [int(value) for value in values]
        for key, values in split.items()
    }
    normalization = {
        "eeg_mean": stats["eeg"][0], "eeg_std": stats["eeg"][1],
        "fnirs_mean": stats["fnirs"][0], "fnirs_std": stats["fnirs"][1],
    }
    normalization["sha256"] = normalization_hash(normalization)
    environment = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": parameters, "elapsed_seconds": time.time() - start_time,
    }
    metrics = {
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_val": best_val, "selected_checkpoint_val": selected_val,
        "test": None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("config.json", config), ("environment.json", environment),
        ("split.json", split_json), ("normalization.json", normalization),
        ("history.json", history), ("metrics.json", metrics),
        ("module_parameters.json", parameter_table(model)),
    ):
        (run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    write_predictions(run_dir / "validation_predictions.csv", val_predictions)
    torch.save(
        {"model_state": best_state, "config": config, "metrics": metrics},
        run_dir / "best_model.pt",
    )
    success = {
        "signature": config["signature"], "fit_complete": True,
        "test_evaluated": False, "failure": None,
    }
    (run_dir / "success.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def evaluate(config, run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"]["signature"] != config["signature"]:
        raise RuntimeError("Checkpoint signature mismatch")
    eeg, fnirs, labels = load_arrays(config)
    split = subject_session_split(labels, config["test_session"], seed=config["split_seed"])
    normalization = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
    stats = {
        "eeg": (normalization["eeg_mean"], normalization["eeg_std"]),
        "fnirs": (normalization["fnirs_mean"], normalization["fnirs_std"]),
    }
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
    print(
        f"[V2] test_loss={metrics['loss']:.4f} "
        f"test_acc={metrics['accuracy']:.4f} "
        f"test_bacc={metrics['balanced_accuracy']:.4f} "
        f"trial_acc={metrics['trial']['accuracy']:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    if args.mode == "fit":
        fit(config, run_dir)
    else:
        evaluate(config, run_dir)


if __name__ == "__main__":
    main()
