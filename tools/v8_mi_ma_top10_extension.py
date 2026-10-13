"""Validation-selected MI/MA Top10 extension for the frozen V8 J1_R4 model.

This extension does not alter the original V8 selection.  It ranks all 29
subjects from B_M50 validation metrics only, freezes two Top10 lists, fits the
already-frozen J1_R4 recipe only where the original VP001-VP010 grid cannot be
reused, freezes before test access, exports those tests, and aggregates them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mambasta_v8 import V8JointModel
from training.train_v2 import make_loader, scalar_stats, set_model_seed
from training.train_v4_fusion import load_data
from tools import v5_all_subjects as legacy
from tools.v8_finalize import _evaluate_one

BASE = ROOT / "experiments" / "mambasta_v8_final_joint_20261012"
OUT = ROOT / "experiments" / "mambasta_v8_mi_ma_top10_extension_20261013"
CONFIG = ROOT / "configs" / "mambasta_v8_final_joint.yaml"
TASKS = ("mi", "ma")
SUBJECTS = tuple(f"VP{i:03d}" for i in range(1, 30))
SESSIONS = (0, 1, 2)
SEEDS = (42, 123, 2026)
PAPER = {"mi": 0.6965, "ma": 0.8514}


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    payload = {key: item for key, item in value.items() if key != "signature"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def require_space(action):
    free = shutil.disk_usage(ROOT.anchor).free / 2**30
    if free < 100:
        raise RuntimeError(f"STOP_DISK below 100 GiB before {action}: {free:.2f}")
    return free


def complete(run, signature, allow_test=False):
    required = ("success.json", "best_model.pt", "validation_predictions.csv", "metrics.json")
    if any(not (run / name).is_file() for name in required):
        return False
    if not allow_test and (run / "test_predictions.csv").exists():
        return False
    try:
        status = json.loads((run / "success.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("signature") == signature and status.get("fit_complete") is True


def original_v8_run(task, subject, session, seed):
    fixed = (int(subject[-3:]) - 1) % 3
    stage = ("s2" if session == fixed else "s3") if seed == 42 else "s5"
    return BASE / stage / "J1_R4" / task / f"{subject}_session{session}_seed{seed}"


def extension_baseline_run(role, seed, task, subject, session):
    return OUT / "baseline_completion" / role / f"seed{seed}" / task / f"{subject}_session{session}"


def extension_run(task, subject, session, seed):
    return OUT / "fits" / "J1_R4" / task / f"{subject}_session{session}_seed{seed}"


def combined_baseline_index():
    index = {}
    historical = legacy.complete_index()
    for task in TASKS:
        for subject in SUBJECTS:
            for seed in SEEDS:
                for session in SESSIONS:
                    key = ("B_M50", seed, task, subject, session)
                    if int(subject[-3:]) <= 10:
                        run, cfg = historical[key]
                    else:
                        cfg = BASE / "_configs" / "s1_baselines" / "B_M50" / f"seed{seed}" / task / f"{subject}_session{session}.json"
                        run = BASE / "s1_baselines" / "B_M50" / f"seed{seed}" / task / f"{subject}_session{session}"
                        if not cfg.is_file():
                            cfg = OUT / "_configs" / "baseline_completion" / "B_M50" / f"seed{seed}" / task / f"{subject}_session{session}.json"
                            run = extension_baseline_run("B_M50", seed, task, subject, session)
                    if not cfg.is_file():
                        raise RuntimeError(f"missing B_M50 config {key}")
                    value = json.loads(cfg.read_text(encoding="utf-8"))
                    if not legacy.complete_fit(run, value["signature"]):
                        raise RuntimeError(f"incomplete B_M50 fold {key}: {run}")
                    index[key] = (run, cfg, value)
    return index


def baseline_completion():
    """Complete only MA VP028/VP029 E32+B_M50 folds needed for ranking."""
    study = yaml.safe_load((ROOT / "configs" / "mambasta_v5_final.yaml").read_text(encoding="utf-8"))
    app = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ledger_path = OUT / "baseline_budget_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "hard_cap": 36, "attempted": 0, "successful": 0, "failed": 0, "records": []
    }
    for subject in ("VP028", "VP029"):
        require_space(f"baseline cache ma/{subject}")
        fnirs, eeg_by_session = legacy.ensure_caches(study, "ma", subject)
        for seed in SEEDS:
            for session in SESSIONS:
                eeg = eeg_by_session[session]
                teacher = extension_baseline_run("E32", seed, "ma", subject, session)
                teacher_cfg = legacy.make_teacher_config("ma", subject, session, seed, eeg[0], eeg[1])
                teacher_cfg.update({"study": "mambasta_v8_mi_ma_top10_extension_20261013", "phase": "top10_baseline_completion", "mamba_python": app["paths"]["mamba_python"], "selection_source": "validation_only", "test_access": False})
                teacher_cfg["extension_runner_sha256"] = sha(Path(__file__))
                teacher_cfg["signature"] = canonical(teacher_cfg)
                _execute_baseline("E32", teacher_cfg, teacher, ledger, ledger_path)
                child = legacy.make_child_config("B_M50", "ma", subject, session, seed, fnirs, eeg, teacher)
                child.update({"study": "mambasta_v8_mi_ma_top10_extension_20261013", "phase": "top10_baseline_completion", "mamba_python": app["paths"]["mamba_python"], "selection_source": "validation_only", "test_access": False})
                child["extension_runner_sha256"] = sha(Path(__file__))
                child["signature"] = canonical(child)
                _execute_baseline("B_M50", child, extension_baseline_run("B_M50", seed, "ma", subject, session), ledger, ledger_path)
    write_json(OUT / "baseline_complete.json", {"status": "complete", "fits": 36, "failed": ledger["failed"], "test_outputs": 0, "free_gib": require_space("baseline completion"), "completed_at": now()})


def _execute_baseline(role, config, run, ledger, ledger_path):
    cfg = OUT / "_configs" / "baseline_completion" / role / f"seed{config['model_seed']}" / config["task"] / f"{config['subject']}_session{config['test_session']}.json"
    if complete(run, config["signature"]):
        return
    if ledger["attempted"] >= ledger["hard_cap"]:
        raise RuntimeError("baseline extension hard cap reached")
    write_json(cfg, config)
    trainer = ROOT / "training" / ("train_v3.py" if role == "E32" else "train_v5.py")
    record = {"role": role, "task": config["task"], "subject": config["subject"], "session": config["test_session"], "seed": config["model_seed"], "status": "running", "started_at": now()}
    ledger["attempted"] += 1
    ledger["records"].append(record)
    write_json(ledger_path, ledger)
    print(f"[V8-TOP10-BASE] {role} {config['task']} {config['subject']} s{config['test_session']} seed{config['model_seed']}", flush=True)
    code = subprocess.run([config["mamba_python"], str(trainer), "--mode", "fit", "--config", str(cfg), "--run-dir", str(run)], cwd=ROOT).returncode
    record.update({"status": "successful" if code == 0 and complete(run, config["signature"]) else "failed", "ended_at": now(), "returncode": code})
    if record["status"] == "successful":
        ledger["successful"] += 1
    else:
        ledger["failed"] += 1
        write_json(ledger_path, ledger)
        raise RuntimeError(f"baseline fit failed: {run}")
    write_json(ledger_path, ledger)


def select_top10():
    if not (OUT / "baseline_complete.json").is_file():
        raise RuntimeError("baseline completion missing")
    index = combined_baseline_index()
    rankings = {}
    for task in TASKS:
        rows = []
        for subject in SUBJECTS:
            values = []
            for seed in SEEDS:
                for session in SESSIONS:
                    run, _, _ = index[("B_M50", seed, task, subject, session)]
                    metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["selected_checkpoint_val"]
                    values.append(float(metric["accuracy"]))
            rows.append({"subject": subject, "validation_window_accuracy": float(np.mean(values)), "folds": 9})
        rows.sort(key=lambda row: (-row["validation_window_accuracy"], row["subject"]))
        rankings[task] = rows
    selected = {task: [row["subject"] for row in rankings[task][:10]] for task in TASKS}
    payload = {"status": "frozen", "selection_source": "B_M50_validation_only", "rule": "Top10 by mean B_M50 validation window accuracy over 3 sessions x 3 seeds; ties by subject ID", "subjects_by_task": selected, "rankings": rankings, "test_read": False, "selection_changed_by_test": False, "created_at": now()}
    write_json(OUT / "top10_selection_freeze.json", payload)
    with (OUT / "validation_rankings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task", "rank", "subject", "validation_window_accuracy", "selected"))
        writer.writeheader()
        for task in TASKS:
            for rank, row in enumerate(rankings[task], 1):
                writer.writerow({"task": task, "rank": rank, "subject": row["subject"], "validation_window_accuracy": row["validation_window_accuracy"], "selected": rank <= 10})
    return payload


def _build_cache(task, subject, session, seed, index):
    path = OUT / "feature_cache" / task / f"{subject}_session{session}_seed{seed}.npz"
    manifest = path.with_suffix(".manifest.json")
    b_run, _, b_cfg = index[("B_M50", seed, task, subject, session)]
    identity = {"task": task, "subject": subject, "session": session, "seed": seed, "b_m50_checkpoint_sha256": sha(b_run / "best_model.pt"), "r_cache_policy": "alias_because_J1_R4_never_reads_r", "adapter_code_sha256": sha(ROOT / "models" / "mambasta_v8.py"), "test_access": False}
    identity["signature"] = canonical(identity)
    if path.is_file() and manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature") == identity["signature"] and old.get("complete") is True:
            return path, old
        raise RuntimeError(f"stale extension cache: {path}")
    fh, eh, eeg, fnirs, labels, split = load_data(b_cfg)
    stats = {"eeg": scalar_stats(eeg, split["train_indices"]), "fnirs": scalar_stats(fnirs, split["train_indices"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_model_seed(seed)
    checkpoint = torch.load(b_run / "best_model.pt", map_location="cpu", weights_only=False)
    model = V8JointModel({"config_id": "J1_R4", "d_model": 32, "d_state": 8, "d_conv": 4, "expand": 2, "query_rank": 8, "dropout": .2}).to(device)
    model.load_b_m50_state(checkpoint["model_state"])
    model.freeze_backbone()
    model.eval()
    arrays = {}
    for name in ("train", "val"):
        loader = make_loader(eeg, fnirs, labels, split, name, stats, b_cfg, False)
        pieces = {key: [] for key in ("e", "n", "q", "m", "w", "b", "y", "i")}
        with torch.no_grad():
            for ee, nn_, yy, ii in loader:
                features = model.extract_features(ee.to(device), nn_.to(device), include_teacher_probability=True)
                values = {"e": features.e_main, "n": features.n_main, "q": features.e_query, "m": features.memory, "w": features.base_weights, "b": features.b_m50_log_probability, "y": yy, "i": ii}
                for key, value in values.items():
                    pieces[key].append(value.detach().cpu().numpy())
        for key, value in pieces.items():
            arrays[f"{name}_{key}"] = np.concatenate(value)
        arrays[f"{name}_r"] = arrays[f"{name}_b"].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    identity.update({"complete": True, "cache_sha256": sha(path), "train_windows": len(arrays["train_y"]), "val_windows": len(arrays["val_y"]), "test_windows": 0})
    write_json(manifest, identity)
    fh.close(); eh.close()
    return path, identity


def fit_selected():
    freeze = json.loads((OUT / "top10_selection_freeze.json").read_text(encoding="utf-8"))
    index = combined_baseline_index()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ledger_path = OUT / "fit_budget_ledger.json"
    selected_new = [(task, subject) for task in TASKS for subject in freeze["subjects_by_task"][task] if int(subject[-3:]) > 10]
    hard_cap = len(selected_new) * 9
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"hard_cap": hard_cap, "attempted": 0, "successful": 0, "failed": 0, "records": []}
    if ledger["hard_cap"] != hard_cap:
        raise RuntimeError("extension fit hard cap changed")
    for task, subject in selected_new:
        for session in SESSIONS:
            for seed in SEEDS:
                require_space(f"J1_R4 {task}/{subject}/s{session}/seed{seed}")
                cache, manifest = _build_cache(task, subject, session, seed, index)
                b_run, _, b_cfg = index[("B_M50", seed, task, subject, session)]
                run = extension_run(task, subject, session, seed)
                fold = {**config["model"], "study": "mambasta_v8_mi_ma_top10_extension_20261013", "stage": "frozen_J1_R4_top10_extension", "config_id": "J1_R4", "task": task, "subject": subject, "test_session": session, "model_seed": seed, "feature_cache_path": str(cache.resolve()), "feature_cache_signature": manifest["signature"], "b_m50_checkpoint": str((b_run / "best_model.pt").resolve()), "b_m50_signature": b_cfg["signature"], "max_epochs": 80, "patience": 10, "batch_size": 64, "learning_rate": config["training"]["head_lr"], "weight_decay": config["training"]["weight_decay"], "gradient_clip": config["training"]["gradient_clip"], "kd_weight": config["training"]["kd_weight"], "interaction_penalty": config["training"]["interaction_penalty"], "temperature": config["training"]["temperature"], "selection_source": "frozen_model_validation_selected_subjects", "test_access": False, "top10_selection_freeze_sha256": sha(OUT / "top10_selection_freeze.json"), "trainer_sha256": sha(ROOT / "training" / "train_v8_cached.py"), "model_sha256": sha(ROOT / "models" / "mambasta_v8.py")}
                fold["signature"] = canonical(fold)
                cfg = OUT / "_configs" / "fits" / "J1_R4" / task / f"{subject}_session{session}_seed{seed}.json"
                if complete(run, fold["signature"]):
                    continue
                if ledger["attempted"] >= ledger["hard_cap"]:
                    raise RuntimeError("extension fit hard cap reached")
                write_json(cfg, fold)
                record = {"task": task, "subject": subject, "session": session, "seed": seed, "signature": fold["signature"], "status": "running", "started_at": now()}
                ledger["attempted"] += 1; ledger["records"].append(record); write_json(ledger_path, ledger)
                print(f"[V8-TOP10-FIT] J1_R4 {task} {subject} s{session} seed{seed}", flush=True)
                code = subprocess.run([config["paths"]["mamba_python"], str(ROOT / "training" / "train_v8_cached.py"), "--config", str(cfg), "--run-dir", str(run)], cwd=ROOT).returncode
                record.update({"status": "successful" if code == 0 and complete(run, fold["signature"]) else "failed", "ended_at": now(), "returncode": code})
                if record["status"] == "successful": ledger["successful"] += 1
                else:
                    ledger["failed"] += 1; write_json(ledger_path, ledger); raise RuntimeError(f"extension fit failed: {run}")
                write_json(ledger_path, ledger)
    write_json(OUT / "fit_complete.json", {"status": "complete", "selected_new_entries": selected_new, "new_fits": hard_cap, "reused_original_v8_folds": sum(int(subject[-3:]) <= 10 for task in TASKS for subject in freeze["subjects_by_task"][task]) * 9, "failed": ledger["failed"], "test_outputs": 0, "completed_at": now()})


def _selected_run(task, subject, session, seed):
    return original_v8_run(task, subject, session, seed) if int(subject[-3:]) <= 10 else extension_run(task, subject, session, seed)


def freeze_before_test():
    if not (OUT / "fit_complete.json").is_file():
        raise RuntimeError("fit completion missing")
    selection = json.loads((OUT / "top10_selection_freeze.json").read_text(encoding="utf-8"))
    rows = []
    for task in TASKS:
        for subject in selection["subjects_by_task"][task]:
            for session in SESSIONS:
                for seed in SEEDS:
                    run = _selected_run(task, subject, session, seed)
                    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
                    if not complete(run, cfg["signature"], allow_test=int(subject[-3:]) <= 10):
                        raise RuntimeError(f"incomplete selected fold: {run}")
                    rows.append({"task": task, "subject": subject, "session": session, "seed": seed, "run_dir": str(run.resolve()), "signature": cfg["signature"], "source": "original_v8_reuse" if int(subject[-3:]) <= 10 else "extension_new"})
    payload = {"status": "frozen_before_test", "selection_source": "B_M50_validation_only", "subjects_by_task": selection["subjects_by_task"], "folds": len(rows), "rows": rows, "selection_changed_by_test": False, "created_at": now()}
    write_json(OUT / "before_test_freeze.json", payload)


def evaluate():
    freeze = json.loads((OUT / "before_test_freeze.json").read_text(encoding="utf-8"))
    records = []
    ledger = OUT / "test_export_ledger.json"
    if ledger.is_file(): records = json.loads(ledger.read_text(encoding="utf-8"))["records"]
    done = {(x["task"], x["subject"], x["session"], x["seed"]) for x in records if x["status"] == "successful"}
    reused = 0; new = 0
    for row in freeze["rows"]:
        key = (row["task"], row["subject"], row["session"], row["seed"])
        run = Path(row["run_dir"])
        if row["source"] == "original_v8_reuse":
            if not (run / "test_predictions.csv").is_file(): raise RuntimeError(f"missing original V8 test: {run}")
            reused += 1; continue
        if key not in done:
            require_space(f"test export {key}")
            print(f"[V8-TOP10-TEST] {key}", flush=True)
            _evaluate_one(run)
            records.append({"task": key[0], "subject": key[1], "session": key[2], "seed": key[3], "status": "successful", "completed_at": now()})
            write_json(ledger, {"records": records, "updated_at": now()})
        new += 1
    write_json(OUT / "test_export_complete.json", {"status": "complete", "new_exports": new, "reused_original_v8": reused, "failed": 0, "selection_changed_by_test": False, "completed_at": now()})


def _prediction_metric(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    p = np.array([[float(x["probability_0"]), float(x["probability_1"])] for x in rows])
    y = np.array([int(x["target"]) for x in rows])
    trial = np.array([int(x["trial_global"]) for x in rows])
    window = float((p.argmax(1) == y).mean())
    trial_values = []
    for value in np.unique(trial):
        q = trial == value
        trial_values.append(float(p[q].mean(0).argmax() == y[q][0]))
    return window, float(np.mean(trial_values))


def aggregate():
    if not (OUT / "test_export_complete.json").is_file(): raise RuntimeError("test export incomplete")
    selection = json.loads((OUT / "top10_selection_freeze.json").read_text(encoding="utf-8"))
    subject_rows = []
    for task in TASKS:
        for subject in selection["subjects_by_task"][task]:
            windows = []; trials = []
            for session in SESSIONS:
                for seed in SEEDS:
                    w, t = _prediction_metric(_selected_run(task, subject, session, seed) / "test_predictions.csv")
                    windows.append(w); trials.append(t)
            subject_rows.append({"task": task, "subject": subject, "window_accuracy": float(np.mean(windows)), "trial_accuracy": float(np.mean(trials))})
    summaries = []
    for task in TASKS:
        rows = [x for x in subject_rows if x["task"] == task]
        window = float(np.mean([x["window_accuracy"] for x in rows])); trial = float(np.mean([x["trial_accuracy"] for x in rows]))
        summaries.append({"task": task, "n_subjects": 10, "window_accuracy": window, "trial_accuracy": trial, "sta_net_paper_reference": PAPER[task], "window_minus_sta_net_pp": (window - PAPER[task]) * 100})
    with (OUT / "top10_subject_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(subject_rows[0])); writer.writeheader(); writer.writerows(subject_rows)
    with (OUT / "top10_task_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    lines = ["# V8 MI/MA Validation-Selected Top10 Extension", "", "The frozen V8 J1_R4 model/recipe was not reselected. Subjects were ranked using B_M50 validation only; tests were exported only after the lists and all required fits were frozen.", "", "| Task | Subjects | Window ACC | Trial ACC | STA-Net paper | Window delta |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        subjects = ", ".join(selection["subjects_by_task"][row["task"]])
        lines.append(f"| {row['task'].upper()} | {subjects} | {row['window_accuracy']:.2%} | {row['trial_accuracy']:.2%} | {row['sta_net_paper_reference']:.2%} | {row['window_minus_sta_net_pp']:+.2f} pp |")
    lines += ["", "STA-Net references are paper-level full-cohort subject-specific window accuracies, not paired Top10 replications.", "The extension is post-selection descriptive and does not revise the original V8 model-selection conclusion.", ""]
    (OUT / "V8_MI_MA_Top10_Extension_Report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "execution_complete.json", {"status": "complete", "summaries": summaries, "subjects_by_task": selection["subjects_by_task"], "selection_changed_by_test": False, "free_gib": require_space("extension completion"), "completed_at": now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "select", "fit", "freeze", "evaluate", "aggregate", "all"), required=True)
    args = parser.parse_args()
    stages = ("baseline", "select", "fit", "freeze", "evaluate", "aggregate") if args.stage == "all" else (args.stage,)
    actions = {"baseline": baseline_completion, "select": select_top10, "fit": fit_selected, "freeze": freeze_before_test, "evaluate": evaluate, "aggregate": aggregate}
    for stage in stages:
        process = OUT / f"{stage}_process.json"
        write_json(process, {"pid": __import__("os").getpid(), "stage": stage, "status": "running", "started_at": now()})
        try:
            actions[stage]()
        except Exception as error:
            write_json(process, {"pid": __import__("os").getpid(), "stage": stage, "status": "failed", "error": repr(error), "ended_at": now()})
            raise
        write_json(process, {"pid": __import__("os").getpid(), "stage": stage, "status": "complete", "ended_at": now()})


if __name__ == "__main__":
    main()
