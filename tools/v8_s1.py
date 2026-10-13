"""Complete and audit the frozen V8 B_M50/E32/R_U validation baselines."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tools import v5_all_subjects as legacy

EXP = ROOT / "experiments" / "mambasta_v8_final_joint_20261012"
CONFIG = ROOT / "configs" / "mambasta_v8_final_joint.yaml"
TASKS = ("mi", "ma")
SUBJECTS = tuple(f"VP{i:03d}" for i in range(11, 30))
SESSIONS = (0, 1, 2)
SEEDS = (42, 123, 2026)
ROLES = ("E32", "B_M50", "R_U")
EXPECTED_NEW = len(TASKS) * len(SUBJECTS) * len(SESSIONS) * len(SEEDS) * len(ROLES)
EXPECTED_REUSED = 1242


def now(): return datetime.now().astimezone().isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def require_space(config, action):
    free = shutil.disk_usage(ROOT.anchor).free / 2**30
    if free < config["disk_stop_threshold_gib"]:
        raise RuntimeError(f"STOP_DISK_BELOW_{config['disk_stop_threshold_gib']}_GIB before {action}: {free:.3f} GiB")
    return free


def run_dir(role, seed, task, subject, session):
    return EXP / "s1_baselines" / role / f"seed{seed}" / task / f"{subject}_session{session}"


def config_path(role, seed, task, subject, session):
    return EXP / "_configs" / "s1_baselines" / role / f"seed{seed}" / task / f"{subject}_session{session}.json"


def complete(path, signature):
    required = ("success.json", "best_model.pt", "validation_predictions.csv", "metrics.json", "normalization.json")
    if any(not (path/name).is_file() for name in required) or (path/"test_predictions.csv").exists(): return False
    try: status = json.loads((path/"success.json").read_text(encoding="utf-8"))
    except Exception: return False
    return status.get("signature") == signature and status.get("fit_complete") is True and status.get("test_evaluated") is False


def v8_signature(config):
    config["study"] = "mambasta_v8_final_joint_20261012"
    config["selection_source"] = "validation_only"
    config["test_access"] = False
    config["v8_s1_runner_sha256"] = legacy.sha256_file(Path(__file__))
    config["signature"] = legacy.canonical(config)
    return config


def audit_reuse():
    index = legacy.complete_index()
    rows = []
    for key, (run, cfg) in sorted(index.items()):
        role, seed, task, subject, session = key
        if role not in ROLES: continue
        config = json.loads(cfg.read_text(encoding="utf-8"))
        if not legacy.complete_fit(run, config["signature"]):
            raise RuntimeError(f"incomplete historical baseline: {run}")
        rows.append({"role": role, "seed": seed, "task": task, "subject": subject,
                     "session": session, "run_dir": str(run.resolve()),
                     "signature": config["signature"], "validation_only_reuse": True})
    if len(rows) != EXPECTED_REUSED:
        raise RuntimeError(f"historical reuse mismatch: {len(rows)} != {EXPECTED_REUSED}")
    write_json(EXP/"s1_historical_reuse_registry.json", {"count": len(rows), "per_role": 414, "test_read": False, "rows": rows})
    return rows


def load_ledger():
    path = EXP / "budget_ledger.json"
    default = {"optimization_fit_hard_cap": 6500, "attempted": 0, "successful": 0,
               "failed": 0, "reused": 0, "records": []}
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    return path, ledger


def execute(config, role, ledger, ledger_path):
    path = run_dir(role, config["model_seed"], config["task"], config["subject"], config["test_session"])
    cfg = config_path(role, config["model_seed"], config["task"], config["subject"], config["test_session"])
    if complete(path, config["signature"]): return False
    if ledger["attempted"] >= ledger["optimization_fit_hard_cap"]:
        raise RuntimeError("V8 optimization fit hard cap reached")
    write_json(cfg, config)
    trainer = ROOT / "training" / ("train_v3.py" if role == "E32" else "train_v5.py" if role == "B_M50" else "train_v4_fusion.py")
    record = {"stage": "s1", "role": role, "task": config["task"], "subject": config["subject"],
              "test_session": config["test_session"], "model_seed": config["model_seed"],
              "signature": config["signature"], "status": "running", "started_at": now()}
    ledger["attempted"] += 1; ledger["records"].append(record); write_json(ledger_path, ledger)
    print(f"[V8-S1] {role} seed={config['model_seed']} {config['task']} {config['subject']} s{config['test_session']}", flush=True)
    code = subprocess.run([str(config["mamba_python"]), str(trainer), "--mode", "fit", "--config", str(cfg), "--run-dir", str(path)], cwd=ROOT).returncode
    record.update({"ended_at": now(), "returncode": code})
    if code == 0 and complete(path, config["signature"]):
        record["status"] = "successful"; ledger["successful"] += 1
    else:
        record["status"] = "failed"; ledger["failed"] += 1; write_json(ledger_path, ledger)
        raise RuntimeError(f"V8 S1 fit failed: {path}")
    write_json(ledger_path, ledger); return True


def fit():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require_space(config, "S1 initialization")
    if not (EXP/"s0_contracts.json").exists() or not json.loads((EXP/"s0_contracts.json").read_text(encoding="utf-8"))["all_required_passed"]:
        raise RuntimeError("S0 contracts must pass before S1")
    reused = audit_reuse()
    ledger_path, ledger = load_ledger()
    if ledger["reused"] == 0: ledger["reused"] = len(reused); write_json(ledger_path, ledger)
    elif ledger["reused"] != len(reused): raise RuntimeError("reuse ledger changed")
    study = yaml.safe_load((ROOT/"configs/mambasta_v5_final.yaml").read_text(encoding="utf-8"))
    for task in TASKS:
        for subject in SUBJECTS:
            require_space(config, f"cache {task}/{subject}")
            fnirs, eeg_by_session = legacy.ensure_caches(study, task, subject)
            for seed in SEEDS:
                for session in SESSIONS:
                    require_space(config, f"fit {task}/{subject}/s{session}/seed{seed}")
                    eeg = eeg_by_session[session]
                    teacher = run_dir("E32", seed, task, subject, session)
                    teacher_config = legacy.make_teacher_config(task, subject, session, seed, eeg[0], eeg[1])
                    teacher_config.update({"phase": "v8_s1_baseline", "mamba_python": config["paths"]["mamba_python"]})
                    v8_signature(teacher_config)
                    execute(teacher_config, "E32", ledger, ledger_path)
                    for role in ("B_M50", "R_U"):
                        child = legacy.make_child_config(role, task, subject, session, seed, fnirs, eeg, teacher)
                        child.update({"phase": "v8_s1_baseline", "mamba_python": config["paths"]["mamba_python"]})
                        v8_signature(child)
                        execute(child, role, ledger, ledger_path)
    successes = list((EXP/"s1_baselines").glob("**/success.json"))
    tests = list((EXP/"s1_baselines").glob("**/test_predictions.csv"))
    if len(successes) != EXPECTED_NEW or tests:
        raise RuntimeError(f"S1 completion invariant failed successes={len(successes)} tests={len(tests)}")
    write_json(EXP/"s1_fit_complete.json", {"status":"complete", "new_fits":EXPECTED_NEW,
               "historical_reuses":EXPECTED_REUSED, "failures":ledger["failed"], "test_outputs":0,
               "free_gib":require_space(config,"S1 completion"), "completed_at":now()})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage", choices=("fit",), required=True); parser.parse_args()
    process = EXP/"s1_process.json"
    write_json(process,{"pid":os.getpid(),"stage":"s1","status":"running","started_at":now()})
    try:
        fit()
    except Exception as error:
        write_json(process,{"pid":os.getpid(),"stage":"s1","status":"failed","error":repr(error),"ended_at":now()})
        raise
    write_json(process,{"pid":os.getpid(),"stage":"s1","status":"complete","ended_at":now()})


if __name__ == "__main__": main()
