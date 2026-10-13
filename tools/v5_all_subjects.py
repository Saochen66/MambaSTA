"""Run the frozen V5 model and internal controls on every locally available subject.

The command is deliberately split into fit/freeze/evaluate stages so that no test
prediction can be produced before the complete fitting grid is sealed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = ROOT / "configs/mambasta_v5_final.yaml"
EXP = ROOT / "experiments/mambasta_v5_final_20261006"
V3 = ROOT / "experiments/mambasta_v3_20261004"
V4 = ROOT / "experiments/mambasta_v4_alignment_20261005"
SEEDS = (42, 123, 2026)
SESSIONS = (0, 1, 2)
ROLES = ("E32", "B_M50", "R_U")


def now(): return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(payload) -> str:
    value = {key: payload[key] for key in sorted(payload) if key != "signature"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def available_entries():
    with (EXP / "all_subject_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = [(row["task"], row["subject"]) for row in rows if row["status"] == "available"]
    if len(values) != 46 or len(set(values)) != 46:
        raise RuntimeError(f"expected 46 unique locally available entries, got {len(values)}")
    return values


def code_hashes(paths): return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def validate_eeg_cache(task, subject, session):
    path = V3 / "strict_cache" / task / subject / f"session{session}.npz"
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists(): return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        embedded = str(data["preprocess_hash"].item())
    if embedded != manifest["preprocess_hash"]: raise RuntimeError(f"EEG cache signature mismatch: {path}")
    return path, embedded


def validate_fnirs_cache(task, subject):
    path = V4 / "strict_cache" / "F2" / task / f"{subject}.npz"
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists(): return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        embedded = str(data["complete_sha256"].item())
    if embedded != manifest["complete_sha256"]: raise RuntimeError(f"fNIRS cache signature mismatch: {path}")
    return path, embedded


def ensure_caches(study, task, subject):
    fnirs = validate_fnirs_cache(task, subject)
    if fnirs is None:
        command = [study["paths"]["preprocessing_python"], str(ROOT / "tools/build_v4_fnirs_subject.py"), "--task", task, "--subject", subject, "--version", "F2"]
        if subprocess.run(command, cwd=ROOT).returncode: raise RuntimeError(f"fNIRS preprocessing failed: {task}/{subject}")
        fnirs = validate_fnirs_cache(task, subject)
    eeg = {}
    for session in SESSIONS:
        item = validate_eeg_cache(task, subject, session)
        if item is None:
            command = [study["paths"]["preprocessing_python"], str(ROOT / "tools/build_strict_v3_fold.py"), "--task", task, "--subject", subject, "--test-session", str(session)]
            if subprocess.run(command, cwd=ROOT).returncode: raise RuntimeError(f"EEG preprocessing failed: {task}/{subject}/s{session}")
            item = validate_eeg_cache(task, subject, session)
        eeg[session] = item
    return fnirs, eeg


def complete_fit(run_dir: Path, signature: str):
    needed = ("success.json", "best_model.pt", "validation_predictions.csv", "metrics.json", "normalization.json")
    if any(not (run_dir / name).exists() for name in needed): return False
    try: success = json.loads((run_dir / "success.json").read_text(encoding="utf-8"))
    except Exception: return False
    return success.get("signature") == signature and success.get("fit_complete") is True


def make_teacher_config(task, subject, session, seed, eeg_path, preprocess_hash):
    template = json.loads((EXP / "confirm_teachers/seed123/wg/VP001_session0/config.json").read_text(encoding="utf-8"))
    config = {key: template[key] for key in ("architecture", "split_seed", "max_epochs", "patience", "batch_size", "learning_rate", "weight_decay", "gradient_clip", "d_model", "d_state", "d_conv", "expand", "query_rank", "dropout")}
    config.update({"study": "mambasta_v5_final_20261006", "phase": "all_subject_teacher", "task": task, "subject": subject, "test_session": session, "cache_path": str(eeg_path), "preprocess_hash": preprocess_hash, "model_seed": seed, "selection_source": "frozen_model_full_subject_run", "test_access": False, "code_sha256": code_hashes([ROOT/"training/train_v3.py", ROOT/"training/train_v2.py", ROOT/"models/mambasta_v3.py", ROOT/"models/encoder_audit.py", ROOT/"models/lightweight_baseline.py", ROOT/"models/mamba_sta_v2.py", ROOT/"models/time_provenance_adapter.py"])})
    config["signature"] = canonical(config); return config


def make_child_config(role, task, subject, session, seed, fnirs, eeg, teacher_dir):
    source = EXP / "confirm_seeds" / role / "seed123/wg/VP001_session0/config.json"
    template = json.loads(source.read_text(encoding="utf-8"))
    keep = ("split_seed", "max_epochs", "patience", "batch_size", "learning_rate", "weight_decay", "gradient_clip", "d_model", "d_state", "d_conv", "expand", "query_rank", "dropout")
    config = {key: template[key] for key in keep}
    teacher_config = json.loads((teacher_dir/"config.json").read_text(encoding="utf-8")); teacher_checkpoint = teacher_dir/"best_model.pt"
    config.update({"study": "mambasta_v5_final_20261006", "phase": "all_subject_frozen", "config_id": role, "task": task, "subject": subject, "test_session": session, "model_seed": seed, "fnirs_cache_path": str(fnirs[0]), "fnirs_complete_sha256": fnirs[1], "eeg_cache_path": str(eeg[0]), "teacher_checkpoint": str(teacher_checkpoint), "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint), "teacher_run_signature": teacher_config["signature"], "selection_source": "frozen_model_full_subject_run", "test_access": False})
    if role == "B_M50": config.update({"family": "probability", "mixer": "mamba", "alpha": 0.5, "code_sha256": code_hashes([ROOT/"training/train_v5.py", ROOT/"training/train_v2.py", ROOT/"models/mambasta_v5.py", ROOT/"models/v5_heads.py", ROOT/"models/v5_delta.py"])})
    else: config.update({"architecture": "r_u", "code_sha256": code_hashes([ROOT/"training/train_v4_fusion.py", ROOT/"training/train_v2.py", ROOT/"models/mambasta_v4_fusion.py"])})
    config["signature"] = canonical(config); return config


def new_run(role, seed, task, subject, session): return EXP/"all_subject_new"/role/f"seed{seed}"/task/f"{subject}_session{session}"
def new_config(role, seed, task, subject, session): return EXP/"_configs/all_subject_new"/role/f"seed{seed}"/task/f"{subject}_session{session}.json"


def load_ledger(study):
    path = EXP / "full_subject_budget_ledger.json"
    return path, (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"hard_cap": study["full_subject_fit_cap_without_refit"], "attempted": 0, "successful": 0, "failed": 0, "records": []})


def execute_fit(study, role, config, run_dir, config_path, ledger, ledger_path):
    if complete_fit(run_dir, config["signature"]): return
    if ledger["attempted"] >= ledger["hard_cap"]: raise RuntimeError("full-subject fit hard cap reached")
    write_json(config_path, config); ledger["attempted"] += 1
    record = {"stage": "all_subject_fit", "role": role, "seed": config["model_seed"], "task": config["task"], "subject": config["subject"], "test_session": config["test_session"], "signature": config["signature"], "status": "running", "started_at": now()}
    ledger["records"].append(record); write_json(ledger_path, ledger)
    trainer = ROOT/"training"/("train_v3.py" if role == "E32" else "train_v5.py" if role == "B_M50" else "train_v4_fusion.py")
    command = [study["paths"]["mamba_python"], str(trainer), "--mode", "fit", "--config", str(config_path), "--run-dir", str(run_dir)]
    print(f"[V5-all-fit] {role} seed={config['model_seed']} {config['task']} {config['subject']} s{config['test_session']}", flush=True)
    code = subprocess.run(command, cwd=ROOT).returncode; record["ended_at"] = now()
    if code == 0 and complete_fit(run_dir, config["signature"]): record["status"] = "successful"; ledger["successful"] += 1
    else: record["status"] = "failed"; record["returncode"] = code; ledger["failed"] += 1; write_json(ledger_path, ledger); raise RuntimeError(f"fit failed: {run_dir}")
    write_json(ledger_path, ledger)


def fit_new_subjects(study):
    ledger_path, ledger = load_ledger(study)
    new_entries = [(task, subject) for task, subject in available_entries() if int(subject[-3:]) > 10]
    if new_entries != [("wg", f"VP{number:03d}") for number in range(11, 27)]: raise RuntimeError(f"unexpected non-development grid: {new_entries}")
    for task, subject in new_entries:
        fnirs, eeg_by_session = ensure_caches(study, task, subject)
        for seed in SEEDS:
            for session in SESSIONS:
                eeg = eeg_by_session[session]
                teacher_dir = new_run("E32", seed, task, subject, session)
                teacher_cfg = make_teacher_config(task, subject, session, seed, eeg[0], eeg[1])
                execute_fit(study, "E32", teacher_cfg, teacher_dir, new_config("E32", seed, task, subject, session), ledger, ledger_path)
                for role in ("B_M50", "R_U"):
                    config = make_child_config(role, task, subject, session, seed, fnirs, eeg, teacher_dir)
                    execute_fit(study, role, config, new_run(role, seed, task, subject, session), new_config(role, seed, task, subject, session), ledger, ledger_path)
    write_json(EXP/"all_subject_new_fit_complete.json", {"status": "complete", "entries": 16, "fits": 432, "roles": list(ROLES), "seeds": list(SEEDS), "sessions": list(SESSIONS), "test_outputs": 0, "completed_at": now()})


def existing_index():
    specs = {
        "E32": [V3/"p1/e32", V3/"p3/e32", EXP/"confirm_teachers"],
        "B_M50": [EXP/"screen/w1/B_M50", EXP/"complete_sessions/B_M50", EXP/"confirm_seeds/B_M50"],
        "R_U": [V4/"s2/r_u", V4/"s4/r_u", EXP/"confirm_seeds/R_U"],
    }
    index = {}
    for role, roots in specs.items():
        for root in roots:
            for path in root.rglob("config.json"):
                config = json.loads(path.read_text(encoding="utf-8")); key = (role, int(config["model_seed"]), config["task"], config["subject"], int(config["test_session"]))
                if key in index: raise RuntimeError(f"duplicate historical fold: {key}")
                index[key] = (path.parent, path)
    for role in ROLES:
        if sum(key[0] == role for key in index) != 270: raise RuntimeError(f"historical {role} grid is not 270 folds")
    return index


def complete_index():
    index = existing_index()
    for task, subject in available_entries():
        if int(subject[-3:]) <= 10: continue
        for role in ROLES:
            for seed in SEEDS:
                for session in SESSIONS:
                    key = (role, seed, task, subject, session); path = new_config(role, seed, task, subject, session)
                    if key in index: raise RuntimeError(f"duplicate new fold: {key}")
                    index[key] = (new_run(role, seed, task, subject, session), path)
    return index


def freeze_before_test():
    index = complete_index(); expected = {(role, seed, task, subject, session) for role in ROLES for seed in SEEDS for task,subject in available_entries() for session in SESSIONS}
    if set(index) != expected: raise RuntimeError(f"full grid mismatch: actual={len(index)} expected={len(expected)}")
    rows = []
    for key in sorted(index):
        run_dir, config_path = index[key]; config = json.loads(config_path.read_text(encoding="utf-8"))
        if not complete_fit(run_dir, config["signature"]): raise RuntimeError(f"incomplete fold before test freeze: {run_dir}")
        success = json.loads((run_dir/"success.json").read_text(encoding="utf-8"))
        if key[3][0:2] == "VP" and int(key[3][-3:]) > 10 and success.get("test_evaluated") is not False: raise RuntimeError(f"new fold test touched before freeze: {run_dir}")
        rows.append({"role": key[0], "seed": key[1], "task": key[2], "subject": key[3], "session": key[4], "signature": config["signature"], "run_dir": str(run_dir), "checkpoint_sha256": sha256_file(run_dir/"best_model.pt")})
    write_json(EXP/"all_subject_before_test_freeze.json", {"status": "frozen", "selection": "B_M50_holdout32", "selection_source": "development_validation_only", "fit_count": len(rows), "per_role": 414, "available_entries": 46, "missing_entries": 38, "test_results_read_for_selection": False, "rows": rows, "frozen_at": now()})


def evaluated(run_dir, signature):
    if not complete_fit(run_dir, signature) or not (run_dir/"test_predictions.csv").exists(): return False
    success = json.loads((run_dir/"success.json").read_text(encoding="utf-8"))
    with (run_dir/"test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        fields = set(next(csv.reader(stream), []))
    return success.get("test_evaluated") is True and {"raw_0", "raw_1", "raw_type"} <= fields


def evaluate_all(study):
    if not (EXP/"all_subject_before_test_freeze.json").exists(): raise RuntimeError("all-subject fit grid is not frozen")
    index = complete_index(); records_path = EXP/"all_subject_test_export_ledger.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else {"expected": 1242, "completed": 0, "records": []}
    for key in sorted(index):
        role,seed,task,subject,session = key; run_dir,config_path = index[key]; config=json.loads(config_path.read_text(encoding="utf-8"))
        if evaluated(run_dir, config["signature"]): continue
        trainer = ROOT/"training"/("train_v3.py" if role=="E32" else "train_v5.py" if role=="B_M50" else "train_v4_fusion.py")
        print(f"[V5-all-test] {role} seed={seed} {task} {subject} s{session}",flush=True)
        code=subprocess.run([study["paths"]["mamba_python"],str(trainer),"--mode","evaluate","--config",str(config_path),"--run-dir",str(run_dir)],cwd=ROOT).returncode
        item={"role":role,"seed":seed,"task":task,"subject":subject,"session":session,"signature":config["signature"],"status":"successful" if code==0 and evaluated(run_dir,config["signature"]) else "failed","ended_at":now()};records["records"].append(item)
        if item["status"]!="successful":write_json(records_path,records);raise RuntimeError(f"test evaluation failed: {run_dir}")
        records["completed"]+=1;write_json(records_path,records)
    incomplete=[]
    for key,(run_dir,config_path) in index.items():
        config=json.loads(config_path.read_text(encoding="utf-8"))
        if not evaluated(run_dir,config["signature"]): incomplete.append(str(run_dir))
    if incomplete: raise RuntimeError(f"test export final audit failed for {len(incomplete)} folds")
    records["completed_total"]=len(index);write_json(records_path,records)
    write_json(EXP/"all_subject_test_export_complete.json", {"status":"complete","folds":len(index),"per_role":414,"available_entries":46,"raw_outputs_verified":True,"completed_at":now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("fit","freeze","evaluate"),required=True);args=parser.parse_args()
    study=yaml.safe_load(STUDY_PATH.read_text(encoding="utf-8"))
    write_json(EXP/f"all_subject_{args.stage}_process.json", {"pid":os.getpid(),"stage":args.stage,"status":"running","started_at":now(),"command":"tools/v5_all_subjects.py"})
    if args.stage=="fit": fit_new_subjects(study)
    elif args.stage=="freeze": freeze_before_test()
    else: evaluate_all(study)
    write_json(EXP/f"all_subject_{args.stage}_process.json", {"pid":os.getpid(),"stage":args.stage,"status":"complete","ended_at":now(),"command":"tools/v5_all_subjects.py"})


if __name__ == "__main__": main()
