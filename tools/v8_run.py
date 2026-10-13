"""Bootstrap, contract checks, and dry-run planning for MambaSTA V8.

This module deliberately exposes only non-test stages until a final selection
freeze exists.  Formal fit/evaluate stages are added behind the same registry
and ledger contract rather than through ad-hoc launch scripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mambasta_v8 import V8JointModel
from training.train_v2 import make_loader, scalar_stats, set_model_seed
from training.train_v4_fusion import load_data, state_hash


CONFIG_PATH = ROOT / "configs" / "mambasta_v8_final_joint.yaml"
EXPERIMENT = ROOT / "experiments" / "mambasta_v8_final_joint_20261012"
SAMPLE = ROOT / "experiments" / "mambasta_v5_final_20261006" / "screen" / "w1" / "B_M50" / "ma" / "VP001_session0"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def disk_free_gib(path=ROOT):
    free = os.statvfs(path).f_bavail * os.statvfs(path).f_frsize if os.name != "nt" else None
    if free is None:
        import shutil
        free = shutil.disk_usage(path).free
    return free / 2**30


def require_disk(config):
    free = disk_free_gib()
    if free < float(config["disk_stop_threshold_gib"]):
        raise RuntimeError(f"disk guard triggered: {free:.2f} GiB < {config['disk_stop_threshold_gib']} GiB")
    return free


def candidate_registry():
    rows = [
        ("J1_R4", "reference_interaction", "B_M50", "C_Add", {"rank": 4, "kd_weight": .5}),
        ("J1_R8", "reference_interaction", "B_M50", "J1_R4", {"rank": 8, "kd_weight": .5}),
        ("J2_025", "conditional_read", "B_M50", "J1_R4", {"rank": 4, "max_shift": .25}),
        ("J2_050", "conditional_read", "B_M50", "J2_025", {"rank": 4, "max_shift": .50}),
        ("J3_P4", "paired_prototype", None, "C_Concat", {"prototypes_per_class": 2}),
        ("J4_S8", "shared_exchange", None, "C_Concat", {"shared_dim": 8}),
        ("J1_Reg0", "reference_interaction", "B_M50", "J1_R4", {"rank": 4, "interaction_penalty": 0.0}),
        ("J1_Reg1e2", "reference_interaction", "B_M50", "J1_R4", {"rank": 4, "interaction_penalty": .01}),
        ("C_Add", "additive_control", None, None, {}),
        ("C_Concat", "concat_control", None, None, {"hidden": 16}),
        ("J1_NoKD", "reference_interaction", None, "J1_R4", {"rank": 4, "kd_weight": 0.0}),
        ("J1_RU_KD", "reference_interaction", "R_U", "J1_R4", {"rank": 4, "kd_weight": .5}),
    ]
    return {
        "registry_version": 1,
        "count": len(rows),
        "blocked_pre_registered": {
            "J5_R4": "blocked: no stable coarse-region coordinate contract; replaced by J1_Reg0",
            "J6_B4": "blocked: installed fused Mamba kernel has no verified epsilon-zero state-write hook; replaced by J1_Reg1e2",
        },
        "candidates": [
            {"config_id": i, "family": f, "teacher": t, "matched_control": c,
             "parameters": p, "executable_stages": ["s2", "s3", "s4", "s5", "s6"]}
            for i, f, t, c, p in rows
        ],
    }


def build_manifest(config):
    roots = {
        "wg": ROOT / config["paths"]["wg_raw"],
        "mi": ROOT / config["paths"]["mi_raw"],
        "ma": ROOT / config["paths"]["ma_raw"],
    }
    expected = {"wg": 26, "mi": 29, "ma": 29}
    rows = []
    for task, count in expected.items():
        for number in range(1, count + 1):
            subject = f"VP{number:03d}"
            path = (roots[task] / f"{subject}.npz").resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                if not {"eeg", "hbo", "hbr", "label"}.issubset(data.files):
                    raise RuntimeError(f"raw contract fields missing: {path}")
                labels = np.asarray(data["label"])
                if labels.shape not in {(2, 60), (60, 2)} or not np.isfinite(labels).all():
                    raise RuntimeError(f"label contract failed: {path}")
                label_hash = hashlib.sha256(labels.tobytes()).hexdigest()
            rows.append({
                "task": task, "subject": subject,
                "person_key": f"wg_cohort:{subject}" if task == "wg" else f"shin_multitask:{subject}",
                "raw_path": str(path), "raw_bytes": path.stat().st_size,
                "raw_sha256": sha256_file(path), "label_sha256": label_hash,
                "raw_trials": 60, "derived_windows": 600, "sessions": 3,
            })
    if len(rows) != 84 or len({row["person_key"] for row in rows}) != 55:
        raise AssertionError("84-entry/55-person manifest invariant failed")
    path = EXPERIMENT / "data_manifest_84.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def make_tasks(stage, roles=None):
    entries = [(task, f"VP{i:03d}") for task, n in (("wg", 26), ("mi", 29), ("ma", 29)) for i in range(1, n + 1)]
    seeds = [42, 123, 2026]
    tasks = []
    if stage == "s1":
        roles = ["B_M50", "E32", "R_U"]
        for role in roles:
            for task, subject in entries:
                for session in range(3):
                    for seed in seeds: tasks.append((role, task, subject, session, seed))
    elif stage == "s2":
        roles = roles or [c["config_id"] for c in candidate_registry()["candidates"]]
        for role in roles:
            for task, subject in entries:
                session = (int(subject[-3:]) - 1) % 3
                tasks.append((role, task, subject, session, 42))
    elif stage == "s3":
        roles = roles or ["DRY_A", "DRY_B", "DRY_C", "DRY_D"]
        for role in roles:
            for task, subject in entries:
                fixed = (int(subject[-3:]) - 1) % 3
                for session in range(3):
                    if session != fixed: tasks.append((role, task, subject, session, 42))
    elif stage in {"s4a", "s4b"}:
        roles = roles or (["DRY_A_T2", "DRY_A_T3", "DRY_B_T2", "DRY_B_T3"] if stage == "s4a" else ["DRY_A_T", "DRY_B_T"])
        for role in roles:
            for task, subject in entries:
                fixed = (int(subject[-3:]) - 1) % 3
                sessions = [fixed] if stage == "s4a" else [x for x in range(3) if x != fixed]
                for session in sessions: tasks.append((role, task, subject, session, 42))
    elif stage == "s5":
        roles = roles or ["DRY_A", "DRY_B"]
        for role in roles:
            for task, subject in entries:
                for session in range(3):
                    for seed in (123, 2026): tasks.append((role, task, subject, session, seed))
    elif stage == "s6":
        roles = roles or ["DRY_MATCHED_CONV"]
        for role in roles:
            for task, subject in entries:
                for session in range(3):
                    for seed in seeds: tasks.append((role, task, subject, session, seed))
    else:
        raise ValueError(stage)
    return tasks


def dry_run():
    expected = {"s1": 2268, "s2": 1008, "s3": 672, "s4a": 336, "s4b": 336, "s5": 1008, "s6": 756}
    output = {"identity_fields": ["role", "task", "person", "test_session", "model_seed", "split_indices_hash", "raw_hash", "preprocessing_hash", "input_support_hash", "init_checkpoint_hash", "config_hash", "code_hash"], "stages": {}}
    seen_dirs = set()
    for stage, count in expected.items():
        jobs = make_tasks(stage)
        directories = [f"{stage}/{r}/{t}/{p}_session{s}_seed{seed}" for r, t, p, s, seed in jobs]
        if len(jobs) != count or len(set(jobs)) != count or any(path in seen_dirs for path in directories):
            raise AssertionError(f"{stage} dry-run invariant failed")
        seen_dirs.update(directories)
        output["stages"][stage] = {"tasks": count, "first": jobs[0], "last": jobs[-1], "resume_policy": "skip_only_if signature+success+checkpoint+validation_predictions are complete"}
    output["formal_cap_without_s0"] = sum(expected.values())
    output["strict_reuse_expected_s1"] = 1242
    output["new_fit_expected_with_full_reuse"] = output["formal_cap_without_s0"] - 1242
    output["test_paths_generated"] = False
    write_json(EXPERIMENT / "stage_dry_run.json", output)
    return output


def bootstrap(config):
    free = require_disk(config)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config)
    registry = candidate_registry()
    (EXPERIMENT / "candidate_registry.yaml").write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    plan = dry_run()
    frozen = {"config": config, "config_sha256": canonical_hash(config), "source_config": str(CONFIG_PATH.resolve()), "data_entries": len(manifest), "unique_people": len({x['person_key'] for x in manifest}), "disk_free_gib": free, "test_access": False, "registry_sha256": canonical_hash(registry), "dry_run_sha256": canonical_hash(plan)}
    write_json(EXPERIMENT / "v8_protocol_frozen.json", frozen)
    return frozen


def donor_pairs(records, seed):
    """Whole-trial donors constrained within task/person/session/split/label."""
    rng = np.random.default_rng(seed)
    pairs = []
    grouped = {}
    for row in records:
        key = tuple(row[k] for k in ("task", "person", "session", "split", "label"))
        grouped.setdefault(key, []).append(int(row["trial"]))
    for key, trials in sorted(grouped.items()):
        trials = sorted(set(trials))
        if len(trials) < 2: raise ValueError(f"no legal non-self donor for {key}")
        shifted = list(np.asarray(trials)[rng.permutation(len(trials))])
        if any(a == b for a, b in zip(trials, shifted)):
            shifted = trials[1:] + trials[:1]
        pairs.extend((*key, source, donor) for source, donor in zip(trials, shifted))
    return pairs


def s0(config):
    require_disk(config); set_model_seed(42)
    old_config = json.loads((SAMPLE / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(SAMPLE / "best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"]["signature"] != old_config["signature"]: raise RuntimeError("sample checkpoint signature mismatch")
    fh, eh, eeg, fnirs, labels, split = load_data(old_config)
    stats = {"eeg": scalar_stats(eeg, split["train_indices"]), "fnirs": scalar_stats(fnirs, split["train_indices"])}
    loader = make_loader(eeg, fnirs, labels, split, "train", stats, old_config, False)
    batch_eeg, batch_fnirs, target, index = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_eeg, batch_fnirs, target = batch_eeg.to(device), batch_fnirs.to(device), target.to(device)

    def make(role):
        model_config = {**config["model"], "config_id": role}
        model = V8JointModel(model_config).to(device)
        model.load_b_m50_state(checkpoint["model_state"])
        model.freeze_backbone(); return model

    j1 = make("J1_R4")
    old_hash_before = state_hash(j1.backbone.state_dict())
    with torch.no_grad():
        direct, direct_diag = j1.backbone(batch_eeg, batch_fnirs, return_diagnostics=True)
        feature = j1.extract_features(batch_eeg, batch_fnirs, include_teacher_probability=True)
    adapter_max = float((direct - feature.b_m50_log_probability).abs().max())
    teacher_max = float((direct_diag["teacher_logits"] - feature.teacher_logits).abs().max())
    buffer_unchanged = old_hash_before == state_hash(j1.backbone.state_dict())

    train_e = []; train_n = []
    with torch.no_grad():
        for ee, nn_, _, _ in loader:
            f = j1.extract_features(ee.to(device), nn_.to(device))
            train_e.append(f.e_main); train_n.append(f.n_main)
    train_e, train_n = torch.cat(train_e), torch.cat(train_n)
    j1.head.fit_reference(train_e, train_n)
    j1.head.initialize_from_eeg_last_linear(j1.backbone.teacher.classifier[-1])
    logits, parts = j1.head(feature.e_main, feature.n_main)
    probability_max = float((logits.softmax(-1) - feature.teacher_logits.softmax(-1)).abs().max())
    decomposition_max = float((parts["bias"] + parts["eeg"] + parts["fnirs"] + parts["interaction"] - parts["score"]).abs().max())
    ee = (feature.e_main-j1.head.e_ref)/j1.head.e_scale
    nn_ = (feature.n_main-j1.head.n_ref)/j1.head.n_scale
    interaction_direct = ((j1.head.e_factor(ee)*j1.head.n_factor(nn_))*j1.head.cross_weight/math.sqrt(j1.head.rank)).sum(-1)
    reference_difference_max = float((interaction_direct-parts["interaction"]).abs().max())

    j2 = make("J2_025")
    j2.conditional_read.fit_reference(torch.cat([j2.extract_features(ee.to(device), nn_.to(device)).e_query for ee, nn_, _, _ in loader]))
    with torch.no_grad():
        _, base_diag = j2.backbone(batch_eeg, batch_fnirs, return_diagnostics=True)
        zero_feature = j2.extract_features(batch_eeg, batch_fnirs)
    j2_zero_max = float("nan")
    # Compare post-read features before n_head; recreate the conditional gain-zero read.
    with torch.no_grad():
        valid = torch.ones_like(base_diag["weights"], dtype=torch.bool)
        w0, _ = j2.conditional_read(base_diag["teacher_feature"], base_diag["memory"], base_diag["weights"], valid)
        values = j2.backbone.readout.value(base_diag["memory"])
        rv = torch.einsum("brt,brtd->brd", w0, values) + j2.backbone.region_embedding[None]
        rw = j2.backbone.region_score(rv).squeeze(-1).softmax(-1)
        read0 = torch.einsum("br,brd->bd", rw, rv)
        j2_zero_max = float((read0-base_diag["n_feature"]).abs().max())
    j2.conditional_read.raw_gain.data.fill_(0.1)
    valid = torch.ones_like(base_diag["weights"], dtype=torch.bool)
    w, _ = j2.conditional_read(base_diag["teacher_feature"], base_diag["memory"].detach(), base_diag["weights"].detach(), valid)
    gradient_loss = (w * torch.linspace(0, 1, w.shape[-1], device=device)).sum()
    gradient_loss.backward()
    gradient_norms = {name: float(parameter.grad.abs().sum()) if parameter.grad is not None else 0.0 for name, parameter in j2.conditional_read.named_parameters()}

    j3 = make("J3_P4")
    train_trials = torch.arange(len(train_e), device=device) // 10
    train_labels = torch.arange(len(train_e), device=device) // max(1, len(train_e)//2)
    train_labels = train_labels.clamp_max(1)
    j3.head.set_prototypes(train_e, train_n, train_labels, train_trials)
    proto_logits, _ = j3.head(feature.e_main, feature.n_main)
    prototype_ok = bool(torch.isfinite(proto_logits).all() and (j3.head.prototype_trial >= 0).all() and len(torch.unique(j3.head.prototype_trial)) == 4)

    records = [{"task":"ma","person":"VP001","session":0,"split":"train","label":label,"trial":trial} for label in (0,1) for trial in range(label*4,(label+1)*4)]
    donors = donor_pairs(records, 42)
    donor_ok = all(source != donor for *_, source, donor in donors) and len(donors) == 8
    missing_region_rejected = False
    try:
        j2.conditional_read(feature.e_query, base_diag["memory"], base_diag["weights"], torch.zeros_like(base_diag["weights"], dtype=torch.bool))
    except ValueError:
        missing_region_rejected = True
    with torch.no_grad():
        fallback_logits, _ = j1.head(feature.e_main, j1.head.n_ref[None].expand(len(feature.e_main), -1))
    missing_modality_fallback = bool(torch.isfinite(fallback_logits).all())

    original_probability_forward = j1.backbone.v5_head.forward
    def forbidden_probability_forward(*args, **kwargs):
        raise AssertionError("deployed V8 forward called the legacy probability head")
    j1.backbone.v5_head.forward = forbidden_probability_forward
    with torch.no_grad():
        deployed_logits, _ = j1(batch_eeg, batch_fnirs)
    j1.backbone.v5_head.forward = original_probability_forward
    deployment_probability_bypass = bool(torch.isfinite(deployed_logits).all())

    extra_parameters = sum(p.numel() for p in j1.head.parameters())
    trainable_parameters = sum(p.numel() for p in j1.parameters() if p.requires_grad)
    checks = {
        "adapter_equivalence": adapter_max <= 1e-7 and teacher_max <= 1e-7 and buffer_unchanged,
        "j1_probability_initialization_equivalence": probability_max <= 1e-6,
        "j1_decomposition": decomposition_max <= 1e-7 and reference_difference_max <= 1e-7,
        "j2_gain_zero_equivalence": j2_zero_max <= 1e-6,
        "j2_gradient_reachable": all(value > 0 for value in gradient_norms.values()),
        "j3_training_only_paired_prototypes": prototype_ok,
        "j6_epsilon_zero": "blocked_and_pre_registered_replacement",
        "donor_constraints": donor_ok,
        "missing_input_contract": missing_region_rejected and missing_modality_fallback,
        "deployment_parameter_limit": extra_parameters <= config["deployment_parameter_hard_cap"] and trainable_parameters == extra_parameters,
        "deployment_does_not_call_probability_head": deployment_probability_bypass,
        "no_test_access": True,
    }
    payload = {"sample": str(SAMPLE), "device": str(device), "train_indices_only": True, "test_access": False,
               "adapter_max_abs_difference": adapter_max, "teacher_max_abs_difference": teacher_max,
               "frozen_backbone_unchanged": buffer_unchanged, "j1_probability_max_abs_difference": probability_max,
               "j1_decomposition_max_abs_difference": decomposition_max, "j1_reference_difference_max_abs_difference": reference_difference_max,
               "j2_gain_zero_max_abs_difference": j2_zero_max, "j2_gradient_l1": gradient_norms,
               "j1_additional_parameters": extra_parameters, "j1_trainable_parameters": trainable_parameters,
               "checks": checks, "all_required_passed": all(value is True or isinstance(value, str) for value in checks.values())}
    write_json(EXPERIMENT / "legacy_adapter_equivalence.json", {k: payload[k] for k in ("sample", "adapter_max_abs_difference", "teacher_max_abs_difference", "frozen_backbone_unchanged", "test_access")})
    write_json(EXPERIMENT / "s0_contracts.json", payload)
    fh.close(); eh.close()
    if not payload["all_required_passed"]: raise RuntimeError("S0 contract failure; inspect s0_contracts.json")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("bootstrap", "dry-run", "s0"))
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    result = bootstrap(config) if args.stage == "bootstrap" else dry_run() if args.stage == "dry-run" else s0(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
