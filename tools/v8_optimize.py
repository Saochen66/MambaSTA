"""Validation-only V8 optimization on the amended MI/MA VP001-VP010 scope."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from models.mambasta_v8 import V8JointModel
from models.mambasta_v4_fusion import make_v4_fusion
from training.train_v2 import make_loader,scalar_stats,set_model_seed
from training.train_v4_fusion import load_data
from tools import v5_all_subjects as legacy

EXP=ROOT/"experiments/mambasta_v8_final_joint_20261012"
CONFIG=ROOT/"configs/mambasta_v8_final_joint.yaml"
ENTRIES={
    "wg":("VP022","VP003","VP017","VP023","VP024","VP014","VP007","VP005","VP018","VP016"),
    "mi":tuple(f"VP{i:03d}" for i in range(1,11)),
    "ma":tuple(f"VP{i:03d}" for i in range(1,11)),
}
TASKS=tuple(ENTRIES);SEED=42


def now():return datetime.now().astimezone().isoformat()
def write_json(path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps({k:v for k,v in x.items() if k!="signature"},sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def require_space(config,action):
    free=shutil.disk_usage(ROOT.anchor).free/2**30
    if free<config["disk_stop_threshold_gib"]:raise RuntimeError(f"STOP_DISK below 100 GiB before {action}: {free:.2f}")
    return free
def complete(run,signature):
    needed=("success.json","best_model.pt","validation_predictions.csv","metrics.json")
    if any(not(run/x).is_file() for x in needed) or (run/"test_predictions.csv").exists():return False
    try:s=json.loads((run/"success.json").read_text(encoding="utf-8"))
    except Exception:return False
    return s.get("signature")==signature and s.get("fit_complete") is True and s.get("test_evaluated") is False


def baseline_index(all_sessions=False,seeds=(SEED,)):
    index=legacy.complete_index();result={}
    for role in ("B_M50","R_U"):
        for task in TASKS:
            for subject in ENTRIES[task]:
                sessions=range(3) if all_sessions else ((int(subject[-3:])-1)%3,)
                for seed in seeds:
                    for session in sessions:
                        key=(role,seed,task,subject,session)
                        if key not in index:raise RuntimeError(f"missing historical baseline {key}")
                        run,cfg=index[key];config=json.loads(cfg.read_text(encoding="utf-8"))
                        if not legacy.complete_fit(run,config["signature"]):raise RuntimeError(f"incomplete baseline {run}")
                        result[key]=(run,cfg,config)
    return result


def feature_cache_path(task,subject,session,seed=SEED):
    suffix="" if seed==SEED else f"_seed{seed}"
    return EXP/"feature_cache"/task/f"{subject}_session{session}{suffix}.npz"


def build_feature_cache(task,subject,session,index,seed=SEED):
    path=feature_cache_path(task,subject,session,seed);manifest=path.with_suffix(".manifest.json")
    b_run,_,b_cfg=index[("B_M50",seed,task,subject,session)]
    r_run,_,r_cfg=index[("R_U",seed,task,subject,session)]
    identity={"task":task,"subject":subject,"session":session,"seed":seed,
              "b_m50_checkpoint_sha256":sha(b_run/"best_model.pt"),"r_u_checkpoint_sha256":sha(r_run/"best_model.pt"),
              "adapter_code_sha256":sha(ROOT/"models/mambasta_v8.py"),"test_access":False}
    identity["signature"]=canonical(identity)
    if path.exists() and manifest.exists():
        old=json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature")==identity["signature"] and old.get("complete") is True:return path,old
        raise RuntimeError(f"stale feature cache requires explicit review: {path}")
    fh,eh,eeg,fnirs,labels,split=load_data(b_cfg)
    stats={"eeg":scalar_stats(eeg,split["train_indices"]),"fnirs":scalar_stats(fnirs,split["train_indices"])}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");set_model_seed(seed)
    bck=torch.load(b_run/"best_model.pt",map_location="cpu",weights_only=False)
    v8=V8JointModel({"config_id":"J1_R4","d_model":32,"d_state":8,"d_conv":4,"expand":2,"query_rank":8,"dropout":.2}).to(device)
    v8.load_b_m50_state(bck["model_state"]);v8.freeze_backbone();v8.eval()
    rck=torch.load(r_run/"best_model.pt",map_location="cpu",weights_only=False)
    ru=make_v4_fusion(r_cfg["architecture"],dim=r_cfg["d_model"],state=r_cfg["d_state"],conv_kernel=r_cfg["d_conv"],expand=r_cfg["expand"],rank=r_cfg["query_rank"],dropout=r_cfg["dropout"]).to(device)
    ru.load_state_dict(rck["model_state"]);ru.eval()
    arrays={}
    for name in ("train","val"):
        loader=make_loader(eeg,fnirs,labels,split,name,stats,b_cfg,False);pieces={k:[] for k in ("e","n","q","m","w","b","r","y","i")}
        with torch.no_grad():
            for ee,nn_,yy,ii in loader:
                ee,nn_=ee.to(device),nn_.to(device);features=v8.extract_features(ee,nn_,include_teacher_probability=True)
                rlogits,_=ru(ee,nn_)
                values={"e":features.e_main,"n":features.n_main,"q":features.e_query,"m":features.memory,
                        "w":features.base_weights,"b":features.b_m50_log_probability,"r":torch.log_softmax(rlogits,-1),
                        "y":yy,"i":ii}
                for key,value in values.items():pieces[key].append(value.detach().cpu().numpy())
        for key,value in pieces.items():arrays[f"{name}_{key}"]=np.concatenate(value)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    identity.update({"complete":True,"cache_sha256":sha(path),"train_windows":len(arrays["train_y"]),"val_windows":len(arrays["val_y"]),"test_windows":0})
    write_json(manifest,identity);fh.close();eh.close();return path,identity


def candidates():
    registry=yaml.safe_load((EXP/"candidate_registry.yaml").read_text(encoding="utf-8"))
    values=registry["candidates"]
    if len(values)!=12:raise RuntimeError("candidate registry is not frozen at 12")
    return values


def load_ledger():
    path=EXP/"budget_ledger.json";ledger=json.loads(path.read_text(encoding="utf-8"))
    return path,ledger


def _metrics(run):
    return json.loads((run/"metrics.json").read_text(encoding="utf-8"))["selected_checkpoint_val"]


def _u(values):
    ordered=sorted(values,reverse=True)
    return float(np.mean([np.mean(ordered[:8]),np.mean(ordered[:10]),np.mean(ordered[:12])]))


def select_s2():
    """Freeze the amended-scope S2 decision using validation metrics only."""
    if not (EXP/"s2_fit_complete.json").is_file():raise RuntimeError("S2 is not complete")
    if list(EXP.glob("**/test_predictions.csv")):raise RuntimeError("test output exists before selection")
    rows=[]
    for role in [x["config_id"] for x in candidates()]:
        task_rows=[]
        for task in TASKS:
            window=[];trial=[]
            for subject in ENTRIES[task]:
                session=(int(subject[-3:])-1)%3
                value=_metrics(EXP/"s2"/role/task/f"{subject}_session{session}_seed42")
                window.append(float(value["accuracy"]));trial.append(float(value["trial"]["accuracy"]))
            item={"task":task,"window_accuracy":float(np.mean(window)),"trial_accuracy":float(np.mean(trial)),"u_top8_10_12":_u(window)}
            task_rows.append(item)
        rows.append({"role":role,"tasks":task_rows,"equal_task_window_accuracy":float(np.mean([x["window_accuracy"] for x in task_rows])),
                     "macro_trial_accuracy":float(np.mean([x["trial_accuracy"] for x in task_rows])),"minimum_task_u":float(min(x["u_top8_10_12"] for x in task_rows))})
    by_role={x["role"]:x for x in rows}
    # The plan protects the best J1 and J2, retains the best distinct research
    # direction, and carries the matched ordinary-fusion control alongside it.
    j1=max((x for x in rows if x["role"].startswith("J1_")),key=lambda x:(x["minimum_task_u"],x["equal_task_window_accuracy"],-len(x["role"])))
    j2=max((x for x in rows if x["role"].startswith("J2_")),key=lambda x:(x["minimum_task_u"],x["equal_task_window_accuracy"],-len(x["role"])))
    other=max((x for x in rows if x["role"].startswith(("J3_","J4_"))),key=lambda x:(x["minimum_task_u"],x["equal_task_window_accuracy"]))
    control="C_Concat" if other["role"]=="J3_P4" else "C_Add"
    promoted=[j1["role"],j2["role"],other["role"],control]
    payload={"status":"frozen","selection_source":"validation_only","scope":"30 amended entries; fixed session; seed42",
             "near_tie_pp":0.5,"promoted_roles":promoted,"rationale":{"protected_best_j1":j1["role"],"protected_best_j2":j2["role"],
             "best_other_research_direction":other["role"],"matched_control":control},"candidate_metrics":rows,
             "test_access":False,"selection_changed_by_test":False,"created_at":now()}
    write_json(EXP/"s2_selection_freeze.json",payload);return payload


def _run_cached_stage(stage,roles,task_specs):
    config=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));require_space(config,stage.upper())
    seeds=tuple(sorted({x[3] for x in task_specs}));index=baseline_index(all_sessions=True,seeds=seeds);ledger_path,ledger=load_ledger();registry={x["config_id"]:x for x in candidates()}
    expected=len(roles)*len(task_specs)
    for role in roles:
        if role not in registry:raise RuntimeError(f"unregistered role {role}")
        for task,subject,session,seed in task_specs:
            require_space(config,f"{stage} {role} {task}/{subject}")
            cache,manifest=build_feature_cache(task,subject,session,index,seed)
            b_run,_,b_cfg=index[("B_M50",seed,task,subject,session)]
            run=EXP/stage/role/task/f"{subject}_session{session}_seed{seed}"
            fold={**config["model"],"study":config["study"],"stage":f"{stage}_amended_old10","config_id":role,
                  "task":task,"subject":subject,"test_session":session,"model_seed":seed,
                  "feature_cache_path":str(cache.resolve()),"feature_cache_signature":manifest["signature"],
                  "b_m50_checkpoint":str((b_run/"best_model.pt").resolve()),"b_m50_signature":b_cfg["signature"],
                  "max_epochs":80,"patience":10,"batch_size":64,"learning_rate":config["training"]["head_lr"],
                  "weight_decay":config["training"]["weight_decay"],"gradient_clip":config["training"]["gradient_clip"],
                  "kd_weight":config["training"]["kd_weight"],"interaction_penalty":config["training"]["interaction_penalty"],
                  "temperature":config["training"]["temperature"],"selection_source":"validation_only","test_access":False,
                  "scope_amendment_sha256":sha(EXP/"protocol_amendment_20261013.json"),
                  "trainer_sha256":sha(ROOT/"training/train_v8_cached.py"),"model_sha256":sha(ROOT/"models/mambasta_v8.py")}
            fold["signature"]=canonical(fold);cfg=EXP/"_configs"/stage/role/task/f"{subject}_session{session}_seed{seed}.json"
            if complete(run,fold["signature"]):continue
            if ledger["attempted"]>=ledger["optimization_fit_hard_cap"]:raise RuntimeError("6500 fit hard cap reached")
            write_json(cfg,fold);record={"stage":stage,"role":role,"task":task,"subject":subject,"test_session":session,"model_seed":seed,"signature":fold["signature"],"status":"running","started_at":now()}
            ledger["attempted"]+=1;ledger["records"].append(record);write_json(ledger_path,ledger)
            print(f"[V8-{stage.upper()}] {role} {task} {subject} s{session} seed{seed}",flush=True)
            code=subprocess.run([config["paths"]["mamba_python"],str(ROOT/"training/train_v8_cached.py"),"--config",str(cfg),"--run-dir",str(run)],cwd=ROOT).returncode
            record.update({"ended_at":now(),"returncode":code})
            if code==0 and complete(run,fold["signature"]):record["status"]="successful";ledger["successful"]+=1
            else:record["status"]="failed";ledger["failed"]+=1;write_json(ledger_path,ledger);raise RuntimeError(f"{stage} failed {run}")
            write_json(ledger_path,ledger)
    actual=list((EXP/stage).glob("**/success.json"));tests=list((EXP/stage).glob("**/test_predictions.csv"))
    if len(actual)!=expected or tests:raise RuntimeError(f"{stage} completion mismatch {len(actual)}/{expected}, tests={len(tests)}")
    write_json(EXP/f"{stage}_fit_complete.json",{"status":"complete","roles":roles,"new_fits":expected,"failed":0,"test_outputs":0,"completed_at":now()})


def run_s3():
    freeze=json.loads((EXP/"s2_selection_freeze.json").read_text(encoding="utf-8"))
    roles=freeze["promoted_roles"]
    specs=[]
    for task in TASKS:
        for subject in ENTRIES[task]:
            fixed=(int(subject[-3:])-1)%3
            specs.extend((task,subject,s,SEED) for s in range(3) if s!=fixed)
    _run_cached_stage("s3",roles,specs)


def select_s3():
    if not (EXP/"s3_fit_complete.json").is_file():raise RuntimeError("S3 is not complete")
    if list(EXP.glob("**/test_predictions.csv")):raise RuntimeError("test output exists before selection")
    roles=json.loads((EXP/"s2_selection_freeze.json").read_text(encoding="utf-8"))["promoted_roles"]
    summary=[]
    for role in roles:
        task_rows=[]
        for task in TASKS:
            subject_values=[];trial_values=[]
            for subject in ENTRIES[task]:
                fixed=(int(subject[-3:])-1)%3;windows=[];trials=[]
                for session in range(3):
                    stage="s2" if session==fixed else "s3"
                    value=_metrics(EXP/stage/role/task/f"{subject}_session{session}_seed42")
                    windows.append(float(value["accuracy"]));trials.append(float(value["trial"]["accuracy"]))
                subject_values.append(float(np.mean(windows)));trial_values.extend(trials)
            task_rows.append({"task":task,"window_accuracy":float(np.mean(subject_values)),"trial_accuracy":float(np.mean(trial_values)),"u_top8_10_12":_u(subject_values)})
        summary.append({"role":role,"tasks":task_rows,"equal_task_window_accuracy":float(np.mean([x["window_accuracy"] for x in task_rows])),
                        "macro_trial_accuracy":float(np.mean([x["trial_accuracy"] for x in task_rows])),"minimum_task_u":float(min(x["u_top8_10_12"] for x in task_rows))})
    research=[x for x in summary if x["role"].startswith("J")]
    # Maximin U is primary; within the preregistered 0.5 pp near-tie band,
    # retain the stronger equal-task mean. Keep two structurally distinct families.
    first=max(research,key=lambda x:(x["minimum_task_u"],x["equal_task_window_accuracy"]))
    family=lambda r:r.split("_")[0]
    remaining=[x for x in research if family(x["role"])!=family(first["role"])]
    second=max(remaining,key=lambda x:(x["equal_task_window_accuracy"],x["minimum_task_u"]))
    selected=[first["role"],second["role"]]
    payload={"status":"frozen","selection_source":"validation_only","scope":"30 amended entries; three sessions; seed42",
             "selected_structures":selected,"candidate_metrics":summary,"test_access":False,"selection_changed_by_test":False,"created_at":now()}
    write_json(EXP/"s3_selection_freeze.json",payload);return payload


def run_s4a():
    config=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));require_space(config,"S4a")
    roles=json.loads((EXP/"s3_selection_freeze.json").read_text(encoding="utf-8"))["selected_structures"]
    index=baseline_index(all_sessions=True);ledger_path,ledger=load_ledger();expected=len(roles)*2*30
    for role in roles:
        for recipe in ("T2","T3"):
            variant=f"{role}_{recipe}"
            for task in TASKS:
                for subject in ENTRIES[task]:
                    session=(int(subject[-3:])-1)%3;require_space(config,f"S4a {variant} {task}/{subject}")
                    cache,manifest=build_feature_cache(task,subject,session,index);b_run,_,b_cfg=index[("B_M50",SEED,task,subject,session)]
                    source=EXP/"s2"/role/task/f"{subject}_session{session}_seed42";source_cfg=json.loads((source/"config.json").read_text(encoding="utf-8"))
                    run=EXP/"s4a"/variant/task/f"{subject}_session{session}_seed42"
                    fold={**b_cfg,**config["model"],"study":config["study"],"stage":"s4a_amended_old10","config_id":role,"recipe":recipe,
                          "task":task,"subject":subject,"test_session":session,"model_seed":SEED,"feature_cache_path":str(cache.resolve()),
                          "feature_cache_signature":manifest["signature"],"source_checkpoint":str((source/"best_model.pt").resolve()),
                          "source_signature":source_cfg["signature"],"max_epochs":40 if recipe=="T2" else 20,"patience":8,
                          "batch_size":32,"head_lr":5e-4,"n_lr":5e-5,"e_lr":1e-5,"weight_decay":config["training"]["weight_decay"],
                          "gradient_clip":config["training"]["gradient_clip"],"kd_weight":0.1,"interaction_penalty":config["training"]["interaction_penalty"],
                          "temperature":config["training"]["temperature"],"selection_source":"validation_only","test_access":False,
                          "scope_amendment_sha256":sha(EXP/"protocol_amendment_20261013.json"),"trainer_sha256":sha(ROOT/"training/train_v8_staged.py"),"model_sha256":sha(ROOT/"models/mambasta_v8.py")}
                    fold["signature"]=canonical(fold);cfg=EXP/"_configs"/"s4a"/variant/task/f"{subject}_session{session}_seed42.json"
                    if complete(run,fold["signature"]):continue
                    if ledger["attempted"]>=ledger["optimization_fit_hard_cap"]:raise RuntimeError("6500 fit hard cap reached")
                    write_json(cfg,fold);record={"stage":"s4a","role":variant,"task":task,"subject":subject,"test_session":session,"model_seed":SEED,"signature":fold["signature"],"status":"running","started_at":now()}
                    ledger["attempted"]+=1;ledger["records"].append(record);write_json(ledger_path,ledger)
                    print(f"[V8-S4A] {variant} {task} {subject} s{session}",flush=True)
                    code=subprocess.run([config["paths"]["mamba_python"],str(ROOT/"training/train_v8_staged.py"),"--config",str(cfg),"--run-dir",str(run)],cwd=ROOT).returncode
                    record.update({"ended_at":now(),"returncode":code})
                    if code==0 and complete(run,fold["signature"]):record["status"]="successful";ledger["successful"]+=1
                    else:record["status"]="failed";ledger["failed"]+=1;write_json(ledger_path,ledger);raise RuntimeError(f"S4a failed {run}")
                    write_json(ledger_path,ledger)
    actual=list((EXP/"s4a").glob("**/success.json"));tests=list((EXP/"s4a").glob("**/test_predictions.csv"))
    if len(actual)!=expected or tests:raise RuntimeError(f"S4a completion mismatch {len(actual)}/{expected}, tests={len(tests)}")
    write_json(EXP/"s4a_fit_complete.json",{"status":"complete","structures":roles,"recipes":["T2","T3"],"new_fits":expected,"failed":0,"test_outputs":0,"completed_at":now()})


def select_s4a():
    if not (EXP/"s4a_fit_complete.json").is_file():raise RuntimeError("S4a is not complete")
    structures=json.loads((EXP/"s3_selection_freeze.json").read_text(encoding="utf-8"))["selected_structures"]
    decisions=[]
    for role in structures:
        variants=[]
        for recipe,stage,variant in (("T1","s2",role),("T2","s4a",f"{role}_T2"),("T3","s4a",f"{role}_T3")):
            task_rows=[]
            for task in TASKS:
                windows=[];trials=[]
                for subject in ENTRIES[task]:
                    session=(int(subject[-3:])-1)%3;value=_metrics(EXP/stage/variant/task/f"{subject}_session{session}_seed42")
                    windows.append(float(value["accuracy"]));trials.append(float(value["trial"]["accuracy"]))
                task_rows.append({"task":task,"window_accuracy":float(np.mean(windows)),"trial_accuracy":float(np.mean(trials)),"u_top8_10_12":_u(windows)})
            variants.append({"recipe":recipe,"tasks":task_rows,"equal_task_window_accuracy":float(np.mean([x["window_accuracy"] for x in task_rows])),
                             "macro_trial_accuracy":float(np.mean([x["trial_accuracy"] for x in task_rows])),"minimum_task_u":float(min(x["u_top8_10_12"] for x in task_rows))})
        chosen=max(variants,key=lambda x:(x["minimum_task_u"],x["equal_task_window_accuracy"],x["macro_trial_accuracy"],-{"T1":0,"T2":1,"T3":2}[x["recipe"]]))
        decisions.append({"structure":role,"selected_recipe":chosen["recipe"],"variants":variants})
    payload={"status":"frozen","selection_source":"validation_only","decisions":decisions,"test_access":False,"selection_changed_by_test":False,"created_at":now()}
    write_json(EXP/"s4_recipe_selection_freeze.json",payload)
    # Both structures retained T1 in this run, so their other sessions already
    # exist from S3; S4b requires no duplicate fitting.
    if all(x["selected_recipe"]=="T1" for x in decisions):
        write_json(EXP/"s4b_fit_complete.json",{"status":"complete","new_fits":0,"reason":"T1 retained for both structures; S2+S3 already cover three sessions","failed":0,"test_outputs":0,"completed_at":now()})
    return payload


def run_s5():
    freeze=json.loads((EXP/"s4_recipe_selection_freeze.json").read_text(encoding="utf-8"))
    if not all(x["selected_recipe"]=="T1" for x in freeze["decisions"]):raise RuntimeError("non-T1 S4b continuation is not frozen complete")
    roles=[x["structure"] for x in freeze["decisions"]];specs=[]
    for task in TASKS:
        for subject in ENTRIES[task]:
            for session in range(3):
                for seed in (123,2026):specs.append((task,subject,session,seed))
    _run_cached_stage("s5",roles,specs)


def _three_seed_metrics(role):
    task_rows=[]
    for task in TASKS:
        subject_values=[];trial_values=[]
        for subject in ENTRIES[task]:
            fixed=(int(subject[-3:])-1)%3;windows=[]
            for seed in (42,123,2026):
                for session in range(3):
                    stage=("s2" if session==fixed else "s3") if seed==42 else "s5"
                    value=_metrics(EXP/stage/role/task/f"{subject}_session{session}_seed{seed}")
                    windows.append(float(value["accuracy"]));trial_values.append(float(value["trial"]["accuracy"]))
            subject_values.append(float(np.mean(windows)))
        task_rows.append({"task":task,"window_accuracy":float(np.mean(subject_values)),"trial_accuracy":float(np.mean(trial_values)),"u_top8_10_12":_u(subject_values)})
    return {"role":role,"tasks":task_rows,"equal_task_window_accuracy":float(np.mean([x["window_accuracy"] for x in task_rows])),
            "macro_trial_accuracy":float(np.mean([x["trial_accuracy"] for x in task_rows])),"minimum_task_u":float(min(x["u_top8_10_12"] for x in task_rows))}


def select_s5():
    if not (EXP/"s5_fit_complete.json").is_file():raise RuntimeError("S5 is not complete")
    if list(EXP.glob("**/test_predictions.csv")):raise RuntimeError("test output exists before final selection")
    roles=[x["structure"] for x in json.loads((EXP/"s4_recipe_selection_freeze.json").read_text(encoding="utf-8"))["decisions"]]
    metrics=[_three_seed_metrics(role) for role in roles]
    winner=max(metrics,key=lambda x:(x["minimum_task_u"],x["equal_task_window_accuracy"],x["macro_trial_accuracy"]))
    payload={"status":"frozen","selection_source":"validation_only","scope":"30 amended entries; three sessions; three seeds",
             "winner":winner["role"],"winner_recipe":"T1","candidate_metrics":metrics,"matched_conv_role":f"{winner['role']}_CONV",
             "test_access":False,"selection_changed_by_test":False,"created_at":now()}
    write_json(EXP/"selection_freeze.json",payload);return payload


def _complete_v5(run,signature):
    needed=("success.json","best_model.pt","validation_predictions.csv","metrics.json")
    if any(not (run/x).is_file() for x in needed) or (run/"test_predictions.csv").exists():return False
    try:return json.loads((run/"success.json").read_text(encoding="utf-8")).get("signature")==signature
    except Exception:return False


def _historical_conv_index():
    root=ROOT/"experiments/mambasta_v5_final_20261006";result={}
    for cfg in root.glob("**/B_C50/**/config.json"):
        value=json.loads(cfg.read_text(encoding="utf-8"));key=(value["task"],value["subject"],int(value["test_session"]),int(value["model_seed"]));run=cfg.parent
        if _complete_v5(run,value["signature"]):result[key]=(run,value)
    return result


def _ensure_conv_baseline(task,subject,session,seed,b_index,m_index,config,ledger):
    key=(task,subject,session,seed)
    if key in b_index:return b_index[key]
    m_run,_,m_cfg=m_index[("B_M50",seed,task,subject,session)]
    run=EXP/"s6_conv_pretrain"/task/f"{subject}_session{session}_seed{seed}"
    fold={k:v for k,v in m_cfg.items() if k!="signature"};fold.update({"study":config["study"],"phase":"s6_matched_conv_pretrain",
          "config_id":"B_C50","family":"probability","mixer":"conv","alpha":0.5,"selection_source":"validation_only","test_access":False,
          "source_mamba_baseline_signature":m_cfg["signature"],"code_sha256":sha(ROOT/"training/train_v5.py")})
    fold["signature"]=canonical(fold);cfg=EXP/"_configs"/"s6_conv_pretrain"/task/f"{subject}_session{session}_seed{seed}.json"
    if not _complete_v5(run,fold["signature"]):
        write_json(cfg,fold);print(f"[V8-S6-PRETRAIN] B_C50 {task} {subject} s{session} seed{seed}",flush=True)
        code=subprocess.run([config["paths"]["mamba_python"],str(ROOT/"training/train_v5.py"),"--config",str(cfg),"--run-dir",str(run)],cwd=ROOT).returncode
        if code or not _complete_v5(run,fold["signature"]):raise RuntimeError(f"S6 Conv pretrain failed {run}")
        ledger.append({"task":task,"subject":subject,"test_session":session,"model_seed":seed,"status":"successful","run_dir":str(run.resolve()),"completed_at":now()})
        write_json(EXP/"s6_conv_pretrain_ledger.json",{"records":ledger,"updated_at":now()})
    b_index[key]=(run,fold);return b_index[key]


def _conv_cache(task,subject,session,seed,run,run_cfg):
    path=EXP/"feature_cache_conv"/task/f"{subject}_session{session}_seed{seed}.npz";manifest=path.with_suffix(".manifest.json")
    identity={"task":task,"subject":subject,"session":session,"seed":seed,"b_c50_checkpoint_sha256":sha(run/"best_model.pt"),
              "adapter_code_sha256":sha(ROOT/"models/mambasta_v8.py"),"test_access":False};identity["signature"]=canonical(identity)
    if path.exists() and manifest.exists():
        old=json.loads(manifest.read_text(encoding="utf-8"))
        if old.get("signature")==identity["signature"] and old.get("complete") is True:return path,old
        raise RuntimeError(f"stale Conv feature cache requires explicit review: {path}")
    fh,eh,eeg,fnirs,labels,split=load_data(run_cfg);stats={"eeg":scalar_stats(eeg,split["train_indices"]),"fnirs":scalar_stats(fnirs,split["train_indices"])}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");set_model_seed(seed);ck=torch.load(run/"best_model.pt",map_location="cpu",weights_only=False)
    model=V8JointModel({"config_id":"J1_R4_CONV","backbone_mixer":"conv","d_model":32,"d_state":8,"d_conv":4,"expand":2,"query_rank":8,"dropout":.2}).to(device)
    model.load_b_m50_state(ck["model_state"]);model.freeze_backbone();model.eval();arrays={}
    for name in ("train","val"):
        loader=make_loader(eeg,fnirs,labels,split,name,stats,run_cfg,False);pieces={k:[] for k in ("e","n","q","m","w","b","r","y","i")}
        with torch.no_grad():
            for ee,nn_,yy,ii in loader:
                ee,nn_=ee.to(device),nn_.to(device);features=model.extract_features(ee,nn_,include_teacher_probability=True)
                values={"e":features.e_main,"n":features.n_main,"q":features.e_query,"m":features.memory,"w":features.base_weights,
                        "b":features.b_m50_log_probability,"r":features.b_m50_log_probability,"y":yy,"i":ii}
                for k,v in values.items():pieces[k].append(v.detach().cpu().numpy())
        for k,v in pieces.items():arrays[f"{name}_{k}"]=np.concatenate(v)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays);identity.update({"complete":True,"cache_sha256":sha(path),"test_windows":0});write_json(manifest,identity);fh.close();eh.close();return path,identity


def run_s6():
    config=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));require_space(config,"S6")
    freeze=json.loads((EXP/"selection_freeze.json").read_text(encoding="utf-8"));winner=freeze["winner"]
    if winner!="J1_R4":raise RuntimeError("S6 Conv implementation is frozen for J1_R4 winner only")
    m_index=baseline_index(all_sessions=True,seeds=(42,123,2026));b_index=_historical_conv_index();pre=[]
    if (EXP/"s6_conv_pretrain_ledger.json").is_file():pre=json.loads((EXP/"s6_conv_pretrain_ledger.json").read_text(encoding="utf-8"))["records"]
    ledger_path,ledger=load_ledger();expected=30*3*3;role="J1_R4_CONV"
    for task in TASKS:
        for subject in ENTRIES[task]:
            for session in range(3):
                for seed in (42,123,2026):
                    require_space(config,f"S6 {task}/{subject}");base_run,base_cfg=_ensure_conv_baseline(task,subject,session,seed,b_index,m_index,config,pre)
                    cache,manifest=_conv_cache(task,subject,session,seed,base_run,base_cfg);run=EXP/"s6"/role/task/f"{subject}_session{session}_seed{seed}"
                    fold={**config["model"],"study":config["study"],"stage":"s6_matched_conv_amended","config_id":role,"backbone_mixer":"conv",
                          "task":task,"subject":subject,"test_session":session,"model_seed":seed,"feature_cache_path":str(cache.resolve()),
                          "feature_cache_signature":manifest["signature"],"b_m50_checkpoint":str((base_run/"best_model.pt").resolve()),"b_m50_signature":base_cfg["signature"],
                          "max_epochs":80,"patience":10,"batch_size":64,"learning_rate":config["training"]["head_lr"],"weight_decay":config["training"]["weight_decay"],
                          "gradient_clip":config["training"]["gradient_clip"],"kd_weight":config["training"]["kd_weight"],"interaction_penalty":config["training"]["interaction_penalty"],
                          "temperature":config["training"]["temperature"],"selection_source":"validation_only","test_access":False,
                          "trainer_sha256":sha(ROOT/"training/train_v8_cached.py"),"model_sha256":sha(ROOT/"models/mambasta_v8.py")}
                    fold["signature"]=canonical(fold);cfg=EXP/"_configs"/"s6"/role/task/f"{subject}_session{session}_seed{seed}.json"
                    if complete(run,fold["signature"]):continue
                    if ledger["attempted"]>=ledger["optimization_fit_hard_cap"]:raise RuntimeError("6500 fit hard cap reached")
                    write_json(cfg,fold);record={"stage":"s6","role":role,"task":task,"subject":subject,"test_session":session,"model_seed":seed,"signature":fold["signature"],"status":"running","started_at":now()}
                    ledger["attempted"]+=1;ledger["records"].append(record);write_json(ledger_path,ledger);print(f"[V8-S6] {role} {task} {subject} s{session} seed{seed}",flush=True)
                    code=subprocess.run([config["paths"]["mamba_python"],str(ROOT/"training/train_v8_cached.py"),"--config",str(cfg),"--run-dir",str(run)],cwd=ROOT).returncode
                    record.update({"ended_at":now(),"returncode":code})
                    if code==0 and complete(run,fold["signature"]):record["status"]="successful";ledger["successful"]+=1
                    else:record["status"]="failed";ledger["failed"]+=1;write_json(ledger_path,ledger);raise RuntimeError(f"S6 failed {run}")
                    write_json(ledger_path,ledger)
    actual=list((EXP/"s6").glob("**/success.json"));tests=list((EXP/"s6").glob("**/test_predictions.csv"))
    if len(actual)!=expected or tests:raise RuntimeError(f"S6 completion mismatch {len(actual)}/{expected}, tests={len(tests)}")
    write_json(EXP/"s6_fit_complete.json",{"status":"complete","winner":winner,"matched_conv":role,"new_fits":expected,"failed":0,"test_outputs":0,"completed_at":now()})


def run_s2():
    config=yaml.safe_load(CONFIG.read_text(encoding="utf-8"));require_space(config,"S2")
    amendment=json.loads((EXP/"protocol_amendment_20261013.json").read_text(encoding="utf-8"))
    if amendment["selection_entries"]["entry_count"]!=30 or amendment["test_access"]:raise RuntimeError("scope amendment invalid")
    index=baseline_index();ledger_path,ledger=load_ledger();expected=12*30
    for task in TASKS:
        for subject in ENTRIES[task]:
            session=(int(subject[-3:])-1)%3;cache,manifest=build_feature_cache(task,subject,session,index)
            b_run,_,b_cfg=index[("B_M50",SEED,task,subject,session)]
            for candidate in candidates():
                require_space(config,f"S2 {candidate['config_id']} {task}/{subject}")
                role=candidate["config_id"];run=EXP/"s2"/role/task/f"{subject}_session{session}_seed42"
                fold={**config["model"],"study":config["study"],"stage":"s2_amended_old10","config_id":role,
                      "task":task,"subject":subject,"test_session":session,"model_seed":SEED,
                      "feature_cache_path":str(cache.resolve()),"feature_cache_signature":manifest["signature"],
                      "b_m50_checkpoint":str((b_run/"best_model.pt").resolve()),"b_m50_signature":b_cfg["signature"],
                      "max_epochs":80,"patience":10,"batch_size":64,"learning_rate":config["training"]["head_lr"],
                      "weight_decay":config["training"]["weight_decay"],"gradient_clip":config["training"]["gradient_clip"],
                      "kd_weight":config["training"]["kd_weight"],"interaction_penalty":config["training"]["interaction_penalty"],
                      "temperature":config["training"]["temperature"],"selection_source":"validation_only",
                      "test_access":False,"scope_amendment_sha256":sha(EXP/"protocol_amendment_20261013.json"),
                      "trainer_sha256":sha(ROOT/"training/train_v8_cached.py"),"model_sha256":sha(ROOT/"models/mambasta_v8.py")}
                fold["signature"]=canonical(fold);cfg=EXP/"_configs"/"s2"/role/task/f"{subject}_session{session}_seed42.json"
                if complete(run,fold["signature"]):continue
                if ledger["attempted"]>=ledger["optimization_fit_hard_cap"]:raise RuntimeError("6500 fit hard cap reached")
                write_json(cfg,fold);record={"stage":"s2","role":role,"task":task,"subject":subject,"test_session":session,"model_seed":SEED,"signature":fold["signature"],"status":"running","started_at":now()}
                ledger["attempted"]+=1;ledger["records"].append(record);write_json(ledger_path,ledger)
                print(f"[V8-S2] {role} {task} {subject} s{session}",flush=True)
                code=subprocess.run([config["paths"]["mamba_python"],str(ROOT/"training/train_v8_cached.py"),"--config",str(cfg),"--run-dir",str(run)],cwd=ROOT).returncode
                record.update({"ended_at":now(),"returncode":code})
                if code==0 and complete(run,fold["signature"]):record["status"]="successful";ledger["successful"]+=1
                else:record["status"]="failed";ledger["failed"]+=1;write_json(ledger_path,ledger);raise RuntimeError(f"S2 failed {run}")
                write_json(ledger_path,ledger)
    actual=list((EXP/"s2").glob("**/success.json"));tests=list((EXP/"s2").glob("**/test_predictions.csv"))
    if len(actual)!=expected or tests:raise RuntimeError(f"S2 completion mismatch {len(actual)}/{expected}, tests={len(tests)}")
    write_json(EXP/"s2_fit_complete.json",{"status":"complete","scope":"mi_ma_old10_plus_wg_validation_top10","new_fits":expected,"failed":0,"test_outputs":0,"completed_at":now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("s2","select-s2","s3","select-s3","s4a","select-s4a","s5","select-s5","s6"),required=True);args=parser.parse_args()
    if args.stage=="select-s2":select_s2();return
    if args.stage=="select-s3":select_s3();return
    if args.stage=="select-s4a":select_s4a();return
    if args.stage=="select-s5":select_s5();return
    process=EXP/f"{args.stage}_process.json";write_json(process,{"pid":os.getpid(),"stage":args.stage,"status":"running","started_at":now()})
    try:
        if args.stage=="s2":run_s2()
        elif args.stage=="s3":run_s3()
        elif args.stage=="s4a":run_s4a()
        elif args.stage=="s5":run_s5()
        else:run_s6()
    except Exception as error:write_json(process,{"pid":os.getpid(),"stage":args.stage,"status":"failed","error":repr(error),"ended_at":now()});raise
    write_json(process,{"pid":os.getpid(),"stage":args.stage,"status":"complete","ended_at":now()})


if __name__=="__main__":main()
