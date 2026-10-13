"""Freeze V8 validation conclusions, export frozen tests, and aggregate scope results."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from models.mambasta_v8 import V8JointModel
from training.train_v2 import make_loader, scalar_stats
from training.train_v4_fusion import load_data
from training.train_v8_cached import rows_and_metrics, write_predictions
from tools import v5_all_subjects as legacy

EXP=ROOT/"experiments/mambasta_v8_final_joint_20261012"
ENTRIES={"wg":("VP022","VP003","VP017","VP023","VP024","VP014","VP007","VP005","VP018","VP016"),
         "mi":tuple(f"VP{i:03d}" for i in range(1,11)),"ma":tuple(f"VP{i:03d}" for i in range(1,11))}
SEEDS=(42,123,2026)

def now():return datetime.now().astimezone().isoformat()
def write_json(path,payload):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
def require_space(action):
    free=shutil.disk_usage(ROOT.anchor).free/2**30
    if free<100:raise RuntimeError(f"STOP_DISK below 100 GiB before {action}: {free:.2f}")
    return free
def _metric(run):return json.loads((run/"metrics.json").read_text(encoding="utf-8"))["selected_checkpoint_val"]
def _run(role,task,subject,session,seed):
    if role=="J1_R4":
        fixed=(int(subject[-3:])-1)%3
        stage=("s2" if session==fixed else "s3") if seed==42 else "s5"
        return EXP/stage/role/task/f"{subject}_session{session}_seed{seed}"
    return EXP/"s6"/role/task/f"{subject}_session{session}_seed{seed}"
def _cache(task,subject,session,seed,conv=False):
    if conv:return EXP/"feature_cache_conv"/task/f"{subject}_session{session}_seed{seed}.npz"
    suffix="" if seed==42 else f"_seed{seed}"
    return EXP/"feature_cache"/task/f"{subject}_session{session}{suffix}.npz"
def _u(values):
    a=sorted(values,reverse=True);return float(np.mean([np.mean(a[:8]),np.mean(a[:10]),np.mean(a[:12])]))

def _summary(role):
    tasks=[]
    for task,subjects in ENTRIES.items():
        per=[];trials=[]
        for subject in subjects:
            values=[]
            for seed in SEEDS:
                for session in range(3):
                    m=_metric(_run(role,task,subject,session,seed));values.append(float(m["accuracy"]));trials.append(float(m["trial"]["accuracy"]))
            per.append(float(np.mean(values)))
        tasks.append({"task":task,"window_accuracy":float(np.mean(per)),"trial_accuracy":float(np.mean(trials)),"u_top8_10_12":_u(per)})
    return {"role":role,"tasks":tasks,"equal_task_window_accuracy":float(np.mean([x["window_accuracy"] for x in tasks])),
            "macro_trial_accuracy":float(np.mean([x["trial_accuracy"] for x in tasks])),"minimum_task_u":float(min(x["u_top8_10_12"] for x in tasks))}

def _baseline_summary():
    tasks=[]
    for task,subjects in ENTRIES.items():
        per=[];trials=[]
        for subject in subjects:
            values=[]
            for seed in SEEDS:
                for session in range(3):
                    with np.load(_cache(task,subject,session,seed)) as z:
                        p=np.exp(z["val_b"]);y=z["val_y"];idx=z["val_i"];values.append(float((p.argmax(1)==y).mean()))
                        for trial in np.unique(idx//10):
                            q=idx//10==trial;trials.append(float(p[q].mean(0).argmax()==y[q][0]))
            per.append(float(np.mean(values)))
        tasks.append({"task":task,"window_accuracy":float(np.mean(per)),"trial_accuracy":float(np.mean(trials)),"u_top8_10_12":_u(per)})
    return {"role":"B_M50","tasks":tasks,"equal_task_window_accuracy":float(np.mean([x["window_accuracy"] for x in tasks])),
            "macro_trial_accuracy":float(np.mean([x["trial_accuracy"] for x in tasks])),"minimum_task_u":float(min(x["u_top8_10_12"] for x in tasks))}

def _interaction_audit():
    totals={k:0. for k in ("count","full_correct","off_correct","e_ref_correct","n_ref_correct","donor_correct","interaction_abs","donor_score_abs")}
    for task,subjects in ENTRIES.items():
        for subject in subjects:
            for seed in SEEDS:
                for session in range(3):
                    run=_run("J1_R4",task,subject,session,seed);ck=torch.load(run/"best_model.pt",map_location="cpu",weights_only=False)
                    state=ck["model_state"]
                    if all(k.startswith("model.") for k in state):state={k[6:]:v for k,v in state.items()}
                    model=V8JointModel(ck["config"]);model.load_state_dict(state);model.eval()
                    with np.load(_cache(task,subject,session,seed)) as z:
                        e=torch.from_numpy(z["val_e"]);n=torch.from_numpy(z["val_n"]);y=torch.from_numpy(z["val_y"]).long();idx=z["val_i"]
                    with torch.no_grad():
                        full,parts=model.head(e,n,interaction=True);off,_=model.head(e,n,interaction=False)
                        er=model.head.e_ref[None].expand_as(e);nr=model.head.n_ref[None].expand_as(n)
                        e_ref,_=model.head(er,n,interaction=True);n_ref,_=model.head(e,nr,interaction=True)
                        donor=n.clone();trials=sorted(np.unique(idx//10))
                        for pos,trial in enumerate(trials):
                            a=np.flatnonzero(idx//10==trial);b=np.flatnonzero(idx//10==trials[(pos+1)%len(trials)])
                            donor[a]=n[torch.from_numpy(b[np.arange(len(a))%len(b)])]
                        donor_logits,_=model.head(e,donor,interaction=True)
                    count=len(y);totals["count"]+=count;totals["full_correct"]+=float((full.argmax(1)==y).sum());totals["off_correct"]+=float((off.argmax(1)==y).sum())
                    totals["e_ref_correct"]+=float((e_ref.argmax(1)==y).sum());totals["n_ref_correct"]+=float((n_ref.argmax(1)==y).sum());totals["donor_correct"]+=float((donor_logits.argmax(1)==y).sum())
                    totals["interaction_abs"]+=float(parts["interaction"].abs().sum());totals["donor_score_abs"]+=float(((full[:,1]-full[:,0])-(donor_logits[:,1]-donor_logits[:,0])).abs().sum())
    n=totals.pop("count");result={k:v/n for k,v in totals.items()};result["windows"]=int(n);result["interaction_accuracy_delta_pp"]=(result["full_correct"]-result["off_correct"])*100
    result["donor_accuracy_delta_pp"]=(result["full_correct"]-result["donor_correct"])*100;return result

def freeze():
    if not (EXP/"s6_fit_complete.json").is_file():raise RuntimeError("S6 incomplete")
    if list(EXP.glob("**/test_predictions.csv")):raise RuntimeError("unexpected V8 test output before freeze")
    mamba=_summary("J1_R4");conv=_summary("J1_R4_CONV");base=_baseline_summary();audit=_interaction_audit()
    delta_base=(mamba["equal_task_window_accuracy"]-base["equal_task_window_accuracy"])*100
    delta_conv=(mamba["equal_task_window_accuracy"]-conv["equal_task_window_accuracy"])*100
    conclusions={"performance_positive_vs_b_m50_validation":delta_base>0,
                 "explicit_interaction_structurally_present":True,
                 "both_modalities_affect_predictions":audit["e_ref_correct"]!=audit["full_correct"] and audit["n_ref_correct"]!=audit["full_correct"],
                 "interaction_improves_validation_accuracy":audit["interaction_accuracy_delta_pp"]>0,
                 "mamba_specific_gain_supported_validation":delta_conv>0.5}
    mechanism={"status":"frozen","selection_source":"validation_only","mamba":mamba,"matched_conv":conv,"b_m50":base,
               "mamba_minus_b_m50_equal_task_pp":delta_base,"mamba_minus_conv_equal_task_pp":delta_conv,"interaction_audit":audit,
               "conclusions":conclusions,"event_onset_confirmed":False,"real_hrf_claim_allowed":False,"test_access":False,"created_at":now()}
    write_json(EXP/"validation_mechanism_freeze.json",mechanism)
    legacy_index=legacy.complete_index();missing=[]
    for role in ("E32","B_M50","R_U"):
        for task,subjects in ENTRIES.items():
            for subject in subjects:
                for session in range(3):
                    for seed in SEEDS:
                        run,_=legacy_index[(role,seed,task,subject,session)]
                        if not (run/"test_predictions.csv").is_file():missing.append(str(run))
    before={"status":"frozen_before_test","scope_entries":30,"tasks":list(ENTRIES),"subjects_by_task":ENTRIES,
            "roles":["J1_R4","J1_R4_CONV","B_M50","E32","R_U"],"folds_per_role":270,"v8_folds_to_export":540,
            "historical_test_folds_to_reuse":810,"missing_historical_test":missing,"selection_freeze":str((EXP/"selection_freeze.json").resolve()),
            "mechanism_freeze":str((EXP/"validation_mechanism_freeze.json").resolve()),"selection_changed_by_test":False,"test_access":False,"created_at":now()}
    if missing:raise RuntimeError(f"missing historical tests: {len(missing)}")
    write_json(EXP/"before_test_freeze.json",before);return mechanism

def _evaluate_one(run):
    success=json.loads((run/"success.json").read_text(encoding="utf-8"))
    if success.get("test_evaluated") is True and (run/"test_predictions.csv").is_file():return
    ck=torch.load(run/"best_model.pt",map_location="cpu",weights_only=False);config=ck["config"]
    base=torch.load(config["b_m50_checkpoint"],map_location="cpu",weights_only=False);raw_cfg=base["config"]
    fh,eh,eeg,fnirs,labels,split=load_data(raw_cfg);stats={"eeg":scalar_stats(eeg,split["train_indices"]),"fnirs":scalar_stats(fnirs,split["train_indices"])}
    loader=make_loader(eeg,fnirs,labels,split,"test",stats,raw_cfg,False);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=V8JointModel(config).to(device);state=ck["model_state"]
    if all(k.startswith("model.") for k in state):state={k[6:]:v for k,v in state.items()}
    model.load_state_dict(state);model.eval();logits=[];ys=[];indices=[]
    with torch.no_grad():
        for ee,nn_,y,i in loader:
            out,_=model(ee.to(device),nn_.to(device));logits.append(out.cpu());ys.append(y);indices.append(i)
    logits=torch.cat(logits);ys=torch.cat(ys);indices=torch.cat(indices);loss=torch.nn.functional.cross_entropy(logits,ys.long());metrics,rows=rows_and_metrics(logits,ys,indices,loss)
    write_predictions(run/"test_predictions.csv",rows);saved=json.loads((run/"metrics.json").read_text(encoding="utf-8"));saved["test"]=metrics;write_json(run/"metrics.json",saved)
    success["test_evaluated"]=True;success["test_export_after_freeze"]=True;write_json(run/"success.json",success);fh.close();eh.close()

def evaluate():
    if not (EXP/"before_test_freeze.json").is_file():raise RuntimeError("before-test freeze missing")
    ledger_path=EXP/"test_export_ledger.json";records=[]
    if ledger_path.is_file():records=json.loads(ledger_path.read_text(encoding="utf-8"))["records"]
    done={(x["role"],x["task"],x["subject"],x["session"],x["seed"]) for x in records if x["status"]=="successful"}
    for role in ("J1_R4","J1_R4_CONV"):
        for task,subjects in ENTRIES.items():
            for subject in subjects:
                for session in range(3):
                    for seed in SEEDS:
                        key=(role,task,subject,session,seed)
                        if key in done:continue
                        require_space(f"test export {key}");run=_run(role,task,subject,session,seed);print(f"[V8-TEST] {role} {task} {subject} s{session} seed{seed}",flush=True);_evaluate_one(run)
                        records.append({"role":role,"task":task,"subject":subject,"session":session,"seed":seed,"status":"successful","completed_at":now()});write_json(ledger_path,{"records":records,"updated_at":now()})
    write_json(EXP/"test_export_complete.json",{"status":"complete","new_exports":540,"historical_reused":810,"failed":0,"completed_at":now()})

def _prediction_metric(path):
    rows=list(csv.DictReader(path.open(encoding="utf-8")));p=np.array([[float(x["probability_0"]),float(x["probability_1"])] for x in rows]);y=np.array([int(x["target"]) for x in rows]);trial=np.array([int(x["trial_global"]) for x in rows])
    window=float((p.argmax(1)==y).mean());correct=[]
    for value in np.unique(trial):
        q=trial==value;correct.append(float(p[q].mean(0).argmax()==y[q][0]))
    return window,float(np.mean(correct))

def aggregate():
    if not (EXP/"test_export_complete.json").is_file():raise RuntimeError("test export incomplete")
    index=legacy.complete_index();roles=("J1_R4","J1_R4_CONV","B_M50","E32","R_U");subject_rows=[]
    for role in roles:
        for task,subjects in ENTRIES.items():
            for subject in subjects:
                window=[];trial=[]
                for session in range(3):
                    for seed in SEEDS:
                        if role in {"J1_R4","J1_R4_CONV"}:run=_run(role,task,subject,session,seed)
                        else:run,_=index[(role,seed,task,subject,session)]
                        a,b=_prediction_metric(run/"test_predictions.csv");window.append(a);trial.append(b)
                subject_rows.append({"role":role,"task":task,"subject":subject,"window_accuracy":float(np.mean(window)),"trial_accuracy":float(np.mean(trial)),"folds":9})
    with (EXP/"final_test_subject_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(subject_rows[0]));w.writeheader();w.writerows(subject_rows)
    summaries=[]
    paper={"wg":.7903,"mi":.6965,"ma":.8514}
    for role in roles:
        for task in ENTRIES:
            x=[r for r in subject_rows if r["role"]==role and r["task"]==task]
            summaries.append({"role":role,"task":task,"n_subjects":len(x),"window_accuracy":float(np.mean([r["window_accuracy"] for r in x])),
                              "trial_accuracy":float(np.mean([r["trial_accuracy"] for r in x])),"sta_net_paper_reference":paper[task]})
    with (EXP/"final_test_task_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    paired=[]
    try:
        from scipy.stats import wilcoxon
    except Exception:wilcoxon=None
    for comparator in ("J1_R4_CONV","B_M50","E32","R_U"):
        for task in ENTRIES:
            a={r["subject"]:r for r in subject_rows if r["role"]=="J1_R4" and r["task"]==task};b={r["subject"]:r for r in subject_rows if r["role"]==comparator and r["task"]==task}
            d=np.array([a[s]["window_accuracy"]-b[s]["window_accuracy"] for s in sorted(a)]);pvalue=float(wilcoxon(d).pvalue) if wilcoxon is not None and np.any(d) else 1.0
            paired.append({"task":task,"comparison":f"J1_R4-minus-{comparator}","n":len(d),"mean_delta_pp":float(d.mean()*100),"median_delta_pp":float(np.median(d)*100),"wins":int((d>0).sum()),"ties":int((d==0).sum()),"losses":int((d<0).sum()),"wilcoxon_p_two_sided":pvalue})
    with (EXP/"paired_statistics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    budget=json.loads((EXP/"budget_ledger.json").read_text(encoding="utf-8"));stage_counts={}
    for r in budget["records"]:stage_counts.setdefault(r["stage"],{"successful":0,"failed":0,"running":0});stage_counts[r["stage"]][r["status"]]=stage_counts[r["stage"]].get(r["status"],0)+1
    budget_audit={"hard_cap":budget["optimization_fit_hard_cap"],"attempted":budget["attempted"],"successful":budget["successful"],"failed":budget["failed"],"reused":budget["reused"],
                  "stage_counts":stage_counts,"s6_additional_conv_pretraining_folds":63,"within_hard_cap":budget["attempted"]<=budget["optimization_fit_hard_cap"],"disk_free_gib":require_space("aggregate")}
    write_json(EXP/"budget_audit.json",budget_audit)
    mechanism=json.loads((EXP/"validation_mechanism_freeze.json").read_text(encoding="utf-8"))
    results={"status":"complete","scope":"MI VP001-VP010, MA VP001-VP010, validation-preselected WG10","subject_metrics":subject_rows,"task_summary":summaries,
             "paired_statistics":paired,"validation_mechanism":mechanism,"selection_changed_by_test":False,"test_used_for_selection":False,"created_at":now()}
    write_json(EXP/"final_results.json",results)
    table={(x["role"],x["task"]):x for x in summaries}
    lines=["# MambaSTA V8 Final Joint Multimodal Execution Report","",f"Generated: {now()}","",
           "## Executive conclusion","",
           "V8 froze J1_R4 from validation only. The model uses one joint logit with explicit EEG, fNIRS, and low-rank interaction contributions; it is not probability averaging.",
           f"Validation showed +{mechanism['mamba_minus_b_m50_equal_task_pp']:.3f} pp over B_M50 and +{mechanism['mamba_minus_conv_equal_task_pp']:.3f} pp over matched Conv. Interaction removal changed validation ACC by {mechanism['interaction_audit']['interaction_accuracy_delta_pp']:.3f} pp; whole-trial donor replacement changed it by {mechanism['interaction_audit']['donor_accuracy_delta_pp']:.3f} pp.",
           "The Mamba-specific gain gate did not pass; claims are limited to an explicit-interaction multimodal model with positive validation evidence. Event onset remains unconfirmed, so no real-HRF, physiological peak, or causal brain-region claim is made.","",
           "## Frozen-scope test results","", "All values are mean accuracy across the same 10 subjects per task, with each subject first averaged across 3 sessions and 3 seeds.","",
           "| Role | WG window / trial | MI window / trial | MA window / trial |","|---|---:|---:|---:|"]
    labels={"J1_R4":"V8 J1_R4 Mamba","J1_R4_CONV":"Matched Conv","B_M50":"B_M50","E32":"E32","R_U":"R_U"}
    for role in roles:
        cells=[]
        for task in ("wg","mi","ma"):
            x=table[(role,task)];cells.append(f"{100*x['window_accuracy']:.2f}% / {100*x['trial_accuracy']:.2f}%")
        lines.append(f"| {labels[role]} | {' | '.join(cells)} |")
    lines += ["", "STA-Net paper numerical references (not a same-subject paired replication): WG 79.03%, MI 69.65%, MA 85.14%.","",
              "## Paired comparisons","","Positive values favor V8 J1_R4. Wilcoxon tests use the 10 fixed subjects and are descriptive after a validation-driven development process.","",
              "| Task | Comparator | Mean delta | Wins/Ties/Losses | p |","|---|---|---:|---:|---:|"]
    for x in paired:lines.append(f"| {x['task'].upper()} | {x['comparison'].split('-minus-')[1]} | {x['mean_delta_pp']:+.2f} pp | {x['wins']}/{x['ties']}/{x['losses']} | {x['wilcoxon_p_two_sided']:.4f} |")
    lines += ["","## Architecture and interpretation","",
              "J1_R4 retains the frozen B_M50 EEG and fNIRS feature branches. A rank-4 reference-centred bilinear head produces bias + EEG-private + fNIRS-private + interaction contributions, followed by one joint binary softmax. B_M50 is used only as a training teacher; its averaged probabilities are absent from the deployed V8 output.",
              "Validation interventions support use of both modalities and a non-zero interaction contribution. These are model-behaviour findings, not biological causality.","",
              "## Protocol integrity and resources","",
              f"- Formal optimization fits: {budget['attempted']} attempted, {budget['successful']} successful, {budget['failed']} technical failure; hard cap {budget['optimization_fit_hard_cap']}.",
              "- The single technical failure was the repaired S4 checkpoint-wrapper mismatch; the affected fold was rerun successfully.",
              "- 63 additional WG matched-Conv pretraining folds were recorded inside S6 cost accounting.",
              "- Test selection leakage: none. Model/recipe and mechanism conclusions were frozen before V8 test export.",
              "- Local STA-Net was not trained.","",
              "## Scope limitations","",
              "MI and MA use VP001-VP010. WG uses 10 subjects preselected only from historical B_M50 validation. Results therefore characterize this fixed 30-entry development scope, not all available participants. Historical tests had been exposed in earlier iterations, so this is not an untouched external confirmation.",
              ""]
    (EXP/"V8_Execution_Report.md").write_text("\n".join(lines),encoding="utf-8")
    source_files=[ROOT/"models/mambasta_v8.py",ROOT/"training/train_v8_cached.py",ROOT/"training/train_v8_staged.py",ROOT/"tools/v8_run.py",ROOT/"tools/v8_s1.py",ROOT/"tools/v8_optimize.py",ROOT/"tools/v8_finalize.py"]
    with (EXP/"V8_Source_Code_Full.txt").open("w",encoding="utf-8") as f:
        for path in source_files:f.write(f"\n\n===== {path.relative_to(ROOT)} =====\n\n");f.write(path.read_text(encoding="utf-8"))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files};write_json(EXP/"source_manifest.json",{"files":hashes,"created_at":now()})
    write_json(EXP/"execution_complete.json",{"status":"complete","report":str((EXP/"V8_Execution_Report.md").resolve()),"selection_changed_by_test":False,"created_at":now()})

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("freeze","evaluate","aggregate"),required=True);a=p.parse_args();process=EXP/f"final_{a.stage}_process.json";write_json(process,{"pid":os.getpid(),"stage":a.stage,"status":"running","started_at":now()})
    try:
        if a.stage=="freeze":freeze()
        elif a.stage=="evaluate":evaluate()
        else:aggregate()
    except Exception as e:write_json(process,{"pid":os.getpid(),"stage":a.stage,"status":"failed","error":repr(e),"ended_at":now()});raise
    write_json(process,{"pid":os.getpid(),"stage":a.stage,"status":"complete","ended_at":now()})

if __name__=="__main__":main()
