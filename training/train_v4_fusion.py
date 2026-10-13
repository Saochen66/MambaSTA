"""Validation-only one-fold trainer for V4 frozen-anchor fusion models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mambasta_v4_fusion import make_v4_fusion
from training.train_v2 import make_loader, normalization_hash, parameter_table, run_epoch, scalar_stats, set_model_seed, write_predictions


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def load_data(config):
    fnirs_handle = np.load(config["fnirs_cache_path"], allow_pickle=False)
    eeg_handle = np.load(config["eeg_cache_path"], allow_pickle=False)
    if str(fnirs_handle["complete_sha256"].item()) != config["fnirs_complete_sha256"]:
        raise RuntimeError("fNIRS cache signature mismatch")
    if not np.array_equal(fnirs_handle["label"], eeg_handle["label"]):
        raise RuntimeError("EEG/fNIRS label mismatch")
    split = {key: eeg_handle[key] for key in ("train_indices", "val_indices", "test_indices")}
    return fnirs_handle, eeg_handle, eeg_handle["eeg"], fnirs_handle["fnirs"], fnirs_handle["label"], split


def fit(config, run_dir):
    started = time.time(); set_model_seed(config["model_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fh, eh, eeg, fnirs, labels, split = load_data(config)
    stats = {"eeg": scalar_stats(eeg, split["train_indices"]), "fnirs": scalar_stats(fnirs, split["train_indices"])}
    train_loader = make_loader(eeg, fnirs, labels, split, "train", stats, config, True)
    val_loader = make_loader(eeg, fnirs, labels, split, "val", stats, config, False)
    model = make_v4_fusion(config["architecture"], dim=config["d_model"], state=config["d_state"], conv_kernel=config["d_conv"], expand=config["expand"], rank=config["query_rank"], dropout=config["dropout"]).to(device)
    checkpoint = torch.load(config["teacher_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["config"]["signature"] != config["teacher_run_signature"]:
        raise RuntimeError("teacher checkpoint signature mismatch")
    model.load_teacher_state(checkpoint["model_state"])
    teacher_before = state_hash(model.teacher.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(); history=[]; stale=0; best_loss=float("inf"); best_state=best_epoch=best_val=None
    for epoch in range(1, config["max_epochs"]+1):
        train_metrics,_=run_epoch(model,train_loader,criterion,device,optimizer,config["gradient_clip"])
        with torch.no_grad(): val_metrics,_=run_epoch(model,val_loader,criterion,device)
        history.append({"epoch":epoch,"train":train_metrics,"val":val_metrics})
        print(f"[V4-S2] epoch={epoch}/{config['max_epochs']} train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f}",flush=True)
        if val_metrics["loss"]<best_loss:
            best_loss=val_metrics["loss"];best_epoch=epoch;best_val=val_metrics;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else: stale+=1
        if stale>=config["patience"]: break
    teacher_after=state_hash(model.teacher.state_dict())
    if teacher_before != teacher_after: raise RuntimeError("frozen teacher changed during training")
    model.load_state_dict(best_state)
    with torch.no_grad(): selected_val,predictions=run_epoch(model,val_loader,criterion,device)
    run_dir.mkdir(parents=True,exist_ok=True)
    normalization={"eeg_mean":stats["eeg"][0],"eeg_std":stats["eeg"][1],"fnirs_mean":stats["fnirs"][0],"fnirs_std":stats["fnirs"][1]};normalization["sha256"]=normalization_hash(normalization)
    environment={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"parameters_total":sum(p.numel() for p in model.parameters()),"parameters_trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"elapsed_seconds":time.time()-started,"teacher_state_sha256_before":teacher_before,"teacher_state_sha256_after":teacher_after,"teacher_exactly_unchanged":teacher_before==teacher_after}
    metrics={"best_epoch":best_epoch,"epochs_completed":len(history),"best_val":best_val,"selected_checkpoint_val":selected_val,"test":None}
    split_json={key:[int(x) for x in value] for key,value in split.items()}
    for name,payload in (("config.json",config),("environment.json",environment),("split.json",split_json),("normalization.json",normalization),("history.json",history),("metrics.json",metrics),("module_parameters.json",parameter_table(model))): (run_dir/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    write_predictions(run_dir/"validation_predictions.csv",predictions)
    torch.save({"model_state":best_state,"config":config,"metrics":metrics},run_dir/"best_model.pt")
    (run_dir/"success.json").write_text(json.dumps({"signature":config["signature"],"fit_complete":True,"test_evaluated":False,"teacher_exactly_unchanged":True,"failure":None},ensure_ascii=False,indent=2),encoding="utf-8")
    fh.close();eh.close()


def evaluate(config, run_dir):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(run_dir/"best_model.pt",map_location="cpu",weights_only=False)
    if checkpoint["config"]["signature"] != config["signature"]: raise RuntimeError("checkpoint signature mismatch")
    fh,eh,eeg,fnirs,labels,split=load_data(config)
    normalization=json.loads((run_dir/"normalization.json").read_text(encoding="utf-8"));stats={"eeg":(normalization["eeg_mean"],normalization["eeg_std"]),"fnirs":(normalization["fnirs_mean"],normalization["fnirs_std"])}
    loader=make_loader(eeg,fnirs,labels,split,"test",stats,config,False)
    model=make_v4_fusion(config["architecture"],dim=config["d_model"],state=config["d_state"],conv_kernel=config["d_conv"],expand=config["expand"],rank=config["query_rank"],dropout=config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"]);model.eval()
    with torch.no_grad():metrics,predictions=run_epoch(model,loader,nn.CrossEntropyLoss(),device)
    saved=json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"));saved["test"]=metrics
    (run_dir/"metrics.json").write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding="utf-8");write_predictions(run_dir/"test_predictions.csv",predictions)
    success=json.loads((run_dir/"success.json").read_text(encoding="utf-8"));success["test_evaluated"]=True
    (run_dir/"success.json").write_text(json.dumps(success,ensure_ascii=False,indent=2),encoding="utf-8")
    fh.close();eh.close();print(f"[V4] test={metrics['loss']:.4f}/{metrics['accuracy']:.4f} trial={metrics['trial']['accuracy']:.4f}",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("fit","evaluate"),default="fit");parser.add_argument("--config",required=True);parser.add_argument("--run-dir",required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"));(fit if args.mode=="fit" else evaluate)(config,Path(args.run_dir))


if __name__=="__main__": main()
