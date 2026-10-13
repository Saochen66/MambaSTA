"""One-fold V5 trainer. Test evaluation is intentionally a separate mode."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from models.mambasta_v5 import make_mambasta_v5
from models.v5_heads import optimizer_groups, v5_loss
from training.train_v2 import IndexedDataset, aggregate_trials, confusion_metrics, make_loader, normalization_hash, parameter_table, scalar_stats, set_model_seed
from training.train_v4_fusion import load_data, state_hash


class TrialBatchSampler(Sampler):
    def __init__(self, model_indices, seed, trials_per_batch=2):
        self.model_indices=np.asarray(model_indices,dtype=np.int64); self.seed=int(seed); self.trials_per_batch=int(trials_per_batch)
        groups={}
        for position,index in enumerate(self.model_indices): groups.setdefault(int(index)//10,[]).append(position)
        if any(len(value)!=10 for value in groups.values()): raise ValueError("trial batches require complete 10-window trials")
        self.groups=groups
    def __iter__(self):
        generator=np.random.default_rng(self.seed); trials=generator.permutation(sorted(self.groups))
        for start in range(0,len(trials),self.trials_per_batch):
            chosen=trials[start:start+self.trials_per_batch]
            if len(chosen)==self.trials_per_batch: yield [position for trial in chosen for position in self.groups[int(trial)]]
    def __len__(self): return len(self.groups)//self.trials_per_batch


def make_trial_loader(eeg,fnirs,labels,split,stats,config):
    indices=split["train_indices"]
    dataset=IndexedDataset(eeg,fnirs,labels,indices,stats["eeg"],stats["fnirs"])
    return DataLoader(dataset,batch_sampler=TrialBatchSampler(indices,config["model_seed"],2),num_workers=0,pin_memory=torch.cuda.is_available())


def run_epoch_v5(model, loader, device, optimizer=None, gradient_clip=1.0, trial_loss=False):
    training = optimizer is not None; model.train(training)
    loss_sum = 0.0; probabilities = []; targets = []; indices = []; raw_outputs = []
    for eeg, fnirs, target, index in loader:
        eeg, fnirs, target = eeg.to(device), fnirs.to(device), target.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output, info = model(eeg, fnirs, return_diagnostics=True)
            loss = v5_loss(output, target, info, model.output_kind, index.to(device)//10 if trial_loss else None)
            if training:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), gradient_clip); optimizer.step()
        log_probability = output.detach() if model.output_kind == "probability" else torch.log_softmax(output.detach(), -1)
        probability = log_probability.exp()
        loss_sum += float(loss.detach()) * len(target)
        probabilities.append(probability.cpu().numpy()); targets.append(target.cpu().numpy()); indices.append(index.numpy()); raw_outputs.append(output.detach().cpu().numpy())
    probabilities, targets, indices, raw_outputs = map(np.concatenate, (probabilities, targets, indices, raw_outputs))
    predictions = probabilities.argmax(1); confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(targets, predictions): confusion[int(actual), int(predicted)] += 1
    metrics = {"loss": loss_sum / len(targets), **confusion_metrics(confusion), "trial": aggregate_trials(probabilities, targets, indices)}
    rows = []
    for index, target, probability, prediction, raw in zip(indices, targets, probabilities, predictions, raw_outputs):
        rows.append({"model_index": int(index), "trial_global": int(index)//10, "session": int(index)//200, "window_in_trial": int(index)%10, "target": int(target), "raw_0": float(raw[0]), "raw_1": float(raw[1]), "raw_type": "log_probability" if model.output_kind == "probability" else "logit", "probability_0": float(probability[0]), "probability_1": float(probability[1]), "prediction": int(prediction)})
    return metrics, rows


def write_json(path, payload): path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fit(config, run_dir):
    started = time.time(); set_model_seed(config["model_seed"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fnirs_handle, eeg_handle, eeg, fnirs, labels, split = load_data(config)
    stats = {"eeg": scalar_stats(eeg, split["train_indices"]), "fnirs": scalar_stats(fnirs, split["train_indices"])}
    train_loader = make_trial_loader(eeg,fnirs,labels,split,stats,config) if config["family"]=="trial_loss" else make_loader(eeg, fnirs, labels, split, "train", stats, config, True)
    val_loader = make_loader(eeg, fnirs, labels, split, "val", stats, config, False)
    model = make_mambasta_v5(config).to(device)
    checkpoint = torch.load(config["teacher_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["config"]["signature"] != config["teacher_run_signature"]: raise RuntimeError("teacher checkpoint signature mismatch")
    model.load_teacher_state(checkpoint["model_state"]); teacher_before = state_hash(model.teacher.state_dict())
    if config["family"]=="conditional_delta" and config.get("profile")=="eeg_constant":
        features=[]
        with torch.no_grad():
            for batch_eeg,_,_,_ in train_loader:
                features.append(model.teacher.eeg_encoder(batch_eeg.to(device).permute(0,4,1,2,3).contiguous()).mean(0))
        model.set_constant_eeg(torch.stack(features).mean(0))
    optimizer = torch.optim.AdamW(optimizer_groups(model, config["weight_decay"]), lr=config["learning_rate"])
    history=[]; stale=0; best_loss=float("inf"); best_state=best_epoch=best_val=None
    for epoch in range(1, config["max_epochs"]+1):
        train_metrics,_ = run_epoch_v5(model, train_loader, device, optimizer, config["gradient_clip"],trial_loss=config["family"]=="trial_loss")
        with torch.no_grad(): val_metrics,_ = run_epoch_v5(model, val_loader, device)
        history.append({"epoch":epoch,"train":train_metrics,"val":val_metrics}); print(f"[V5] {config['config_id']} epoch={epoch} train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f}/{val_metrics['accuracy']:.4f}",flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss=val_metrics["loss"]; best_epoch=epoch; best_val=val_metrics; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=config["patience"]: break
    if state_hash(model.teacher.state_dict()) != teacher_before: raise RuntimeError("frozen teacher changed")
    model.load_state_dict(best_state)
    with torch.no_grad(): selected_val,predictions=run_epoch_v5(model,val_loader,device)
    run_dir.mkdir(parents=True,exist_ok=True)
    normalization={"eeg_mean":stats["eeg"][0],"eeg_std":stats["eeg"][1],"fnirs_mean":stats["fnirs"][0],"fnirs_std":stats["fnirs"][1]}; normalization["sha256"]=normalization_hash(normalization)
    environment={"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"parameters_total":sum(p.numel() for p in model.parameters()),"parameters_trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"elapsed_seconds":time.time()-started,"teacher_state_sha256":teacher_before}
    metrics={"best_epoch":best_epoch,"epochs_completed":len(history),"best_val":best_val,"selected_checkpoint_val":selected_val,"test":None}
    for name,payload in (("config.json",config),("environment.json",environment),("split.json",{k:[int(x) for x in v] for k,v in split.items()}),("normalization.json",normalization),("history.json",history),("metrics.json",metrics),("module_parameters.json",parameter_table(model))): write_json(run_dir/name,payload)
    write_predictions(run_dir/"validation_predictions.csv",predictions); torch.save({"model_state":best_state,"config":config,"metrics":metrics},run_dir/"best_model.pt")
    write_json(run_dir/"success.json",{"signature":config["signature"],"fit_complete":True,"test_evaluated":False,"teacher_exactly_unchanged":True,"failure":None})
    fnirs_handle.close(); eeg_handle.close()


def evaluate(config, run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir/"best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"]["signature"] != config["signature"]:
        raise RuntimeError("checkpoint signature mismatch")
    fnirs_handle, eeg_handle, eeg, fnirs, labels, split = load_data(config)
    normalization = json.loads((run_dir/"normalization.json").read_text(encoding="utf-8"))
    stats = {"eeg": (normalization["eeg_mean"], normalization["eeg_std"]),
             "fnirs": (normalization["fnirs_mean"], normalization["fnirs_std"])}
    loader = make_loader(eeg, fnirs, labels, split, "test", stats, config, False)
    model = make_mambasta_v5(config).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): metrics, predictions = run_epoch_v5(model, loader, device)
    saved = json.loads((run_dir/"metrics.json").read_text(encoding="utf-8")); saved["test"] = metrics
    write_json(run_dir/"metrics.json", saved); write_predictions(run_dir/"test_predictions.csv", predictions)
    success = json.loads((run_dir/"success.json").read_text(encoding="utf-8")); success["test_evaluated"] = True
    write_json(run_dir/"success.json", success)
    fnirs_handle.close(); eeg_handle.close()
    print(f"[V5] test={metrics['loss']:.4f}/{metrics['accuracy']:.4f} trial={metrics['trial']['accuracy']:.4f}", flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("fit","evaluate"),default="fit"); parser.add_argument("--config",required=True); parser.add_argument("--run-dir",required=True); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    (fit if args.mode=="fit" else evaluate)(config,Path(args.run_dir))


if __name__=="__main__": main()
