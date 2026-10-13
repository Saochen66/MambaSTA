"""Train one V8 joint head from a strictly train/validation-only feature cache."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from models.mambasta_v8 import V8JointModel, joint_training_loss
from training.train_v2 import aggregate_trials, confusion_metrics, set_model_seed


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rows_and_metrics(logits, targets, indices, loss):
    probability = logits.softmax(-1).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy(); indices = indices.detach().cpu().numpy()
    predictions = probability.argmax(1); confusion = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(targets, predictions): confusion[int(actual), int(predicted)] += 1
    metrics = {"loss": float(loss), **confusion_metrics(confusion),
               "trial": aggregate_trials(probability, targets, indices)}
    rows = []
    raw = logits.detach().cpu().numpy()
    for idx, target, prob, pred, score in zip(indices, targets, probability, predictions, raw):
        rows.append({"model_index":int(idx), "trial_global":int(idx)//10,
                     "session":int(idx)//200, "window_in_trial":int(idx)%10,
                     "target":int(target), "raw_0":float(score[0]), "raw_1":float(score[1]),
                     "raw_type":"logit", "probability_0":float(prob[0]),
                     "probability_1":float(prob[1]), "prediction":int(pred)})
    return metrics, rows


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


class CachedJoint(nn.Module):
    def __init__(self, config, checkpoint):
        super().__init__()
        self.model = V8JointModel(config)
        self.model.load_b_m50_state(checkpoint["model_state"]); self.model.freeze_backbone()

    def fit_reference(self, data):
        self.model.head.fit_reference(data["e"], data["n"])
        role = self.model.role
        last = self.model.backbone.teacher.classifier[-1]
        if hasattr(self.model.head, "initialize_from_eeg_last_linear"):
            self.model.head.initialize_from_eeg_last_linear(last)
        elif hasattr(self.model.head, "base"):
            self.model.head.base.initialize_from_eeg_last_linear(last)
        if self.model.conditional_read is not None:
            self.model.conditional_read.fit_reference(data["q"])
        if role == "J3_P4":
            self.model.head.set_prototypes(data["e"], data["n"], data["y"], data["i"]//10)

    def forward_cached(self, batch):
        n = batch["n"]
        if self.model.conditional_read is not None:
            valid = torch.ones_like(batch["w"], dtype=torch.bool)
            weights, _ = self.model.conditional_read(batch["q"], batch["m"], batch["w"], valid)
            values = self.model.backbone.readout.value(batch["m"])
            regions = torch.einsum("brt,brtd->brd", weights, values) + self.model.backbone.region_embedding[None]
            region_weights = self.model.backbone.region_score(regions).squeeze(-1).softmax(-1)
            feature = torch.einsum("br,brd->bd", region_weights, regions)
            n = self.model._n_main(feature)
        return self.model.head(batch["e"], n)


def tensor_split(data, prefix, device):
    return {key: torch.from_numpy(data[f"{prefix}_{key}"]).to(device) for key in ("e","n","q","m","w","b","r","y","i")}


def subset(data, positions): return {key:value[positions] for key,value in data.items()}


def fit(config, run_dir):
    started=time.time(); set_model_seed(config["model_seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(config["feature_cache_path"], allow_pickle=False) as cache:
        train=tensor_split(cache,"train",device); val=tensor_split(cache,"val",device)
    checkpoint=torch.load(config["b_m50_checkpoint"],map_location="cpu",weights_only=False)
    if checkpoint["config"]["signature"] != config["b_m50_signature"]: raise RuntimeError("B_M50 identity mismatch")
    model=CachedJoint(config,checkpoint).to(device); model.fit_reference(train)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config["learning_rate"],weight_decay=config["weight_decay"])
    generator=torch.Generator(device="cpu").manual_seed(config["model_seed"])
    best_loss=float("inf"); best_state=None; best_epoch=0; stale=0; history=[]
    role=config["config_id"]
    kd_weight=0.0 if role in {"J1_NoKD","J3_P4","J4_S8","C_Add","C_Concat"} else config["kd_weight"]
    penalty=0.0 if role=="J1_Reg0" else .01 if role=="J1_Reg1e2" else config["interaction_penalty"]
    for epoch in range(1,config["max_epochs"]+1):
        model.train(); order=torch.randperm(len(train["y"]),generator=generator,device="cpu").to(device)
        total=0.0
        for start in range(0,len(order),config["batch_size"]):
            pos=order[start:start+config["batch_size"]]; batch=subset(train,pos)
            optimizer.zero_grad(set_to_none=True); logits,parts=model.forward_cached(batch)
            teacher=batch["r"] if role=="J1_RU_KD" else batch["b"]
            loss,_=joint_training_loss(logits,batch["y"].long(),teacher,parts["interaction"],kd_weight,config["temperature"],penalty)
            loss.backward(); nn.utils.clip_grad_norm_(parameters,config["gradient_clip"]); optimizer.step()
            total += float(loss.detach())*len(pos)
        model.eval()
        with torch.no_grad():
            val_logits,val_parts=model.forward_cached(val)
            val_nll=nn.functional.cross_entropy(val_logits,val["y"].long())
            val_metrics,_=rows_and_metrics(val_logits,val["y"],val["i"],val_nll)
        history.append({"epoch":epoch,"train_objective":total/len(order),"val":val_metrics})
        print(f"[V8-head] {role} epoch={epoch} train={total/len(order):.4f} val_nll={float(val_nll):.4f} acc={val_metrics['accuracy']:.4f}",flush=True)
        # Checkpoint selection is pure joint NLL, never KD or test.
        if float(val_nll) < best_loss-1e-8:
            best_loss=float(val_nll);best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=config["patience"]:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        logits,parts=model.forward_cached(val);nll=nn.functional.cross_entropy(logits,val["y"].long())
        metrics,rows=rows_and_metrics(logits,val["y"],val["i"],nll)
    run_dir.mkdir(parents=True,exist_ok=True)
    write_json(run_dir/"config.json",config);write_json(run_dir/"history.json",history)
    write_json(run_dir/"metrics.json",{"best_epoch":best_epoch,"epochs_completed":len(history),"selected_checkpoint_val":metrics,"test":None})
    write_json(run_dir/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "trainable_parameters":sum(p.numel() for p in parameters),"elapsed_seconds":time.time()-started})
    write_predictions(run_dir/"validation_predictions.csv",rows)
    torch.save({"model_state":best_state,"config":config,"metrics":metrics},run_dir/"best_model.pt")
    write_json(run_dir/"success.json",{"signature":config["signature"],"fit_complete":True,"test_evaluated":False,
        "pure_joint_nll_checkpoint":True,"failure":None})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--run-dir",required=True);args=parser.parse_args()
    fit(json.loads(Path(args.config).read_text(encoding="utf-8")),Path(args.run_dir))


if __name__=="__main__":main()
