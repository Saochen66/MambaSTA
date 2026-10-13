"""Validation-only T2/T3 continuation for a selected V8 joint model."""
from __future__ import annotations

import argparse, json, platform, sys, time, types
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from models.mambasta_v8 import V8Features, V8JointModel, joint_training_loss
from training.train_v2 import make_loader, scalar_stats, set_model_seed
from training.train_v4_fusion import load_data
from training.train_v8_cached import rows_and_metrics, write_json, write_predictions


def _lookup(cache,prefix,device):
    return {int(i):torch.from_numpy(v).to(device) for i,v in zip(cache[f"{prefix}_i"],cache[f"{prefix}_b"])}


def _teacher(indices,lookup):return torch.stack([lookup[int(i)] for i in indices.detach().cpu().numpy()])


def _set_trainable(model,recipe):
    model.requires_grad_(False)
    modules=[model.head]
    if model.conditional_read is not None:modules.append(model.conditional_read)
    modules += [model.backbone.readout,model.backbone.region_score,model.backbone.v5_head.n_head]
    if recipe=="T3":modules.append(model.backbone.teacher.classifier)
    for module in modules:module.requires_grad_(True)
    head=list(model.head.parameters())+(list(model.conditional_read.parameters()) if model.conditional_read is not None else [])
    eeg=list(model.backbone.teacher.classifier.parameters()) if recipe=="T3" else []
    excluded={id(x) for x in head+eeg};n=[p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
    return head,n,eeg


def _enable_eeg_tail_gradient(model):
    """T3-only feature path: frozen encoder, differentiable EEG classifier tail."""
    def extract(this,eeg,fnirs,include_teacher_probability=False):
        this.backbone.teacher.eeg_encoder.eval()
        with torch.no_grad():
            teacher_feature=this.backbone.teacher.eeg_encoder(eeg.permute(0,4,1,2,3).contiguous())
        teacher_logits=this.backbone.teacher.classifier(teacher_feature)
        this.backbone._teacher_feature=torch.zeros_like(teacher_feature)
        n_feature,path=this.backbone.encode_v5_regions(fnirs);memory=path["memory"];base_weights=path["weights"]
        logp=None
        if include_teacher_probability:logp,_=this.backbone.v5_head(teacher_logits,n_feature)
        return V8Features(e_main=this._eeg_main(teacher_feature),n_main=this._n_main(n_feature),e_query=teacher_feature,
                          teacher_logits=teacher_logits,b_m50_log_probability=logp,memory=memory,base_weights=base_weights)
    model.extract_features=types.MethodType(extract,model)


def _epoch(model,loader,device,lookup,optimizer,config):
    training=optimizer is not None;model.train(training)
    # Frozen feature extractors stay in eval mode so BN/dropout buffers cannot
    # drift merely because the selected tail is being fine-tuned.
    model.backbone.teacher.eeg_encoder.eval();model.backbone.patch_encoder.eval();model.backbone.mixer.eval()
    if config["recipe"]=="T2":model.backbone.teacher.eval()
    total=0.;logits_all=[];targets=[];indices=[]
    for eeg,fnirs,y,i in loader:
        eeg,fnirs,y=eeg.to(device),fnirs.to(device),y.to(device);teacher=_teacher(i,lookup)
        if training:optimizer.zero_grad(set_to_none=True)
        logits,parts=model(eeg,fnirs,return_diagnostics=True)
        loss,_=joint_training_loss(logits,y.long(),teacher,parts["interaction"],config["kd_weight"],config["temperature"],config["interaction_penalty"])
        if training:loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config["gradient_clip"]);optimizer.step()
        total+=float(loss.detach())*len(y);logits_all.append(logits.detach());targets.append(y.detach());indices.append(i.to(device))
    logits=torch.cat(logits_all);target=torch.cat(targets);index=torch.cat(indices);nll=nn.functional.cross_entropy(logits,target.long())
    metrics,rows=rows_and_metrics(logits,target,index,nll);metrics["training_objective"]=total/len(target)
    return metrics,rows


def fit(config,run_dir):
    started=time.time();set_model_seed(config["model_seed"]);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fh,eh,eeg,fnirs,labels,split=load_data(config);stats={"eeg":scalar_stats(eeg,split["train_indices"]),"fnirs":scalar_stats(fnirs,split["train_indices"])}
    train=make_loader(eeg,fnirs,labels,split,"train",stats,config,True);val=make_loader(eeg,fnirs,labels,split,"val",stats,config,False)
    prior=torch.load(config["source_checkpoint"],map_location="cpu",weights_only=False)
    if prior["config"]["signature"]!=config["source_signature"]:raise RuntimeError("source checkpoint identity mismatch")
    model=V8JointModel(config).to(device)
    prior_state=prior["model_state"]
    # T1 checkpoints are emitted by CachedJoint, which owns the deployment
    # model under its ``model`` attribute. Strip that wrapper identity only.
    if prior_state and all(name.startswith("model.") for name in prior_state):
        prior_state={name[len("model."):]:value for name,value in prior_state.items()}
    model.load_state_dict(prior_state,strict=True)
    if config["recipe"]=="T3":_enable_eeg_tail_gradient(model)
    head,n,eeg_params=_set_trainable(model,config["recipe"])
    groups=[{"params":head,"lr":config["head_lr"]},{"params":n,"lr":config["n_lr"]}]
    if eeg_params:groups.append({"params":eeg_params,"lr":config["e_lr"]})
    optimizer=torch.optim.AdamW(groups,weight_decay=config["weight_decay"])
    with np.load(config["feature_cache_path"],allow_pickle=False) as cache:
        train_teacher=_lookup(cache,"train",device);val_teacher=_lookup(cache,"val",device)
    best=float("inf");best_state=None;best_epoch=0;stale=0;history=[]
    for epoch in range(1,config["max_epochs"]+1):
        tr,_=_epoch(model,train,device,train_teacher,optimizer,config)
        with torch.no_grad():va,_=_epoch(model,val,device,val_teacher,None,config)
        history.append({"epoch":epoch,"train":tr,"val":va});print(f"[V8-{config['recipe']}] {config['config_id']} epoch={epoch} val_nll={va['loss']:.4f} acc={va['accuracy']:.4f}",flush=True)
        if va["loss"]<best-1e-8:best=va["loss"];best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=config["patience"]:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():metrics,rows=_epoch(model,val,device,val_teacher,None,config)
    run_dir.mkdir(parents=True,exist_ok=True);write_json(run_dir/"config.json",config);write_json(run_dir/"history.json",history)
    write_json(run_dir/"metrics.json",{"best_epoch":best_epoch,"epochs_completed":len(history),"selected_checkpoint_val":metrics,"test":None})
    write_json(run_dir/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"elapsed_seconds":time.time()-started,"recipe":config["recipe"]})
    write_predictions(run_dir/"validation_predictions.csv",rows);torch.save({"model_state":best_state,"config":config,"metrics":metrics},run_dir/"best_model.pt")
    write_json(run_dir/"success.json",{"signature":config["signature"],"fit_complete":True,"test_evaluated":False,"pure_joint_nll_checkpoint":True,"failure":None})
    fh.close();eh.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True);a=p.parse_args();fit(json.loads(Path(a.config).read_text(encoding="utf-8")),Path(a.run_dir))

if __name__=="__main__":main()
