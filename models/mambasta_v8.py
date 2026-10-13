"""MambaSTA V8 joint heads built on an unchanged V5 B_M50 backbone.

The deployment output is a single joint classifier.  B_M50 probabilities are
exposed only as a training teacher and never enter the deployed joint logits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from models.mambasta_v5 import make_mambasta_v5
from models.mambasta_v4 import frozen_eeg_forward


class ReferenceInteractionHead(nn.Module):
    """Reference-centred low-rank joint binary classifier."""

    def __init__(self, e_dim=32, n_dim=32, rank=4):
        super().__init__()
        self.register_buffer("e_ref", torch.zeros(e_dim))
        self.register_buffer("n_ref", torch.zeros(n_dim))
        self.register_buffer("e_scale", torch.ones(e_dim))
        self.register_buffer("n_scale", torch.ones(n_dim))
        self.e_main = nn.Linear(e_dim, 1, bias=False)
        self.n_main = nn.Linear(n_dim, 1, bias=False)
        self.e_factor = nn.Linear(e_dim, rank, bias=False)
        self.n_factor = nn.Linear(n_dim, rank, bias=False)
        self.cross_weight = nn.Parameter(torch.zeros(rank))
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = int(rank)

    @torch.no_grad()
    def fit_reference(self, e, n, weights=None):
        if e.ndim != 2 or n.ndim != 2 or len(e) != len(n):
            raise ValueError("reference features must be aligned [N,D]")
        if weights is None:
            weights = e.new_ones(len(e))
        weights = weights.to(e).reshape(-1)
        if len(weights) != len(e) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("invalid reference weights")
        weights = weights / weights.sum()
        e_mean = (weights[:, None] * e).sum(0)
        n_mean = (weights[:, None] * n).sum(0)
        e_var = (weights[:, None] * (e - e_mean).square()).sum(0)
        n_var = (weights[:, None] * (n - n_mean).square()).sum(0)
        self.e_ref.copy_(e_mean)
        self.n_ref.copy_(n_mean)
        self.e_scale.copy_(e_var.sqrt().clamp_min(1e-3))
        self.n_scale.copy_(n_var.sqrt().clamp_min(1e-3))

    @torch.no_grad()
    def initialize_from_eeg_last_linear(self, last_linear):
        if not isinstance(last_linear, nn.Linear) or last_linear.out_features != 2:
            raise ValueError("expected an old binary Linear layer")
        if last_linear.in_features != self.e_ref.numel():
            raise ValueError("EEG last-layer input dimension mismatch")
        weight = (last_linear.weight[1] - last_linear.weight[0]).to(self.e_ref)
        bias = (last_linear.bias[1] - last_linear.bias[0]).to(self.bias)
        self.e_main.weight.copy_((weight * self.e_scale)[None])
        self.n_main.weight.zero_()
        self.cross_weight.zero_()
        self.bias.copy_(bias + (weight * self.e_ref).sum())

    def forward(self, e, n, interaction=True):
        ee = (e - self.e_ref) / self.e_scale
        nn_ = (n - self.n_ref) / self.n_scale
        e_score = self.e_main(ee).squeeze(-1)
        n_score = self.n_main(nn_).squeeze(-1)
        rank_terms = self.e_factor(ee) * self.n_factor(nn_)
        rank_terms = rank_terms * self.cross_weight / math.sqrt(self.rank)
        if not interaction:
            rank_terms = rank_terms * 0.0
        x_score = rank_terms.sum(-1)
        score = self.bias + e_score + n_score + x_score
        logits = torch.stack((-0.5 * score, 0.5 * score), -1)
        return logits, {
            "bias": self.bias.expand_as(score), "eeg": e_score,
            "fnirs": n_score, "interaction": x_score,
            "rank_terms": rank_terms, "score": score,
        }


class AdditiveJointHead(ReferenceInteractionHead):
    def __init__(self, e_dim=32, n_dim=32):
        super().__init__(e_dim, n_dim, rank=1)
        self.e_factor.requires_grad_(False)
        self.n_factor.requires_grad_(False)
        self.cross_weight.requires_grad_(False)

    def forward(self, e, n, interaction=True):
        return super().forward(e, n, interaction=False)


class ConcatJointHead(nn.Module):
    def __init__(self, e_dim=32, n_dim=32, hidden=16, dropout=0.1):
        super().__init__()
        self.register_buffer("e_ref", torch.zeros(e_dim))
        self.register_buffer("n_ref", torch.zeros(n_dim))
        self.register_buffer("e_scale", torch.ones(e_dim))
        self.register_buffer("n_scale", torch.ones(n_dim))
        self.net = nn.Sequential(nn.Linear(e_dim+n_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    @torch.no_grad()
    def fit_reference(self, e, n, weights=None):
        helper = ReferenceInteractionHead(e.shape[1], n.shape[1], 1).to(e.device)
        helper.fit_reference(e, n, weights)
        self.e_ref.copy_(helper.e_ref); self.n_ref.copy_(helper.n_ref)
        self.e_scale.copy_(helper.e_scale); self.n_scale.copy_(helper.n_scale)

    def forward(self, e, n, interaction=True):
        del interaction
        z = torch.cat(((e-self.e_ref)/self.e_scale, (n-self.n_ref)/self.n_scale), -1)
        logits = self.net(z)
        score = logits[:, 1] - logits[:, 0]
        return logits, {"score": score, "interaction": torch.zeros_like(score)}


class SharedExchangeHead(nn.Module):
    def __init__(self, e_dim=32, n_dim=32, shared=8):
        super().__init__()
        self.base = ReferenceInteractionHead(e_dim, n_dim, rank=1)
        self.base.e_factor.requires_grad_(False); self.base.n_factor.requires_grad_(False); self.base.cross_weight.requires_grad_(False)
        self.e_shared = nn.Linear(e_dim, shared, bias=False)
        self.n_shared = nn.Linear(n_dim, shared, bias=False)
        self.exchange = nn.Sequential(nn.Linear(3*shared, shared), nn.GELU(), nn.Linear(shared, 1))
        nn.init.zeros_(self.exchange[-1].weight); nn.init.zeros_(self.exchange[-1].bias)

    @torch.no_grad()
    def fit_reference(self, e, n, weights=None): self.base.fit_reference(e, n, weights)

    def forward(self, e, n, interaction=True):
        logits, parts = self.base(e, n, interaction=False)
        ee=(e-self.base.e_ref)/self.base.e_scale; nn_=(n-self.base.n_ref)/self.base.n_scale
        a,b=self.e_shared(ee),self.n_shared(nn_)
        x=self.exchange(torch.cat((a,b,a*b),-1)).squeeze(-1)
        if not interaction: x=x*0.0
        score=parts["score"]+x
        return torch.stack((-0.5*score,0.5*score),-1), {**parts,"interaction":x,"score":score}


class PrototypeJointHead(nn.Module):
    """Four fixed train-fold prototypes with one distance-based joint decision."""

    def __init__(self, e_dim=32, n_dim=32, projected=8, prototypes_per_class=2):
        super().__init__()
        count = 2 * int(prototypes_per_class)
        self.e_projection = nn.Linear(e_dim, projected, bias=False)
        self.n_projection = nn.Linear(n_dim, projected, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.register_buffer("prototype_e", torch.zeros(count, projected))
        self.register_buffer("prototype_n", torch.zeros(count, projected))
        self.register_buffer(
            "prototype_label",
            torch.arange(2).repeat_interleave(prototypes_per_class),
        )
        self.register_buffer("prototype_trial", torch.full((count,), -1, dtype=torch.long))
        self.register_buffer("prototypes_ready", torch.tensor(False))

    @torch.no_grad()
    def fit_reference(self, e, n, weights=None):
        del e, n, weights

    @torch.no_grad()
    def set_prototypes(self, e, n, labels, trial_ids):
        """Select deterministic, distinct whole-trial medoids from training only."""
        labels = labels.long().reshape(-1)
        trial_ids = trial_ids.long().reshape(-1)
        if not (len(e) == len(n) == len(labels) == len(trial_ids)):
            raise ValueError("prototype arrays must align")
        ep, np_ = self.e_projection(e), self.n_projection(n)
        selected = []
        per_class = int((self.prototype_label == 0).sum())
        for label in (0, 1):
            candidates = torch.unique(trial_ids[labels == label], sorted=True)
            if len(candidates) < per_class:
                raise ValueError("insufficient distinct train trials for prototypes")
            trial_vectors = []
            for trial in candidates:
                mask = trial_ids == trial
                trial_vectors.append(torch.cat((ep[mask].mean(0), np_[mask].mean(0))))
            vectors = torch.stack(trial_vectors)
            centre = vectors.mean(0)
            order = torch.argsort((vectors - centre).square().sum(-1), stable=True)
            selected.extend((int(candidates[index]), label) for index in order[:per_class])
        for slot, (trial, label) in enumerate(selected):
            mask = trial_ids == trial
            self.prototype_e[slot].copy_(ep[mask].mean(0))
            self.prototype_n[slot].copy_(np_[mask].mean(0))
            self.prototype_label[slot] = label
            self.prototype_trial[slot] = trial
        if len(set(int(x) for x in self.prototype_trial)) != len(self.prototype_trial):
            raise AssertionError("prototype trial ids must be unique")
        self.prototypes_ready.fill_(True)

    def forward(self, e, n, interaction=True):
        del interaction
        if not bool(self.prototypes_ready):
            raise RuntimeError("training-only prototypes were not frozen")
        ep, np_ = self.e_projection(e), self.n_projection(n)
        distance = (ep[:, None] - self.prototype_e[None]).square().mean(-1)
        distance = distance + (np_[:, None] - self.prototype_n[None]).square().mean(-1)
        temperature = self.log_temperature.exp().clamp(0.05, 20.0)
        logits = torch.stack(
            [torch.logsumexp(-distance[:, self.prototype_label == label] / temperature, 1)
             for label in (0, 1)],
            -1,
        )
        score = logits[:, 1] - logits[:, 0]
        return logits, {"score": score, "interaction": score, "prototype_distance": distance}


class BoundedConditionalRead(nn.Module):
    def __init__(self, dim=32, rank=4, max_shift=0.5):
        super().__init__()
        self.register_buffer("query_reference", torch.zeros(dim))
        self.query=nn.Linear(dim,rank,bias=False); self.key=nn.Linear(dim,rank,bias=False)
        nn.init.normal_(self.query.weight,std=0.02); nn.init.normal_(self.key.weight,std=0.02)
        self.raw_gain=nn.Parameter(torch.zeros(())); self.rank=int(rank); self.max_shift=float(max_shift)

    @torch.no_grad()
    def fit_reference(self, query_feature, weights=None):
        if weights is None: weights = query_feature.new_ones(len(query_feature))
        weights = weights.to(query_feature).reshape(-1)
        weights = weights / weights.sum()
        self.query_reference.copy_((weights[:, None] * query_feature).sum(0))

    def forward(self,e_query,memory,base_weights,valid):
        if memory.shape[:-1] != base_weights.shape or valid.shape != base_weights.shape:
            raise ValueError("conditional-read shape mismatch")
        if not valid.any(-1).all(): raise ValueError("empty region requires explicit fallback")
        prior=torch.where(valid,base_weights,torch.zeros_like(base_weights))
        if (prior[valid] <= 0).any(): raise ValueError("legacy weights must be positive")
        prior=prior/prior.sum(-1,keepdim=True)
        q=self.query(e_query-self.query_reference)[:,None,None,:]; k=self.key(memory)
        content=(q*k).sum(-1)/math.sqrt(self.rank)
        gain=self.max_shift*self.raw_gain.tanh()
        score=prior.clamp_min(1e-30).log()+gain*content.tanh()
        score=score.masked_fill(~valid,-torch.inf); weights=score.softmax(-1)
        return weights,{"gain":gain,"base_weights":prior,"content_score":content}


@dataclass
class V8Features:
    e_main: torch.Tensor
    n_main: torch.Tensor
    e_query: torch.Tensor
    teacher_logits: torch.Tensor
    b_m50_log_probability: torch.Tensor
    memory: torch.Tensor
    base_weights: torch.Tensor


class V8JointModel(nn.Module):
    """Unchanged B_M50 feature backbone plus one joint deployment head."""

    def __init__(self, config):
        super().__init__()
        base_config={**config,"family":"probability","mixer":config.get("backbone_mixer","mamba"),"alpha":0.5}
        self.backbone=make_mambasta_v5(base_config)
        role=config["config_id"]; dim=config.get("d_model",32)
        if role in {"J1_R4","J1_R4_CONV","J1_NoKD","J1_RU_KD","J2_025","J2_050","J1_Reg0","J1_Reg1e2"}: self.head=ReferenceInteractionHead(dim,dim,rank=4)
        elif role=="J1_R8": self.head=ReferenceInteractionHead(dim,dim,rank=8)
        elif role=="C_Add": self.head=AdditiveJointHead(dim,dim)
        elif role=="C_Concat": self.head=ConcatJointHead(dim,dim)
        elif role=="J3_P4": self.head=PrototypeJointHead(dim,dim,projected=8,prototypes_per_class=2)
        elif role=="J4_S8": self.head=SharedExchangeHead(dim,dim,shared=8)
        else: raise ValueError(f"unsupported implemented V8 role: {role}")
        self.conditional_read=None
        if role in {"J2_025","J2_050"}:
            self.conditional_read=BoundedConditionalRead(dim,rank=4,max_shift=0.25 if role=="J2_025" else 0.5)
        self.role=role

    def load_b_m50_state(self,state): self.backbone.load_state_dict(state,strict=True)

    def freeze_backbone(self):
        self.backbone.requires_grad_(False); self.backbone.eval()

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()): self.backbone.eval()
        return self

    def _eeg_main(self,eeg_feature): return self.backbone.teacher.classifier[:-1](eeg_feature)
    def _n_main(self,n_feature): return self.backbone.v5_head.n_head[:-1](n_feature)

    def extract_features(self,eeg,fnirs,include_teacher_probability=False):
        teacher_logits,teacher_feature=frozen_eeg_forward(self.backbone.teacher,eeg)
        # Match B_M50 exactly: its fNIRS branch uses a zero temporal query.
        self.backbone._teacher_feature=torch.zeros_like(teacher_feature)
        n_feature,path=self.backbone.encode_v5_regions(fnirs)
        memory=path["memory"]; base_weights=path["weights"]
        if self.conditional_read is not None:
            valid=torch.ones_like(base_weights,dtype=torch.bool)
            weights,_=self.conditional_read(teacher_feature,memory,base_weights,valid)
            values=self.backbone.readout.value(memory)
            region_values=torch.einsum("brt,brtd->brd",weights,values)+self.backbone.region_embedding[None]
            region_weights=self.backbone.region_score(region_values).squeeze(-1).softmax(-1)
            n_feature=torch.einsum("br,brd->bd",region_weights,region_values)
        logp=None
        if include_teacher_probability:
            logp,_=self.backbone.v5_head(teacher_logits,n_feature)
        return V8Features(
            e_main=self._eeg_main(teacher_feature), n_main=self._n_main(n_feature),
            e_query=teacher_feature, teacher_logits=teacher_logits,
            b_m50_log_probability=logp, memory=memory, base_weights=base_weights,
        )

    def forward(self,eeg,fnirs,interaction=True,return_diagnostics=False):
        features=self.extract_features(eeg,fnirs)
        logits,parts=self.head(features.e_main,features.n_main,interaction=interaction)
        diagnostics={**parts,"teacher_logits":features.teacher_logits,
                     "e_main":features.e_main,"n_main":features.n_main}
        return logits, diagnostics if return_diagnostics else {}


def joint_training_loss(logits,target,teacher_log_probability,interaction_score,kd_weight=0.5,temperature=2.0,interaction_penalty=1e-3):
    ce=nn.functional.cross_entropy(logits,target)
    softened=(teacher_log_probability.detach()/temperature).softmax(-1)
    kd=nn.functional.kl_div(nn.functional.log_softmax(logits/temperature,-1),softened,reduction="batchmean")*temperature**2
    reg=interaction_score.square().mean()
    return ce+kd_weight*kd+interaction_penalty*reg,{"joint_nll":ce.detach(),"kd":kd.detach(),"reg":reg.detach()}

