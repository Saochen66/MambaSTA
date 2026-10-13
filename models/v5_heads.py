"""V5 fusion heads and optimizer grouping."""
import torch
from torch import nn
import torch.nn.functional as F


class ConservativeResidualHead(nn.Module):
    def __init__(self, dim=32, bound=0.5, penalty=0.01):
        super().__init__()
        if bound <= 0 or penalty < 0:
            raise ValueError("invalid residual constraint")
        self.bound, self.penalty = float(bound), float(penalty)
        self.features = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(0.2))
        self.out = nn.Linear(dim, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, teacher_logits, eeg_feature, n_feature, available=None):
        raw = self.out(self.features(torch.cat((eeg_feature.detach(), n_feature), -1)))
        delta = self.bound * torch.tanh(raw)
        if available is not None:
            mask = available.to(delta).reshape(-1, 1)
            if mask.shape != delta.shape or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("available must be binary [B]")
            delta = delta * mask
        logits = teacher_logits.detach() + torch.cat((-delta / 2, delta / 2), -1)
        return logits, {"delta": delta, "regularizer": self.penalty * delta.square().mean()}


class ProbabilityFusionHead(nn.Module):
    def __init__(self, dim=32, alpha=0.25):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = float(alpha)
        self.n_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2))

    def forward(self, teacher_logits, n_feature, available=None):
        n_logits = self.n_head(n_feature)
        log_e, log_n = F.log_softmax(teacher_logits.detach(), -1), F.log_softmax(n_logits, -1)
        a = log_e.new_tensor(self.alpha)
        log_p = torch.logaddexp(log_e + torch.log1p(-a), log_n + torch.log(a))
        if available is not None:
            mask = available.to(device=log_p.device, dtype=torch.bool).reshape(-1, 1)
            if mask.shape != (len(log_p), 1):
                raise ValueError("available must have B entries")
            log_p = torch.where(mask, log_p, log_e)
        return log_p, {"n_logits": n_logits}


def v5_loss(output, target, info, kind, trial_ids=None):
    if kind == "probability":
        log_p = output
        loss = F.nll_loss(log_p, target) + 0.2 * F.cross_entropy(info["n_logits"], target)
    elif kind == "residual":
        log_p = F.log_softmax(output, -1)
        loss = F.nll_loss(log_p, target) + info["regularizer"]
    else:
        raise ValueError(kind)
    if trial_ids is not None:
        terms = []
        for trial in torch.unique(trial_ids):
            mask = trial_ids == trial; y = target[mask]
            if len(y) != 10 or not torch.all(y == y[0]):
                raise ValueError("expected one complete 10-window single-label trial")
            log_mean = torch.logsumexp(log_p[mask], dim=0) - log_p.new_tensor(10.0).log()
            terms.append(-log_mean[y[0]])
        loss = loss + 0.2 * torch.stack(terms).mean()
    return loss


def optimizer_groups(model, weight_decay=1e-4):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if getattr(parameter, "_no_weight_decay", False) or parameter.ndim < 2 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
