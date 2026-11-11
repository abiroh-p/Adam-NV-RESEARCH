"""Frozen per-tensor beta1 control with Adam-NV-compatible update and bias correction.

Uses AdamNV's (m,v,P) formulas, but a single constant pilot-calibrated beta1
for each named parameter tensor. Does not consult gradients to choose beta1.
"""
import math
import torch
from torch.optim import Optimizer


class FrozenTensorAdam(Optimizer):
    def __init__(self, named_parameters, beta_map, lr=1e-3, beta_2=0.999, eps=1e-8,
                 weight_decay=0.0):
        named_parameters = list(named_parameters)
        names = {n for n, _ in named_parameters}
        if set(beta_map) != names:
            raise ValueError(f"Pilot coefficient names differ from model: missing={sorted(names-set(beta_map))}; "
                             f"extra={sorted(set(beta_map)-names)}")
        groups = []
        self.names = {}
        for name, p in named_parameters:
            b = float(beta_map[name])
            if not math.isfinite(b) or not (0.0 <= b < 1.0):
                raise ValueError(f"Invalid pilot beta for {name}: {b}")
            groups.append({"params": [p], "beta1": b})
            self.names[id(p)] = name
        super().__init__(groups, dict(lr=lr, beta_2=beta_2, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b, b2, lr, eps, wd = (group[k] for k in ("beta1", "beta_2", "lr", "eps", "weight_decay"))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad if wd == 0 else p.grad.add(p, alpha=wd)
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["P"] = torch.ones_like(p)
                t = st["t"] + 1
                st["m"].mul_(b).add_(g, alpha=1-b)
                st["v"].mul_(b2).addcmul_(g, g, value=1-b2)
                st["P"].mul_(b).clamp_(min=1e-7)  # Adam-NV product floor
                m_hat = st["m"] / torch.clamp(1.0-st["P"], min=eps)
                v_hat = st["v"] / (1.0-b2**t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                st["t"] = t
        return loss


def beta_from_nv_state(st, group):
    """Beta used in Adam-NV's most recent step (rho is already updated)."""
    rho = st["rho"]
    if group["response_mode"] == "distrust":
        scale = (1.0-group["gamma"]*rho).clamp(min=0.0)
        return group["beta_min"]+(group["beta_base"]-group["beta_min"])*scale
    scale = (group["gamma"]*rho).clamp(min=0.0, max=1.0)
    return group["beta_base"]+(group["beta_max"]-group["beta_base"])*scale


class PilotCollector:
    """Sample per-parameter mean beta after an optimizer.step(), at fixed step intervals."""
    def __init__(self, model, optimizer, sample_every=100):
        self.model, self.optimizer = model, optimizer
        self.sample_every = sample_every
        self.n_steps = 0
        self.samples = 0
        self.sums = {name: 0.0 for name, _ in model.named_parameters()}
        self.groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}

    @torch.no_grad()
    def record_step(self):
        self.n_steps += 1
        if self.n_steps % self.sample_every:
            return
        for name, p in self.model.named_parameters():
            if "rho" not in self.optimizer.state[p]:
                raise RuntimeError(f"Missing Adam-NV rho state for {name}")
            self.sums[name] += beta_from_nv_state(self.optimizer.state[p], self.groups[id(p)]).mean().item()
        self.samples += 1

    def result(self):
        if not self.samples:
            raise RuntimeError("No pilot beta samples; lower --sample-every")
        return {name: val/self.samples for name, val in self.sums.items()}
