"""Adam-NV whose controller acts on a subset of the parameter tensors only (Exp. 14).

AdamNVSubset subclasses AdamNV and splits the parameters into two parameter groups by tensor
rank. In the "adaptive" group the controller runs as usual (gamma as given, default 1); in the
other group gamma = 0, so beta_1 stays at beta_base = 0.9 and those tensors follow plain Adam
(exactly, up to the clamp on the running product, see adam_nv.py).

  subset="onedim"  controller on tensors with at most one dimension (normalization parameters and
                   biases; 41 of the 62 tensors of our ResNet-18, 9,610 elements in total);
                   all other tensors follow plain Adam.
  subset="rest"    controller on tensors with two or more dimensions (convolution and classifier
                   weights; 21 tensors, 99.9% of the elements); one-dimensional tensors follow
                   plain Adam.

adam_nv.py is NOT modified. get_beta1_stats() still averages over all tensors, including the
plain-Adam ones (beta_1 = 0.9), so the logged beta1 columns of these runs are diluted.
"""

import torch

from adam_nv import AdamNV


class AdamNVSubset(AdamNV):
    def __init__(self, params, *args, subset="onedim", **kwargs):
        if subset not in ("onedim", "rest"):
            raise ValueError(f"Invalid subset: {subset!r} (use 'onedim' or 'rest')")
        params = list(params)
        if any(isinstance(p, dict) for p in params):
            raise ValueError("AdamNVSubset expects a flat list of tensors, not parameter groups")
        adaptive_fn = (lambda p: p.dim() <= 1) if subset == "onedim" else (lambda p: p.dim() >= 2)
        gamma = kwargs.pop("gamma", 1.0)
        adaptive = [p for p in params if adaptive_fn(p)]
        frozen = [p for p in params if not adaptive_fn(p)]
        groups = []
        if adaptive:
            groups.append({"params": adaptive, "gamma": gamma})
        if frozen:
            groups.append({"params": frozen, "gamma": 0.0})
        super().__init__(groups, *args, gamma=gamma, **kwargs)
        self._subset = subset
        self.n_adaptive_tensors = len(adaptive)
        self.n_plain_tensors = len(frozen)
