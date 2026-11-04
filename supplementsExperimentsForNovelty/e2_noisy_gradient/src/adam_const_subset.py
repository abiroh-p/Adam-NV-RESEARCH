"""Adam with a CONSTANT beta_1 on a subset of the parameter tensors (Exp. 20).

AdamConstSubset subclasses AdamNV and splits the parameters into two groups by tensor rank. In
the selected group gamma = 0 and beta_base = beta_min = const_beta1, so beta_1 is the constant
const_beta1 at every step (and the update is Adam's with beta_1 = const_beta1). In the other group
gamma = 0 and beta_base = beta_min = 0.9 (plain Adam). The controller is not active anywhere.

  subset="onedim"  selected = tensors with at most one dimension (normalization scales and
                   shifts and biases; 41 tensors, 9,610 elements in our ResNet-18)
  subset="rest"    selected = tensors with two or more dimensions

Purpose: Exp. 14 showed that Adam-NV's controller on the one-dimensional tensors alone raises
memorization; this control asks whether a constant, lower beta_1 on those tensors does the same.
adam_nv.py is NOT modified. get_beta1_stats() averages over all tensors, so the logged beta1
columns mix both groups (equal weight per tensor) and are diluted like those of Exp. 14.
"""

from adam_nv import AdamNV


class AdamConstSubset(AdamNV):
    def __init__(self, params, *args, subset="onedim", const_beta1=None, **kwargs):
        if subset not in ("onedim", "rest"):
            raise ValueError(f"Invalid subset: {subset!r} (use 'onedim' or 'rest')")
        if const_beta1 is None or not 0.0 < float(const_beta1) < 1.0:
            raise ValueError(f"const_beta1 must be in (0, 1), got {const_beta1!r}")
        params = list(params)
        if any(isinstance(p, dict) for p in params):
            raise ValueError("AdamConstSubset expects a flat list of tensors, not parameter groups")
        select = (lambda p: p.dim() <= 1) if subset == "onedim" else (lambda p: p.dim() >= 2)
        selected = [p for p in params if select(p)]
        others = [p for p in params if not select(p)]
        plain = float(kwargs.get("beta_base", 0.9))
        c = float(const_beta1)
        groups = []
        if selected:
            groups.append({"params": selected, "gamma": 0.0, "beta_base": c, "beta_min": c})
        if others:
            groups.append({"params": others, "gamma": 0.0, "beta_base": plain, "beta_min": plain})
        super().__init__(groups, *args, gamma=0.0, **kwargs)
        self._subset = subset
        self._const_beta1 = c
        self.n_selected_tensors = len(selected)
        self.n_plain_tensors = len(others)
