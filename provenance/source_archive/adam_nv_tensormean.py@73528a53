"""Adam-NV with a tensor-mean beta_1 control (Exp. 13).

AdamNVTensorMean subclasses AdamNVShuffle and overrides only the hook that Exp. 11 used for the
permutation. After the controller has computed its per-element beta_1 tensor for a parameter,

  mode="tensor_mean"  every element of that parameter receives the tensor's current MEAN beta_1
                      (tensor-level adaptation is kept, all within-tensor spread is removed);
  mode="none"         beta_1 is left untouched (bit-identical to AdamNV); used as a regression
                      check and as the source of per-tensor beta_1 statistics.

The running product P_t uses the beta_1 values actually applied, so the bias correction stays
exact. adam_nv.py and adam_nv_shuffle.py are NOT modified.

Optionally (tensor_log_path) one JSON-lines record per logging step lists, for every parameter
tensor in optimizer order, its shape and the mean, minimum and maximum of the controller's
own per-element beta_1. Together with the element-level distribution log of AdamNVShuffle this
shows which tensors carry the low beta_1 values.
"""

import json

import torch

from adam_nv_shuffle import AdamNVShuffle


class AdamNVTensorMean(AdamNVShuffle):
    def __init__(self, params, *args, mode="tensor_mean", tensor_log_path=None, **kwargs):
        if mode not in ("tensor_mean", "none"):
            raise ValueError(f"Invalid mode: {mode!r} (use 'tensor_mean' or 'none')")
        kwargs["shuffle"] = "none"
        super().__init__(params, *args, **kwargs)
        self._mode = mode
        self._tensor_log_path = tensor_log_path
        self._param_shapes = [list(p.shape) for g in self.param_groups for p in g["params"]]

    def _apply_shuffle(self, p, state, beta_1):
        if self._mode == "none" or beta_1.dim() == 0 or beta_1.numel() < 2:
            return beta_1
        return beta_1.mean().expand_as(beta_1).contiguous()

    def _write_distribution(self, parts):
        super()._write_distribution(parts)
        if self._tensor_log_path is None:
            return
        shapes = self._param_shapes if len(parts) == len(self._param_shapes) else [None] * len(parts)
        tensors = [{"shape": shapes[i], "numel": int(t.numel()), "mean": float(t.mean()),
                    "min": float(t.min()), "max": float(t.max())} for i, t in enumerate(parts)]
        with open(self._tensor_log_path, "a") as fh:
            fh.write(json.dumps({"step": self._step_count + 1, "tensors": tensors}) + "\n")
