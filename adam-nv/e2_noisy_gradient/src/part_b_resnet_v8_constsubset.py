"""Run part_b_resnet_v8.main() with AdamNV replaced by AdamConstSubset (Exp. 20).

part_b_resnet_v8.py and adam_nv.py are NOT modified. This wrapper removes its own options, replaces
the AdamNV name inside part_b_resnet_v8 by AdamConstSubset (constant beta_1 on the selected tensors,
plain Adam elsewhere), writes constsubset_config_seed<k>.json next to the logs and calls
part_b_resnet_v8.main() with the remaining arguments unchanged.

Use it ONLY with --optimizer adam_nv (file names are those of a normal Adam-NV run, so give every
variant its own --log-dir).

Own options:
  --subset {onedim,rest}   tensors that get the constant beta_1 (onedim = rank <= 1)
  --const-beta1 C          the constant beta_1, in (0, 1)

Example:
    python3 part_b_resnet_v8_constsubset.py --subset onedim --const-beta1 0.79 --optimizer adam_nv \
        --schedule constant --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1 --seed 0 \
        --epochs 60 --batch-size 128 --data-dir ../data --log-dir ../logs_exp20_onedim_c079 --deterministic
"""

import argparse
import json
import os
import sys
import time


def _split_args():
    own_parser = argparse.ArgumentParser(add_help=False)
    own_parser.add_argument("--subset", choices=["onedim", "rest"], required=True)
    own_parser.add_argument("--const-beta1", type=float, required=True)
    own, rest = own_parser.parse_known_args()
    if not 0.0 < own.const_beta1 < 1.0:
        sys.exit("--const-beta1 must be in (0, 1)")
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--seed", type=int, required=True)
    peek.add_argument("--log-dir", default=None)
    peek.add_argument("--optimizer", default=None)
    pk, _ = peek.parse_known_args(rest)
    if pk.optimizer != "adam_nv":
        sys.exit("part_b_resnet_v8_constsubset.py must be used with --optimizer adam_nv")
    if pk.log_dir is None:
        sys.exit("--log-dir is required")
    return own, rest, pk


def main():
    own, rest, pk = _split_args()
    os.makedirs(pk.log_dir, exist_ok=True)
    import part_b_resnet_v8 as v8
    from adam_const_subset import AdamConstSubset

    class _Patched(AdamConstSubset):
        def __init__(self, params, **kwargs):
            super().__init__(params, subset=own.subset, const_beta1=own.const_beta1, **kwargs)

    v8.AdamNV = _Patched
    config = {"subset": own.subset, "const_beta1": own.const_beta1, "seed": pk.seed, "argv": rest,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(pk.log_dir, f"constsubset_config_seed{pk.seed}.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    sys.argv = [sys.argv[0]] + rest
    v8.main()


if __name__ == "__main__":
    main()
