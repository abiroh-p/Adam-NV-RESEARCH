"""Run part_b_resnet_v8.main() with AdamNV replaced by AdamNVSubset (Exp. 14).

part_b_resnet_v8.py and adam_nv.py are NOT modified (their recorded hashes stay valid). This
wrapper

  1. removes its own options from the command line,
  2. replaces the AdamNV name inside part_b_resnet_v8 by AdamNVSubset, whose controller acts only
     on the requested subset of tensors (all other tensors follow plain Adam),
  3. writes subset_config_seed<k>.json next to the run's logs, and
  4. calls part_b_resnet_v8.main() with the remaining arguments unchanged.

Use it ONLY with --optimizer adam_nv (the file names are those of a normal Adam-NV run, so give
every variant its own --log-dir).

Own options:
  --subset {onedim,rest}             required: tensors on which the controller is active
                                     (onedim = rank <= 1, rest = rank >= 2)

Example:
    python3 part_b_resnet_v8_subset.py --subset onedim --optimizer adam_nv --schedule constant \\
        --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1 --seed 0 --epochs 60 \\
        --batch-size 128 --data-dir ../data --log-dir ../logs_exp14_onedim --deterministic
"""

import argparse
import json
import os
import sys
import time


def _split_args():
    own_parser = argparse.ArgumentParser(add_help=False)
    own_parser.add_argument("--subset", choices=["onedim", "rest"], required=True)
    own, rest = own_parser.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--seed", type=int, required=True)
    peek.add_argument("--log-dir", default=None)
    peek.add_argument("--optimizer", default=None)
    pk, _ = peek.parse_known_args(rest)
    if pk.optimizer != "adam_nv":
        sys.exit("part_b_resnet_v8_subset.py must be used with --optimizer adam_nv")
    if pk.log_dir is None:
        sys.exit("--log-dir is required")
    return own, rest, pk


def main():
    own, rest, pk = _split_args()
    os.makedirs(pk.log_dir, exist_ok=True)
    import part_b_resnet_v8 as v8
    from adam_nv_subset import AdamNVSubset

    class _Patched(AdamNVSubset):
        def __init__(self, params, **kwargs):
            super().__init__(params, subset=own.subset, **kwargs)

    v8.AdamNV = _Patched

    config = {"subset": own.subset, "seed": pk.seed, "argv": rest,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(pk.log_dir, f"subset_config_seed{pk.seed}.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    sys.argv = [sys.argv[0]] + rest
    v8.main()


if __name__ == "__main__":
    main()
