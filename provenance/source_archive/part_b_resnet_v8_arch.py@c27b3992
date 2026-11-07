"""Run part_b_resnet_v8.main() with the ResNet-18 replaced by another architecture (Exp. 17).

part_b_resnet_v8.py and adam_nv.py are NOT modified. This wrapper
  1. removes its own option from the command line,
  2. replaces build_model inside part_b_resnet_v8 by models_alt.build(arch, num_classes),
  3. writes arch_config_seed<k>.json next to the run's logs, and
  4. calls part_b_resnet_v8.main() with the remaining arguments unchanged.
File names are those of a normal run (they do not contain the architecture), so give every
architecture its own --log-dir.

Own option:
  --arch {vit_tiny,vgg11_bn}   required

Example:
    python3 part_b_resnet_v8_arch.py --arch vit_tiny --optimizer adam_nv --lr 5e-4 --schedule constant \
        --sigma 0 --seed 0 --epochs 60 --batch-size 128 --data-dir ../data --log-dir ../logs_exp17_vit --deterministic
"""

import argparse
import json
import os
import sys
import time


def main():
    own_parser = argparse.ArgumentParser(add_help=False)
    own_parser.add_argument("--arch", choices=["vit_tiny", "vgg11_bn"], required=True)
    own, rest = own_parser.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--seed", type=int, required=True)
    peek.add_argument("--log-dir", default=None)
    pk, _ = peek.parse_known_args(rest)
    if pk.log_dir is None:
        sys.exit("--log-dir is required")
    os.makedirs(pk.log_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import models_alt
    import part_b_resnet_v8 as v8

    def build_model(num_classes: int = 10):
        return models_alt.build(own.arch, num_classes)

    v8.build_model = build_model
    with open(os.path.join(pk.log_dir, f"arch_config_seed{pk.seed}.json"), "w") as fh:
        json.dump({"arch": own.arch, "seed": pk.seed, "argv": rest,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    sys.argv = [sys.argv[0]] + rest
    v8.main()


if __name__ == "__main__":
    main()
