"""Pilot-frozen per-tensor beta control + memorization trajectories.

Reuses the repository's part_b_resnet_v8.py (model, CIFAR pipeline, noisy labels,
metrics) and AdamNV implementation. No replacement of existing source files.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_tensor_adam import FrozenTensorAdam, PilotCollector


def make_loaders(args, gen):
    from part_b_resnet_v8 import get_dataloaders
    return get_dataloaders(args.batch_size, args.data_dir, smoke_test=args.smoke_test,
                           synthetic=args.synthetic_data, loader_generator=gen,
                           label_noise=args.label_noise, label_noise_seed=args.seed + 2000,
                           dataset="cifar10", return_label_diagnostics=not args.synthetic_data)


def atomic_json(path, obj):
    target = str(path) + ".tmp"
    with open(target, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(target, path)


def pilot_path(args):
    return Path(args.log_dir) / f"pilot_ln{args.label_noise}_seed{args.seed}.json"


def train(args):
    from part_b_resnet_v8 import build_model, build_optimizer, evaluate, evaluate_label_memorization
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(args.seed)
    loaders = make_loaders(args, gen)
    if args.synthetic_data:
        train_loader, test_loader = loaders
        diag_loader = None
    else:
        train_loader, test_loader, diag_loader = loaders
    model = build_model(num_classes=10).to(device)
    beta_hash = None
    if args.mode == "frozen":
        with open(args.pilot_map) as f:
            artifact = json.load(f)
        if artifact["label_noise"] != args.label_noise or artifact["epochs"] != args.epochs:
            raise ValueError("Pilot noise/epochs must match evaluation run")
        if artifact["pilot_seed"] in args.eval_seeds:
            raise ValueError("Pilot seed cannot overlap evaluation seeds")
        with open(args.pilot_map, "rb") as f:
            beta_hash = hashlib.sha256(f.read()).hexdigest()
        optimizer = FrozenTensorAdam(model.named_parameters(), artifact["beta_map"], lr=args.lr)
    elif args.mode == "adam":
        optimizer = build_optimizer("adam", model.parameters(), args.lr)
    else:
        optimizer = build_optimizer("adam_nv", model.parameters(), args.lr)
    collector = PilotCollector(model, optimizer, args.sample_every) if args.mode == "pilot" else None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_ln{args.label_noise}_seed{args.seed}"
    outcsv = Path(args.log_dir) / f"trajectory_{tag}.csv"
    meta = {"mode": args.mode, "seed": args.seed, "label_noise": args.label_noise,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "sample_every": args.sample_every, "checkpoints": args.checkpoints,
            "torch": torch.__version__, "device": str(device), "pilot_map": args.pilot_map,
            "pilot_map_sha256": beta_hash, "synthetic_data": args.synthetic_data,
            "script": Path(__file__).name, "started": time.strftime('%Y-%m-%dT%H:%M:%S')}
    atomic_json(Path(args.log_dir) / f"config_{tag}.json", meta)
    checkpoints = set(args.checkpoints)
    fields = ["epoch", "train_loss", "val_loss", "val_acc", "train_clean_subset_acc",
              "train_corrupt_clean_label_acc", "train_corrupt_noisy_label_acc", "lr", "duration_sec"]
    with open(outcsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for epoch in range(1, args.epochs+1):
            start = time.monotonic()
            model.train()
            n, running = 0, 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                if collector is not None:
                    collector.record_step()
                n += len(x)
                running += loss.item()*len(x)
            sched.step()
            val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)
            d = {}
            if diag_loader is not None and epoch in checkpoints:
                d = evaluate_label_memorization(model, diag_loader, device)
            writer.writerow({"epoch": epoch, "train_loss": running/n,
                             "val_loss": val_loss, "val_acc": val_acc,
                             "lr": sched.get_last_lr()[0], "duration_sec": time.monotonic()-start,
                             **d})
            f.flush()
            print(f"[{tag}] epoch {epoch}/{args.epochs} train={running/n:.4f} val={val_acc:.4f} "
                  f"memorization={d.get('train_corrupt_noisy_label_acc', 'NA')}", flush=True)
    if collector is not None:
        beta_map = collector.result()
        atomic_json(pilot_path(args), {"pilot_seed": args.seed, "label_noise": args.label_noise,
                   "epochs": args.epochs, "sample_every": args.sample_every,
                   "samples": collector.samples, "steps": collector.n_steps, "beta_map": beta_map,
                   "beta_min_observed": min(beta_map.values()),
                   "beta_max_observed": max(beta_map.values()),
                   "note": "Step-sampled per-tensor means of actual Adam-NV beta1 over full pilot training; "
                           "frozen throughout the separate independent evaluation runs."})
        print(f"[pilot] wrote {pilot_path(args)}, {collector.samples} samples", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["pilot", "frozen", "adam_nv", "adam"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--label-noise", type=float, default=.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1,5,10,20,30,45,60])
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--log-dir", default="../logs_exp22_23")
    p.add_argument("--pilot-map", default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--synthetic-data", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    a = p.parse_args()
    if a.epochs < 1 or a.sample_every < 1:
        p.error("epochs and sample-every must be >= 1")
    if a.mode == "frozen" and not a.pilot_map:
        p.error("--pilot-map is required for frozen control")
    if a.mode == "pilot" and a.seed in a.eval_seeds:
        p.error("pilot seed must not be in --eval-seeds")
    if any(e < 1 or e > a.epochs for e in a.checkpoints):
        p.error("checkpoints must be within training epochs")
    return a

if __name__ == "__main__":
    train(parse_args())
