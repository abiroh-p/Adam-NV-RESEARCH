"""Noise-proxy diagnostic v2 (Exp. 21): does Adam-NV's score track gradient noise WITHIN tensors?

Extends Exp. 16 (noise_proxy_diagnostic.py, unchanged) to fix its documented weaknesses. The training loop is the
same as in Exp. 16 and in the label-noise runs (Adam-NV, default settings, cosine schedule, 60 epochs), so the
trajectory reproduces the earlier Adam-NV runs; diagnostics only read the model and optimizer state.

At each checkpoint, at the CURRENT weights (BatchNorm in training mode, running statistics restored, no optimizer step),
K minibatch gradients are accumulated in three groups: first K/4, second K/4, last K/2 batches. Per element:
  s     Adam-NV's score from the optimizer moments m, v (lambda = 0.1, eps = 1e-8, as in the optimizer)
  s0    the same score with lambda = 0 and a tiny eps (scale-free in its units; diagnostic only, NOT used by the optimizer)
  q16   direct noise fraction from the first K/4 batches, clamped to [0, 1]   (comparable to Exp. 16 with --legacy-batches)
  qu    direct noise fraction from all K batches, UNCLAMPED (var / mean_sq): ranking is not tied at the clamp
  q     the same, clamped (as in Exp. 16)
  qA, qB   unclamped estimates from the first K/2 and the last K/2 batches: their rank correlation within a tensor is the
           split-half reliability of the direct estimate (Spearman-Brown corrected to K batches): the empirical ceiling
  rms   sqrt(mean_sq): the absolute gradient scale
  qstream  unclamped noise fraction over the last T training steps of the checkpoint epoch (weights drifting), exploratory
Per tensor (all tensors, up to 20,000 sampled elements each): means, within-tensor Spearman correlations (s,qu), (s0,qu),
(s,rms), (s0,rms), (qu,rms), the partial correlation of (s,qu) given rms, the split-half reliability, the clamp share, and
(s,qstream). Pooled over a random sample of 500,000 elements: the Exp. 16 statistics plus the new ones. Between tensors:
rank correlations of tensor means are computed by the analysis. A pooled sample of 100,000 elements is saved as .npz.

Example:
    python3 noise_proxy_diagnostic_v2.py --seed 0 --label-noise 0.2 --epochs 60 --data-dir ../data --log-dir ../logs_exp21_ln02_s0
CPU smoke test:
    python3 noise_proxy_diagnostic_v2.py --seed 0 --epochs 2 --checkpoints 1 2 --k-batches 8 --stream-steps 2 \
        --smoke-test --synthetic-data --data-dir /tmp/d --log-dir /tmp/diag2_smoke
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_proxy_diagnostic import direct_noise_fraction, proxy_score, spearman  # Exp. 16 functions, reused unchanged


def unclamped_noise_fraction(sum_g, sum_g2, k):
    """var / mean_sq with the unbiased variance, NOT clamped (can exceed 1 for noise-only elements)."""
    mean_sq = sum_g2 / k
    mu = sum_g / k
    var = torch.clamp((sum_g2 - k * mu * mu) / (k - 1), min=0.0)
    return var / (mean_sq + 1e-12), mean_sq


def _np(x):
    return x.double().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def _sp(a, b):  # Spearman on numpy arrays or tensors
    a, b = _np(a), _np(b)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def partial_spearman(a, b, c):
    """Partial rank correlation of a and b given c, from the three pairwise Spearman correlations (tensors or arrays)."""
    r_ab, r_ac, r_bc = _sp(a, b), _sp(a, c), _sp(b, c)
    den = (1 - r_ac ** 2) * (1 - r_bc ** 2)
    return float((r_ab - r_ac * r_bc) / np.sqrt(den)) if den > 0 else float("nan")


def spearman_brown(r_half):
    """Reliability of the full-length measure from the correlation of its two halves."""
    return float(2 * r_half / (1 + r_half)) if r_half == r_half else float("nan")


class Accumulators:
    """Sums of g and g^2 over three groups of minibatches (first K/4, second K/4, last K/2) for every parameter."""

    def __init__(self, params, k):
        self.k, self.k1, self.k2 = k, k // 4, k // 2
        self.sg = [[torch.zeros_like(p) for p in params] for _ in range(3)]
        self.sg2 = [[torch.zeros_like(p) for p in params] for _ in range(3)]
        self.n = [0, 0, 0]

    def add(self, b, params):
        grp = 0 if b < self.k1 else (1 if b < self.k2 else 2)
        for i, p in enumerate(params):
            self.sg[grp][i] += p.grad
            self.sg2[grp][i] += p.grad * p.grad
        self.n[grp] += 1

    def combined(self, i, groups):
        g = sum(self.sg[j][i] for j in groups); g2 = sum(self.sg2[j][i] for j in groups); return g, g2, sum(self.n[j] for j in groups)


class StreamAccumulator:
    """Read-only accumulation of the gradients that the optimizer actually receives during the last T steps of an epoch."""

    def __init__(self, params):
        self.sg = [torch.zeros_like(p) for p in params]; self.sg2 = [torch.zeros_like(p) for p in params]; self.n = 0

    def add(self, params):
        with torch.no_grad():
            for i, p in enumerate(params):
                self.sg[i] += p.grad; self.sg2[i] += p.grad * p.grad
        self.n += 1


def collect_diagnostics_v2(model, optimizer, dataset, args, device, epoch, stream=None):
    seed_gen = (777 + epoch) if args.legacy_batches else (args.seed * 100003 + epoch)
    gen = torch.Generator().manual_seed(seed_gen)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    params = [p for g in optimizer.param_groups for p in g["params"]]
    acc = Accumulators(params, args.k_batches)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    done = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        acc.add(done, params)
        done += 1
        if done == args.k_batches:
            break
    model.load_state_dict(saved)
    model.zero_grad(set_to_none=True)
    if done < args.k_batches or args.k_batches < 8:
        raise RuntimeError("need K >= 8 minibatches (K/4 >= 2 for the first group)")
    K = done
    group_of = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    cols = {k: [] for k in ("s", "s0", "q16", "qu", "q", "qA", "qB", "rms", "qs", "tid", "rank")}
    tensors = []
    sel_t = []
    for i, p in enumerate(params):
        st = optimizer.state[p]
        if "m" not in st:
            continue
        g = group_of[id(p)]
        g1, g1sq, n1 = acc.combined(i, [0]); gA, gAsq, nA = acc.combined(i, [0, 1]); gB, gBsq, nB = acc.combined(i, [2]); gT, gTsq, nT = acc.combined(i, [0, 1, 2])
        q16, _ = direct_noise_fraction(g1, g1sq, n1)
        qu, ms = unclamped_noise_fraction(gT, gTsq, nT)
        q, _ = direct_noise_fraction(gT, gTsq, nT)
        qA, _ = unclamped_noise_fraction(gA, gAsq, nA); qB, _ = unclamped_noise_fraction(gB, gBsq, nB)
        s = proxy_score(st["m"], st["v"], g["lam"], g["eps"]); s0 = proxy_score(st["m"], st["v"], 0.0, 1e-30)
        rms = torch.sqrt(ms)
        qs = unclamped_noise_fraction(stream.sg[i], stream.sg2[i], stream.n)[0] if (stream is not None and stream.n >= 2) else torch.full_like(s, float("nan"))
        flat = {"s": s, "s0": s0, "q16": q16, "qu": qu, "q": q, "qA": qA, "qB": qB, "rms": rms, "qs": qs}
        for k_, v_ in flat.items():
            cols[k_].append(v_.reshape(-1).float().cpu())
        cols["tid"].append(torch.full((p.numel(),), len(tensors), dtype=torch.int16)); cols["rank"].append(torch.full((p.numel(),), p.dim(), dtype=torch.int8))
        idx = torch.randperm(p.numel(), generator=torch.Generator().manual_seed(i))[:20000]
        a = {k_: v_.reshape(-1).float().cpu()[idx].numpy().astype(np.float64) for k_, v_ in flat.items()}
        r_half = _sp(a["qA"], a["qB"])
        rec = {"tensor": len(tensors), "rank": p.dim(), "numel": int(p.numel()), "n_used": int(idx.numel()),
               "mean_s": float(s.mean()), "mean_s0": float(s0.mean()), "mean_q": float(q.mean()), "mean_qu": float(qu.mean()), "mean_rms": float(rms.mean()),
               "median_rms": float(rms.median()), "frac_q_eq1": float((q >= 1 - 1e-6).float().mean()),
               "r_s_q": _sp(a["s"], a["q"]), "r_s_qu": _sp(a["s"], a["qu"]), "r_s0_qu": _sp(a["s0"], a["qu"]), "r_s_rms": _sp(a["s"], a["rms"]),
               "r_s0_rms": _sp(a["s0"], a["rms"]), "r_qu_rms": _sp(a["qu"], a["rms"]), "pr_s_qu_given_rms": partial_spearman(a["s"], a["qu"], a["rms"]),
               "pr_s0_qu_given_rms": partial_spearman(a["s0"], a["qu"], a["rms"]), "r_half": r_half, "reliability": spearman_brown(r_half),
               "r_s_qstream": _sp(a["s"], a["qs"]), "r_s_q16": _sp(a["s"], a["q16"])}
        tensors.append(rec)
    c = {k_: torch.cat(v_) for k_, v_ in cols.items()}
    n_el = int(c["s"].numel())
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    sel = torch.from_numpy(rng.choice(n_el, size=min(args.sample, n_el), replace=False))
    S = {k_: c[k_][sel] for k_ in ("s", "s0", "q16", "qu", "q", "rms")}
    out = {"epoch": epoch, "k_batches": K, "k_legacy": acc.n[0], "n_elements": n_el, "n_sampled": int(sel.numel()), "legacy_batches": bool(args.legacy_batches),
           "mean_s": float(c["s"].mean()), "mean_s0": float(c["s0"].mean()), "mean_q16": float(c["q16"].mean()), "mean_q": float(c["q"].mean()),
           "mean_qu": float(c["qu"].mean()), "mean_rms": float(c["rms"].mean()), "frac_s_zero": float((c["s"] == 0).float().mean()),
           "frac_q16_eq1": float((c["q16"] >= 1 - 1e-6).float().mean()), "frac_q_eq1": float((c["q"] >= 1 - 1e-6).float().mean()),
           "spearman_s_q16": spearman(S["s"], S["q16"]), "spearman_s_q": spearman(S["s"], S["q"]), "spearman_s_qu": spearman(S["s"], S["qu"]),
           "spearman_s0_qu": spearman(S["s0"], S["qu"]), "spearman_s_rms": spearman(S["s"], S["rms"]), "spearman_s0_rms": spearman(S["s0"], S["rms"]),
           "spearman_qu_rms": spearman(S["qu"], S["rms"]), "partial_s_qu_given_rms": partial_spearman(S["s"], S["qu"], S["rms"]),
           "stream_steps": int(stream.n) if stream is not None else 0, "tensors": tensors}
    sub = sel[: min(100000, sel.numel())]
    np.savez_compressed(os.path.join(args.log_dir, f"proxy2_sample_ln{args.label_noise}_seed{args.seed}_ep{epoch}.npz"),
                        **{k_: c[k_][sub].numpy().astype(np.float32) for k_ in ("s", "s0", "q16", "qu", "rms", "qs")}, tid=c["tid"][sub].numpy(), rank=c["rank"][sub].numpy())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 10, 20, 30, 45, 60])
    ap.add_argument("--k-batches", type=int, default=64)
    ap.add_argument("--stream-steps", type=int, default=64)
    ap.add_argument("--sample", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--legacy-batches", action="store_true", help="use the Exp. 16 batch generator seed (777 + epoch); default is per-run")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--synthetic-data", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    from part_b_resnet_v8 import build_model, build_optimizer, evaluate, get_dataloaders

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader, test_loader = get_dataloaders(
        args.batch_size, args.data_dir, smoke_test=args.smoke_test, synthetic=args.synthetic_data,
        loader_generator=loader_gen, label_noise=args.label_noise, label_noise_seed=args.seed + 2000,
        dataset="cifar10", return_label_diagnostics=False)
    model = build_model(num_classes=10).to(device)
    optimizer = build_optimizer("adam_nv", model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    params = [p for g in optimizer.param_groups for p in g["params"]]

    os.makedirs(args.log_dir, exist_ok=True)
    tag = f"ln{args.label_noise}_seed{args.seed}"
    diag_path = os.path.join(args.log_dir, f"proxy2_diag_{tag}.jsonl")
    train_path = os.path.join(args.log_dir, f"proxy2_train_{tag}.csv")
    if os.path.exists(diag_path):
        os.remove(diag_path)
    with open(os.path.join(args.log_dir, f"proxy2_config_{tag}.json"), "w") as fh:
        json.dump({"arguments": vars(args), "torch_version": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    with open(train_path, "w") as fh:
        fh.write("epoch,train_loss,val_loss,val_acc\n")

    checkpoints = set(args.checkpoints)
    n_steps = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        stream = StreamAccumulator(params) if epoch in checkpoints else None
        for step, batch in enumerate(train_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            if stream is not None and step >= n_steps - args.stream_steps:
                stream.add(params)
            optimizer.step()
            total += float(loss) * x.size(0)
            n += x.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)
        with open(train_path, "a") as fh:
            fh.write(f"{epoch},{total / n:.6f},{val_loss:.6f},{val_acc:.6f}\n")
        if epoch in checkpoints:
            rec = collect_diagnostics_v2(model, optimizer, train_loader.dataset, args, device, epoch, stream)
            with open(diag_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            wr = np.nanmedian([t["r_s_qu"] for t in rec["tensors"] if t["rank"] == 4]); rel = np.nanmedian([t["reliability"] for t in rec["tensors"] if t["rank"] == 4])
            print(f"[diag] epoch {epoch}: pooled s-qu {rec['spearman_s_qu']:.3f}; median within-conv-tensor s-qu {wr:.3f}, reliability {rel:.3f}; mean qu {rec['mean_qu']:.3f}", flush=True)
        print(f"[train] epoch {epoch}/{args.epochs} loss {total / n:.4f} val_acc {val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
