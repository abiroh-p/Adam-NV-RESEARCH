"""Noise-proxy diagnostic (Exp. 16): does Adam-NV's moment-based score track gradient noise?

Adam-NV's controller uses the proxy
    s = max(v - m^2, 0) / (v + lam*|m| + eps)
built from the optimizer's running moments m (first) and v (second) of the gradient. This script
trains a ResNet-18 with Adam-NV (default settings, as in the label-noise experiments) and, after
selected epochs, measures the gradient noise DIRECTLY at the current weights:

  * K minibatch gradients g_1..g_K are computed at fixed weights (BatchNorm in training mode, the
    running statistics are restored afterwards, no optimizer step is taken);
  * per element: mean_sq = mean(g_k^2), mu = mean(g_k), var = unbiased variance of g_k, and the
    direct noise fraction  q = var / mean_sq  in [0, 1]  (the fraction of the gradient's second moment
    that is variance across minibatches: the direct counterpart of s);
  * scale_ratio = sqrt(mean_sq) / sqrt(v): the current gradient scale relative to the optimizer's
    long-memory second moment. If s mainly reflects shrinking gradients (v remembers larger past
    gradients), s will correlate with 1/scale_ratio and not with q.

For every checkpoint it writes one JSON line with Spearman rank correlations (pooled over a random
sample of 500,000 elements, by tensor rank, and the median over convolution tensors):
  spearman_s_q, spearman_rho_q, spearman_s_scale, spearman_q_scale, plus means of s, rho and q.

Conditions to run (see the runbook): clean labels and 20% label noise, seeds 0 and 1.

Example:
    python3 noise_proxy_diagnostic.py --seed 0 --label-noise 0.2 --epochs 60 --data-dir ../data \
        --log-dir ../logs_exp16_ln02_s0
CPU smoke test:
    python3 noise_proxy_diagnostic.py --seed 0 --epochs 2 --checkpoints 1 2 --k-batches 2 \
        --smoke-test --synthetic-data --data-dir /tmp/d --log-dir /tmp/diag_smoke
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


def spearman(a, b):
    """Spearman rank correlation of two 1-D float tensors (average ranks for ties)."""
    a, b = a.double().cpu().numpy(), b.double().cpu().numpy()
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


def direct_noise_fraction(sum_g, sum_g2, k):
    """Per-element direct noise fraction q = var / mean_sq from K accumulated gradients."""
    mean_sq = sum_g2 / k
    mu = sum_g / k
    var = torch.clamp((sum_g2 - k * mu * mu) / (k - 1), min=0.0)
    q = torch.clamp(var / (mean_sq + 1e-12), max=1.0)
    return q, mean_sq


def proxy_score(m, v, lam, eps):
    """Adam-NV's score s from the optimizer moments (same formula as in adam_nv.py)."""
    return torch.clamp(v - m * m, min=0.0) / (v + lam * m.abs() + eps)


def collect_diagnostics(model, optimizer, dataset, args, device, epoch):
    gen = torch.Generator().manual_seed(777 + epoch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    params = [p for g in optimizer.param_groups for p in g["params"]]
    sum_g = [torch.zeros_like(p) for p in params]
    sum_g2 = [torch.zeros_like(p) for p in params]
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    done = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        for i, p in enumerate(params):
            sum_g[i] += p.grad
            sum_g2[i] += p.grad * p.grad
        done += 1
        if done == args.k_batches:
            break
    model.load_state_dict(saved)
    model.zero_grad(set_to_none=True)
    k = done
    if k < 2:
        raise RuntimeError("need at least 2 minibatches for a variance estimate")

    s_all, rho_all, q_all, sc_all, rank_all, tensor_corr = [], [], [], [], [], []
    group_of = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    for i, p in enumerate(params):
        st = optimizer.state[p]
        if "m" not in st:
            continue
        g = group_of[id(p)]
        q, mean_sq = direct_noise_fraction(sum_g[i], sum_g2[i], k)
        s = proxy_score(st["m"], st["v"], g["lam"], g["eps"])
        scale = torch.sqrt(mean_sq) / (torch.sqrt(st["v"]) + 1e-12)
        s_all.append(s.reshape(-1).float().cpu())
        rho_all.append(st["rho"].reshape(-1).float().cpu())
        q_all.append(q.reshape(-1).float().cpu())
        sc_all.append(scale.reshape(-1).float().cpu())
        rank_all.append(torch.full((p.numel(),), p.dim(), dtype=torch.int8))
        if p.dim() == 4:
            idx = torch.randperm(p.numel(), generator=torch.Generator().manual_seed(i))[:20000]
            tensor_corr.append(spearman(s.reshape(-1).cpu()[idx], q.reshape(-1).cpu()[idx]))
    s, rho, q, sc, rk = (torch.cat(x) for x in (s_all, rho_all, q_all, sc_all, rank_all))
    rng = np.random.default_rng(args.seed * 1000 + epoch)
    sel = torch.from_numpy(rng.choice(s.numel(), size=min(args.sample, s.numel()), replace=False))
    out = {
        "epoch": epoch, "k_batches": k, "n_elements": int(s.numel()), "n_sampled": int(sel.numel()),
        "mean_s": float(s.mean()), "mean_rho": float(rho.mean()), "mean_q": float(q.mean()),
        "frac_s_zero": float((s == 0).float().mean()),
        "spearman_s_q": spearman(s[sel], q[sel]), "spearman_rho_q": spearman(rho[sel], q[sel]),
        "spearman_s_scale": spearman(s[sel], sc[sel]), "spearman_q_scale": spearman(q[sel], sc[sel]),
        "median_conv_tensor_spearman_s_q": float(np.nanmedian(tensor_corr)) if tensor_corr else float("nan"),
    }
    for r in (1, 2, 4):
        m = (rk == r)
        if int(m.sum()) >= 3:
            idx = torch.nonzero(m).squeeze(1)
            if idx.numel() > args.sample:
                idx = idx[torch.from_numpy(rng.choice(idx.numel(), size=args.sample, replace=False))]
            out[f"spearman_s_q_rank{r}"] = spearman(s[idx], q[idx])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 10, 20, 30, 45, 60])
    ap.add_argument("--k-batches", type=int, default=16)
    ap.add_argument("--sample", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
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

    os.makedirs(args.log_dir, exist_ok=True)
    tag = f"ln{args.label_noise}_seed{args.seed}"
    diag_path = os.path.join(args.log_dir, f"proxy_diag_{tag}.jsonl")
    train_path = os.path.join(args.log_dir, f"proxy_train_{tag}.csv")
    if os.path.exists(diag_path):
        os.remove(diag_path)
    with open(os.path.join(args.log_dir, f"proxy_config_{tag}.json"), "w") as fh:
        json.dump({"arguments": vars(args), "torch_version": torch.__version__,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)
    with open(train_path, "w") as fh:
        fh.write("epoch,train_loss,val_loss,val_acc\n")

    checkpoints = set(args.checkpoints)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss) * x.size(0)
            n += x.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)
        with open(train_path, "a") as fh:
            fh.write(f"{epoch},{total / n:.6f},{val_loss:.6f},{val_acc:.6f}\n")
        if epoch in checkpoints:
            rec = collect_diagnostics(model, optimizer, train_loader.dataset, args, device, epoch)
            with open(diag_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[diag] epoch {epoch}: spearman(s,q)={rec['spearman_s_q']:.3f} "
                  f"spearman(s,scale)={rec['spearman_s_scale']:.3f} mean_q={rec['mean_q']:.3f}", flush=True)
        print(f"[train] epoch {epoch}/{args.epochs} loss {total / n:.4f} val_acc {val_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
