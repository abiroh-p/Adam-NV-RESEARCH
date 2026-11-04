"""Alternative CIFAR architectures for Exp. 17 (generalization beyond ResNet-18).

  vit_tiny : a small Vision Transformer for 32x32 inputs. Patch size 4 (64 tokens plus a class
             token), width 192, depth 6, 3 heads, MLP width 384, pre-LayerNorm blocks, learned
             position embeddings, no dropout. About 1.8 million parameters. It uses LayerNorm
             instead of BatchNorm and has no convolutional inductive bias beyond the patch embedding.
  vgg11_bn : VGG-11 with BatchNorm (no residual connections), a single linear classifier on the
             512-dimensional output; about 9.2 million parameters.

No pretrained weights, no tuning. `build(arch, num_classes)` returns a fresh randomly initialized model.
"""

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, num_classes=10, image_size=32, patch=4, dim=192, depth=6, heads=3, mlp_dim=384):
        super().__init__()
        n_tokens = (image_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[_Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


def _vgg11_bn(num_classes=10):
    cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(512, num_classes))


ARCHS = ("vit_tiny", "vgg11_bn")


def build(arch, num_classes=10):
    if arch == "vit_tiny":
        return ViTTiny(num_classes=num_classes)
    if arch == "vgg11_bn":
        return _vgg11_bn(num_classes)
    raise ValueError(f"Unknown architecture {arch!r} (use one of {ARCHS})")
