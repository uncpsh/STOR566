# ===== STOR566 Final: HiLoViT vs Transformer vs ViT vs ResNet18 (Mini-ImageNet) =====
# Dataset: mini-ImageNet (84x84), 4 models, same metrics, same tables/plots as CIFAR script

import os
import time
import random
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from thop import profile
from sklearn.metrics import roc_auc_score, f1_score

import matplotlib
matplotlib.use("Agg")  # no display (HPC-safe)
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from hilo import HiLo  # make sure this is importable on Longleaf

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MINI_ROOT = "./mini_imagenet"   # <<< change this to your dataset root >>>
IMG_SIZE = 84                   # standard mini-ImageNet resolution
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 20
OUTDIR = "./exp_outputs_mini_imagenet"

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------------------------------------------------
# mini-ImageNet data
# ------------------------------------------------------------------
def get_mini_imagenet_loaders(root=MINI_ROOT,
                              img_size=IMG_SIZE,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS):
    """
    Assumes folder structure:
        root/
          train/class_x/*.jpg
          train/class_y/*.jpg
          val/class_x/*.jpg
          val/class_y/*.jpg
    with images roughly 84x84 (standard mini-ImageNet).
    """
    mean = (0.485, 0.456, 0.406)  # ImageNet stats
    std  = (0.229, 0.224, 0.225)
    norm = T.Normalize(mean, std)

    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])

    val_tf = T.Compose([
        T.Resize(int(img_size * 1.1)),  # e.g., 92 for 84
        T.CenterCrop(img_size),
        T.ToTensor(),
        norm,
    ])

    train_ds = ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds   = ImageFolder(os.path.join(root, "val"),   transform=val_tf)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes, img_size

train_loader, val_loader, num_classes, img_size = get_mini_imagenet_loaders()
print("Train images:", len(train_loader.dataset),
      "| Val images:", len(val_loader.dataset),
      "| Num classes:", num_classes,
      "| Img size:", img_size)

# ------------------------------------------------------------------
# Shared building blocks for ViT-style models
# ------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=84, patch_size=7, in_ch=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.grid = img_size // patch_size

    def forward(self, x):
        x = self.proj(x)                    # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)    # (B, N, C)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ------------------------------------------------------------------
# 1) Vanilla Transformer on patch tokens (ViT-MSA)
# ------------------------------------------------------------------
class ViTMSA(nn.Module):
    """
    Minimal ViT-style model:
    - Patch embedding
    - CLS token + Learnable positional embedding
    - Stack of TransformerEncoder layers with MSA
    """
    def __init__(self, img_size=84, patch_size=7, num_classes=64,
                 embed_dim=192, depth=6, num_heads=6,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)                        # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, C)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1), :]
        x = self.blocks(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ------------------------------------------------------------------
# 2) Vanilla ViT (custom MHSA blocks)
# ------------------------------------------------------------------
class MHSA(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    # (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class ViTVanillaBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViTVanilla(nn.Module):
    def __init__(self, img_size=84, patch_size=7, num_classes=64,
                 embed_dim=192, depth=6, num_heads=6,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            ViTVanillaBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ------------------------------------------------------------------
# 3) HiLoViT (using official HiLo attention)
# ------------------------------------------------------------------
class HiLoBlock(nn.Module):
    def __init__(self, dim, num_heads=6,
                 window_size=2, alpha=0.5,
                 drop=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HiLo(dim, num_heads, window_size, alpha)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.drop_path = nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class HiLoViT(nn.Module):
    def __init__(self, img_size=84, patch_size=7, num_classes=64,
                 embed_dim=192, depth=6, num_heads=6,
                 window_size=2, alpha=0.5):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size

        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.grid * self.grid

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.blocks = nn.ModuleList([
            HiLoBlock(embed_dim, num_heads, window_size, alpha)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B = x.size(0)
        H = W = self.grid
        x = self.patch(x)                     # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1), :]

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ------------------------------------------------------------------
# 4) ResNet-18
# ------------------------------------------------------------------
def get_resnet18(num_classes=64):
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops(model, img_size=IMG_SIZE):
    """
    Compute FLOPs on a CPU copy so we don't change the original model's device.
    """
    m = copy.deepcopy(model).to("cpu").eval()
    x = torch.randn(1, 3, img_size, img_size)
    try:
        flops, _ = profile(m, inputs=(x,), verbose=False)
    except Exception:
        flops = -1
    return int(flops)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    """
    Evaluate on validation set:
    - average loss
    - top-1 accuracy
    - average inference batch time
    - throughput (images / second)
    - raw y_true, y_pred, y_prob for AUC/F1
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    batch_times = []
    all_true = []
    all_pred = []
    all_probs = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        batch_times.append(dt)

        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (pred == y).sum().item()
        total += bs

        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())
        all_probs.append(probs.detach().cpu())

    avg_loss = total_loss / total
    acc = 100.0 * total_correct / total

    batch_times = np.array(batch_times, dtype=np.float64)
    avg_batch_time = float(batch_times.mean())
    total_time = float(batch_times.sum())
    throughput = total / total_time if total_time > 0 else float("nan")

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_probs).numpy()

    return avg_loss, acc, avg_batch_time, throughput, y_true, y_pred, y_prob

# ------------------------------------------------------------------
# Training loop for one model
# ------------------------------------------------------------------
def train_one_model(model, name, train_loader, val_loader,
                    epochs=EPOCHS, lr=3e-4, wd=1e-4,
                    outdir=OUTDIR, img_size=IMG_SIZE):
    os.makedirs(outdir, exist_ok=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # For FLOPs / params
    params = count_params(model)
    flops = get_flops(model, img_size=img_size)

    hist = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        run_correct = 0
        n = 0
        ep0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = y.size(0)
            run_loss += loss.item() * bs
            run_correct += (logits.argmax(1) == y).sum().item()
            n += bs

        ep1 = time.time()
        epoch_time = ep1 - ep0

        train_loss = run_loss / n
        train_acc = 100.0 * run_correct / n

        # --- evaluate on validation set ---
        val_loss, val_acc, avg_inf_batch_s, infer_throughput, \
        y_true, y_pred, y_prob = eval_epoch(model, val_loader, criterion)

        # AUC (macro, one-vs-rest)
        try:
            val_auc = roc_auc_score(
                y_true, y_prob,
                multi_class="ovr", average="macro"
            )
        except ValueError:
            val_auc = float("nan")

        # F1-score (macro)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mem = float("nan")

        hist.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_top1": train_acc,
            "val_loss": val_loss,
            "val_top1": val_acc,
            "val_auc_macro": val_auc,
            "val_f1_macro": val_f1,
            "epoch_time_s": epoch_time,
            "avg_infer_batch_s": avg_inf_batch_s,
            "throughput_img_s": infer_throughput,
            "peak_gpu_mem_MB": peak_mem,
            "lr": opt.param_groups[0]["lr"],
        })

        print(f"[{name}] Epoch {ep}/{epochs} | "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.2f}, AUC={val_auc:.4f}, F1={val_f1:.4f}, "
              f"time={epoch_time:.1f}s")

    total_train_time = time.time() - t_train0
    if device.type == "cuda":
        peak_gpu_mem_training = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_gpu_mem_training = float("nan")

    hist_df = pd.DataFrame(hist)
    hist_path = os.path.join(outdir, f"history_{name}.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[{name}] Saved history -> {hist_path}")

    # --- summary metrics over epochs ---
    best_idx = int(hist_df["val_top1"].idxmax())
    best_row = hist_df.iloc[best_idx]
    last_row = hist_df.iloc[-1]

    summary = {
        "model": name,
        # core validation outcomes
        "val_loss_mean": float(hist_df["val_loss"].mean()),
        "val_loss_last": float(last_row["val_loss"]),
        "val_auc_macro_last": float(last_row["val_auc_macro"]),
        "val_f1_macro_last": float(last_row["val_f1_macro"]),
        "val_acc_last": float(last_row["val_top1"]),
        "val_acc_best": float(best_row["val_top1"]),
        "epoch_of_best_val_acc": int(best_row["epoch"]),
        # size & complexity
        "params": int(params),
        "FLOPs_thop": float(flops),
        # speed
        "avg_epoch_time_s": float(hist_df["epoch_time_s"].mean()),
        "total_train_time_s": float(total_train_time),
        "infer_throughput_img_s_last": float(last_row["throughput_img_s"]),
        # GPU memory
        "peak_gpu_mem_MB_last": float(last_row["peak_gpu_mem_MB"]),
        "peak_gpu_mem_MB_training": float(peak_gpu_mem_training),
    }

    return summary, hist_df

# ------------------------------------------------------------------
# Plots / charts for comparison (same format as CIFAR script)
# ------------------------------------------------------------------
def make_plots(summary_df, histories, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)

    # 1) Validation accuracy (last epoch)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["val_acc_last"])
    plt.ylabel("Validation Accuracy (last epoch, %)")
    plt.title("Validation Accuracy (last epoch)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "val_acc_last.png"))
    plt.close()

    # 2) Validation AUC (macro, last epoch)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["val_auc_macro_last"])
    plt.ylabel("Macro AUC (one-vs-rest, last epoch)")
    plt.title("Validation AUC (macro)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "val_auc_last.png"))
    plt.close()

    # 3) Validation F1-score (macro, last epoch)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["val_f1_macro_last"])
    plt.ylabel("Macro F1-score (last epoch)")
    plt.title("Validation F1-score (macro)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "val_f1_last.png"))
    plt.close()

    # 4) Validation average loss (across epochs)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["val_loss_mean"])
    plt.ylabel("Mean Validation Loss")
    plt.title("Validation Loss (mean across epochs)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "val_loss_mean.png"))
    plt.close()

    # 5) Parameter count (log scale)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["params"])
    plt.yscale("log")
    plt.ylabel("Number of Parameters (log scale)")
    plt.title("Model Size (Parameters)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "params.png"))
    plt.close()

    # 6) FLOPs (log scale)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["FLOPs_thop"])
    plt.yscale("log")
    plt.ylabel("FLOPs (log scale)")
    plt.title("Model FLOPs (Img size)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "flops.png"))
    plt.close()

    # 7) Inference throughput (images/sec, last epoch)
    plt.figure(figsize=(7, 5))
    plt.bar(summary_df.index, summary_df["infer_throughput_img_s_last"])
    plt.ylabel("Inference Throughput (img/s)")
    plt.title("Inference Throughput on Validation Set")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "throughput.png"))
    plt.close()

    # 8) Per-epoch validation accuracy curves
    plt.figure(figsize=(7, 5))
    for name, hist_df in histories.items():
        plt.plot(hist_df["epoch"], hist_df["val_top1"], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "val_acc_vs_epoch.png"))
    plt.close()

    # 9) Per-epoch validation loss curves
    plt.figure(figsize=(7, 5))
    for name, hist_df in histories.items():
        plt.plot(hist_df["epoch"], hist_df["val_loss"], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "val_loss_vs_epoch.png"))
    plt.close()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    outdir = OUTDIR
    epochs = EPOCHS

    # (already loaded once above, but do it here for clarity / standalone)
    train_loader, val_loader, num_classes, img_size = get_mini_imagenet_loaders()

    # choose patch size for 84x84
    patch_size = 7

    models_to_run = {
        "resnet18": lambda: get_resnet18(num_classes),
        "vit_msa": lambda: ViTMSA(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes
        ),
        "vit_vanilla": lambda: ViTVanilla(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes
        ),
        "hilovit": lambda: HiLoViT(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes,
            embed_dim=192, depth=6, num_heads=6,
            window_size=2, alpha=0.5
        ),
    }

    summaries = []
    histories = {}

    for name, ctor in models_to_run.items():
        set_seed(42)  # reset seed for fairness
        model = ctor()
        summary, hist_df = train_one_model(
            model, name, train_loader, val_loader,
            epochs=epochs, lr=3e-4, wd=1e-4,
            outdir=outdir, img_size=img_size
        )
        summaries.append(summary)
        histories[name] = hist_df

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(outdir, "summary_mini_imagenet_all_models.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved summary ->", summary_path)
    print(summary_df)

    # ensure consistent order
    summary_df = summary_df.set_index("model").loc[list(models_to_run.keys())]

    make_plots(summary_df, histories, outdir=outdir)
    print("Saved plots in", outdir)

if __name__ == "__main__":
    main()
