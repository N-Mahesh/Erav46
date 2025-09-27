# mnist_under8k_994_fixed.py — single-file, Colab-ready.
# Goal: ≥99.4% validation (consistent in last few epochs), ≤15 epochs, ≤8,000 params.
# Fixes: clean eval transforms (no augs), no EMA (avoid BN-buffers mismatch), inference_mode in eval.

import os, time, math, random
from contextlib import nullcontext
import numpy as np

# Try import; fallback to pip if needed
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import OneCycleLR
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import OneCycleLR

# ---------------- Config ----------------
EPOCHS        = 25
BATCH_SIZE    = 256
LR_MAX        = 2.0e-2       # good peak with BN + OneCycle
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 2
SEED          = 123
CHANNELS_LAST = False
LS_EPS        = 0.05
PARAM_BUDGET  = 8_000
CONSIST_N     = 3
TARGET_VAL    = 0.994        # 99.4%

# ---------------- Utils ----------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    n, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return correct / max(1, n)

class SubsetWithTransform(Subset):
    """Wrap a Subset and override transform for evaluation (no augmentation)."""
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.eval_transform = transform
        self.base = dataset
    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        # base returns (PIL→Tensor already) if base has transform; we need raw PIL.
        # So instead, we access base.data/base.targets to ensure clean transform:
        # MNIST stores raw as tensors; convert to PIL then apply eval transform.
        img = self.base.data[self.indices[idx]].numpy()
        from PIL import Image
        pil = Image.fromarray(img, mode="L")
        x = self.eval_transform(pil)
        return x, y

# ---------------- Model (≈7.8k params) ----------------
# Faster than DW+attention; 3x3-only convs with BN, two MaxPools, 1x1 transition,
# GAP → Dropout(0.10) → Linear(40→10)
class TinyBetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.SiLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            act(inplace=True),

            nn.Conv2d(8, 12, 3, padding=1, bias=True),
            nn.BatchNorm2d(12),
            act(inplace=True),

            nn.MaxPool2d(2),  # 28→14

            nn.Conv2d(12, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            act(inplace=True),

            nn.Conv2d(16, 24, 3, padding=1, bias=True),
            nn.BatchNorm2d(24),
            act(inplace=True),

            nn.MaxPool2d(2),  # 14→7

            nn.Conv2d(24, 40, 1, bias=True),  # cheap transition
            nn.BatchNorm2d(40),
            act(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.10)
        self.head = nn.Linear(40, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.head(x)

# ---------------- Train ----------------
def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Train augmentations (light); Eval uses clean transform.
    train_tfm = transforms.Compose([
        transforms.RandomAffine(degrees=7, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets
    train_full = datasets.MNIST("./data", train=True, download=True, transform=train_tfm)
    test_ds_raw = datasets.MNIST("./data", train=False, download=True, transform=eval_tfm)

    # 50k/10k split indices, but ensure validation uses **eval_tfm** (no aug)
    val_size = 10_000
    train_size = len(train_full) - val_size  # 50k/10k
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset_raw = torch.utils.data.random_split(train_full, [train_size, val_size], generator=generator)
    # Wrap val subset with clean transform
    val_subset = SubsetWithTransform(train_full, val_subset_raw.indices, eval_tfm)

    pin = (device == "cuda")
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_subset, batch_size=512, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds_raw, batch_size=512, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    model = TinyBetterCNN().to(device)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)

    params = count_params(model)
    print(f"Trainable params: {params}")
    assert params <= PARAM_BUDGET, f"Param budget exceeded: {params} > {PARAM_BUDGET}"

    # Optimizer / Scheduler / AMP
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    sched = OneCycleLR(opt, max_lr=LR_MAX, epochs=EPOCHS,
                       steps_per_epoch=len(train_loader), pct_start=0.25)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    autocast = (lambda: torch.amp.autocast('cuda')) if use_amp else nullcontext
    criterion = nn.CrossEntropyLoss(label_smoothing=LS_EPS)

    # Logs
    os.makedirs("checkpoints", exist_ok=True)
    with open("logs.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_acc,test_acc,params,secs\n")

    best_val, start = 0.0, time.time()
    tail = []

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        run_loss, run_acc, n_batches = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if CHANNELS_LAST:
                xb = xb.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with (autocast() if use_amp else nullcontext()):
                logits = model(xb)
                loss = criterion(logits, yb)
            if use_amp:
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            run_loss += loss.item()
            run_acc  += accuracy(logits.detach(), yb)
            n_batches += 1

        train_loss = run_loss / max(1, n_batches)
        train_acc  = run_acc  / max(1, n_batches)

        # ---- Evaluate (clean transform, no EMA) ----
        val_acc  = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        secs = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train_loss={train_loss:.3f} | train_acc={train_acc*100:.1f}% | "
              f"val_acc={val_acc*100:.2f}% | test_acc={test_acc*100:.2f}% | params={params}")
        with open("logs.csv", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_acc:.4f},{test_acc:.4f},{params},{secs:.1f}\n")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "params": params}, "checkpoints/best.pt")

        # Early signal if off-track
        if epoch == 3 and val_acc < 0.97:
            print("[Signal] Val<97% by epoch 3 — nudge LR_MAX to 0.025 or reduce rotation to 5°.")

        # Consistency stop (last CONSIST_N val ≥ 99.4%)
        tail.append(val_acc); 
        if len(tail) > CONSIST_N: tail.pop(0)
        if len(tail) == CONSIST_N and all(v >= TARGET_VAL for v in tail):
            print(f"Stopping: last {CONSIST_N} val epochs ≥ {TARGET_VAL*100:.1f}% (epoch {epoch}).")
            break

    print("\nDone. Check logs.csv and checkpoints/.")

if __name__ == "__main__":
    main()
