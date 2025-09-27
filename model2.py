"""
TARGET:
- Stay ≤8k params and push validation beyond 98% within ≤15 epochs.
- Add dropout near the head, label smoothing, and keep BN everywhere.

RESULT (rounded, typical on a fresh run):
- Validation ≈ 98.1% – 98.6%
- Test ≈ 98.8% – 99.2%

ANALYSIS:
- Better capacity (wider channels) and improved optimization (OneCycle, LS) lift accuracy into high-98s.
- Still misses consistent 99.4%: RandomErasing (if used) can slightly lower peak; we reduce/omit it here.
- Introduces a 1×1 transition before GAP to enrich features cheaply (lecture: transition layers).
- Receptive Field (RF):
  • Block1: Conv3×3 → Conv3×3 → RF=5; MaxPool2 → RF=6 (jump=2)
  • Block2: Conv3×3 → Conv3×3 → RF=10; MaxPool2 → RF=12 (jump=4)
  • 1×1 transition keeps RF=12; GAP integrates globally.
- Next step to reach 99.4%: tighten augmentation (lighter rotations), replace ReLU with GELU,
  tune LR_MAX upward a bit (0.02–0.025), and keep dropout modest (0.05–0.10). That becomes Attempt 3.

Features we have learned:
- Layers: 4 convs + 2 pools + GAP + Linear
- MaxPooling placement: after each conv pair
- 1×1 Convolution: used as a transition layer pre-GAP
- BN after every conv; image normalization; dropout close to head
- Early failure detection at epoch 3; batch size tradeoffs noted

Device: cuda
Trainable params: 7494
Epoch 1/15 | train_loss=1.793 | train_acc=49.5% | val_acc=87.13% | test_acc=90.52% | params=7494
Epoch 2/15 | train_loss=0.803 | train_acc=92.9% | val_acc=93.86% | test_acc=94.95% | params=7494
Epoch 3/15 | train_loss=0.701 | train_acc=95.7% | val_acc=95.64% | test_acc=97.06% | params=7494
[Signal] Val<97% by epoch 3 — reduce degrees to 7°, or try LR_MAX=0.02.
Epoch 4/15 | train_loss=0.666 | train_acc=96.7% | val_acc=95.64% | test_acc=97.43% | params=7494
Epoch 5/15 | train_loss=0.644 | train_acc=97.4% | val_acc=97.45% | test_acc=98.46% | params=7494
Epoch 6/15 | train_loss=0.633 | train_acc=97.7% | val_acc=97.62% | test_acc=98.47% | params=7494
Epoch 7/15 | train_loss=0.625 | train_acc=97.8% | val_acc=98.15% | test_acc=98.80% | params=7494
Epoch 8/15 | train_loss=0.621 | train_acc=97.9% | val_acc=98.41% | test_acc=98.83% | params=7494
Epoch 9/15 | train_loss=0.613 | train_acc=98.1% | val_acc=98.35% | test_acc=98.90% | params=7494
Epoch 10/15 | train_loss=0.607 | train_acc=98.2% | val_acc=98.65% | test_acc=99.02% | params=7494
Epoch 11/15 | train_loss=0.605 | train_acc=98.2% | val_acc=98.70% | test_acc=99.22% | params=7494
Epoch 12/15 | train_loss=0.600 | train_acc=98.5% | val_acc=98.53% | test_acc=99.12% | params=7494
Epoch 13/15 | train_loss=0.599 | train_acc=98.4% | val_acc=98.69% | test_acc=99.18% | params=7494
Epoch 14/15 | train_loss=0.596 | train_acc=98.5% | val_acc=98.69% | test_acc=99.13% | params=7494
Epoch 15/15 | train_loss=0.595 | train_acc=98.6% | val_acc=98.75% | test_acc=99.16% | params=7494
Done.
"""

import os, time, random, math
from contextlib import nullcontext
import numpy as np

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

EPOCHS, BATCH_SIZE, LR_MAX = 15, 256, 1.5e-2
WEIGHT_DECAY, NUM_WORKERS, SEED = 1e-4, 2, 123
CHANNELS_LAST, PARAM_BUDGET = False, 8_000
LS_EPS = 0.10

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class SubsetWithTransform(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices); self.t = transform; self.base = dataset
    def __getitem__(self, i):
        img = self.base.data[self.indices[i]].numpy()
        from PIL import Image
        x = self.t(Image.fromarray(img, mode="L"))
        y = int(self.base.targets[self.indices[i]])
        return x, y

def accuracy(logits, y): return (logits.argmax(1) == y).float().mean().item()

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); n=0; correct=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1)==y).sum().item(); n += y.size(0)
    return correct/max(1,n)

class ModelWiderDrop(nn.Module):
    # Params ~6.6k
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8, 3, padding=1, bias=True),  nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.Conv2d(8, 12, 3, padding=1, bias=True), nn.BatchNorm2d(12), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(12, 16, 3, padding=1, bias=True), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, padding=1, bias=True), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 1×1 transition before GAP
            nn.Conv2d(24, 32, 1, bias=True), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.10)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        x = self.f(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.head(x)

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_t = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.10,0.10), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Note: intentionally no RandomErasing here to preserve peak
    ])
    eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_full = datasets.MNIST("./data", train=True, download=True, transform=train_t)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=eval_t)

    gen = torch.Generator().manual_seed(SEED)
    train_size, val_size = 50_000, 10_000
    train_subset_raw, val_subset_raw = torch.utils.data.random_split(train_full, [train_size, val_size], generator=gen)
    val_subset = SubsetWithTransform(train_full, val_subset_raw.indices, eval_t)

    pin = (device=="cuda")
    train_loader = DataLoader(train_subset_raw, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_subset,       batch_size=512,    shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,          batch_size=512,    shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    model = ModelWiderDrop().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", params); assert params <= PARAM_BUDGET

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    sched = OneCycleLR(opt, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.25)
    criterion = nn.CrossEntropyLoss(label_smoothing=LS_EPS)

    start=time.time()
    for epoch in range(1, EPOCHS+1):
        model.train(); rl=0; ra=0; nb=0
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb); loss = criterion(out, yb)
            loss.backward(); opt.step(); sched.step()
            rl += loss.item(); ra += accuracy(out.detach(), yb); nb += 1

        trl, tra = rl/nb, ra/nb
        va = evaluate(model, val_loader, device)
        ta = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={trl:.3f} | train_acc={tra*100:.1f}% | val_acc={va*100:.2f}% | test_acc={ta*100:.2f}% | params={params}")

        if epoch == 3 and va < 0.97:
            print("[Signal] Val<97% by epoch 3 — reduce degrees to 7°, or try LR_MAX=0.02.")

    print("Done.")

if __name__ == "__main__":
    main()
