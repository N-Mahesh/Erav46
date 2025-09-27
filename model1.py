"""
CHATGPT was used for formatting of ALL LARGE COMMENTS INTO PROPER READABILITY IN THESE FILES
TARGET:
- Build a minimal CNN (<8k params) to establish a baseline within ≤15 epochs.
- Aim for ≥97% validation accuracy to verify the pipeline and RF growth.

RESULT (rounded, typical on a fresh run):
- Validation ≈ 96.5% – 97.2%
- Test ≈ 96.5% – 97.5%

ANALYSIS:
- Underfitting: capacity is too small; no dropout needed yet, but width is insufficient to reach 99%+.
- Kept 3×3-only convs, BN after every conv, MaxPool after each block, GAP+Linear head.
- Light RandomAffine helped robustness; label smoothing off to avoid lowering the peak further at this size.
- Receptive Field (RF):
  • Block1: Conv3×3 → Conv3×3 → RF=5; MaxPool2 → RF=6 (jump=2)
  • Block2: Conv3×3 → Conv3×3 → RF=10; MaxPool2 → RF=12 (jump=4)
  • GAP over 7×7 integrates globally.
- Next step: increase width and add a tiny dropout before the head; consider label smoothing and OneCycle LR to push >98%.

Notes for mapping:
- Layers: 4 convs + 2 pools + GAP + Linear
- MaxPooling position: after feature pairs (transition layers), not next to head
- 1×1 Convs: omitted in this baseline to keep things simple
- Softmax: metrics only; loss uses logits (CrossEntropy)
- BatchNorm: after every conv
- Image Normalization: MNIST (0.1307, 0.3081)
- Early failure signal: if val <95% by epoch 3, LR/aug need tuning or width too small
- Batch size: 256 for speed; can try 128 for tiny generalization gains

Device: cuda
Trainable params: 3058
Epoch 1/15 | train_loss=2.013 | train_acc=36.8% | val_acc=69.46% | test_acc=72.71% | params=3058
Epoch 2/15 | train_loss=0.825 | train_acc=83.0% | val_acc=83.81% | test_acc=86.96% | params=3058
Epoch 3/15 | train_loss=0.278 | train_acc=93.2% | val_acc=91.26% | test_acc=93.77% | params=3058
Epoch 4/15 | train_loss=0.180 | train_acc=95.1% | val_acc=91.93% | test_acc=94.26% | params=3058
Epoch 5/15 | train_loss=0.137 | train_acc=96.1% | val_acc=94.82% | test_acc=95.80% | params=3058
Epoch 6/15 | train_loss=0.117 | train_acc=96.6% | val_acc=96.51% | test_acc=97.15% | params=3058
Epoch 7/15 | train_loss=0.106 | train_acc=96.9% | val_acc=95.96% | test_acc=96.72% | params=3058
Epoch 8/15 | train_loss=0.096 | train_acc=97.1% | val_acc=96.69% | test_acc=97.42% | params=3058
Epoch 9/15 | train_loss=0.087 | train_acc=97.4% | val_acc=97.00% | test_acc=97.88% | params=3058
Epoch 10/15 | train_loss=0.075 | train_acc=97.7% | val_acc=96.57% | test_acc=97.15% | params=3058
Epoch 11/15 | train_loss=0.075 | train_acc=97.8% | val_acc=97.80% | test_acc=98.48% | params=3058
Epoch 12/15 | train_loss=0.067 | train_acc=98.0% | val_acc=97.89% | test_acc=98.32% | params=3058
Epoch 13/15 | train_loss=0.063 | train_acc=98.1% | val_acc=98.07% | test_acc=98.58% | params=3058
Epoch 14/15 | train_loss=0.059 | train_acc=98.2% | val_acc=98.23% | test_acc=98.69% | params=3058
Epoch 15/15 | train_loss=0.059 | train_acc=98.2% | val_acc=98.20% | test_acc=98.68% | params=3058
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

class ModelBaseline(nn.Module):
    # Params ~3.1k
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=True),  nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=True),  nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, 3, padding=1, bias=True), nn.BatchNorm2d(12), nn.ReLU(inplace=True),
            nn.Conv2d(12, 12, 3, padding=1, bias=True),nn.BatchNorm2d(12), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(12, 10)
    def forward(self, x):
        x = self.f(x); x = self.gap(x).flatten(1); return self.head(x)

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_t = transforms.Compose([
        transforms.RandomAffine(degrees=7, translate=(0.08,0.08), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
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

    model = ModelBaseline().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", params)

    assert params <= PARAM_BUDGET

    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    sched = OneCycleLR(opt, max_lr=LR_MAX, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3)
    criterion = nn.CrossEntropyLoss()

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

    print("Done.")

if __name__ == "__main__":
    main()
