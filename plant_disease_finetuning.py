"""
============================================================
  CASE STUDY: Plant Disease Detection
  Fine-Tuning a Pre-trained ResNet50 CNN
============================================================

OBJECTIVE:
    Classify leaf images into 4 categories:
        0 - Healthy
        1 - Bacterial Blight
        2 - Leaf Spot
        3 - Rust

APPROACH:
    1. Load ResNet50 pre-trained on ImageNet
    2. Freeze early convolutional layers (feature extraction)
    3. Replace the classifier head for our 4-class problem
    4. Fine-tune only the last block + new head (transfer learning)
    5. Evaluate and visualize results

NOTE:
    Synthetic dataset is generated automatically so this script
    runs fully offline. Replace the data loader section with
    your own real dataset (e.g., PlantVillage from Kaggle).
"""

# ─────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────
import os
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import make_grid

from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              ConfusionMatrixDisplay)

# ─────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────
CONFIG = {
    "num_classes"   : 4,
    "class_names"   : ["Healthy", "Bacterial Blight", "Leaf Spot", "Rust"],
    "img_size"      : 224,          # ResNet expects 224×224
    "batch_size"    : 32,
    "num_epochs"    : 15,
    "lr_head"       : 1e-3,         # learning rate for new head
    "lr_backbone"   : 1e-5,         # smaller lr for unfrozen backbone layers
    "weight_decay"  : 1e-4,
    "train_samples" : 800,          # synthetic samples per class (train)
    "val_samples"   : 150,          # synthetic samples per class (val)
    "test_samples"  : 100,          # synthetic samples per class (test)
    "unfreeze_layer": "layer4",     # unfreeze from this ResNet block onward
    "dropout_rate"  : 0.4,
    "output_dir"    : "outputs",
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ─────────────────────────────────────────────────────────
# 3. SYNTHETIC DATASET
#    (Replace this section with a real ImageFolder or
#     custom Dataset pointing to your image directory)
# ─────────────────────────────────────────────────────────
class SyntheticLeafDataset(Dataset):
    """
    Generates synthetic leaf-like images on the fly.
    Each disease class has a distinct colour signature so
    the model can actually learn something meaningful.

    TO USE YOUR OWN DATA:
        Replace this class with:
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(root="data/train", transform=train_transform)
    """
    CLASS_COLORS = {
        0: ([0.2, 0.6, 0.2], 0.05),   # Healthy        – green, low noise
        1: ([0.6, 0.5, 0.2], 0.15),   # Bacterial Blight – yellow-brown, noisy
        2: ([0.4, 0.3, 0.1], 0.20),   # Leaf Spot       – dark brown, high noise
        3: ([0.7, 0.3, 0.1], 0.18),   # Rust            – orange-red, noisy
    }

    def __init__(self, num_per_class: int, transform=None):
        self.transform = transform
        self.samples = []  # (label, mean_color, noise_std)
        for label, (color, noise) in self.CLASS_COLORS.items():
            for _ in range(num_per_class):
                self.samples.append((label, color, noise))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, color, noise = self.samples[idx]
        # Build a 224×224 RGB image
        img = np.ones((224, 224, 3), dtype=np.float32)
        for c in range(3):
            img[:, :, c] = color[c]
        img += np.random.normal(0, noise, img.shape).astype(np.float32)
        # Add an oval "leaf" mask for realism
        cy, cx = 112, 112
        Y, X = np.ogrid[:224, :224]
        mask = ((X - cx) / 90) ** 2 + ((Y - cy) / 110) ** 2 <= 1
        background = np.random.uniform(0.85, 0.95, (224, 224, 3)).astype(np.float32)
        img = np.where(mask[:, :, None], img, background)
        img = np.clip(img, 0, 1)
        # Convert to PIL-like tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # C×H×W
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label


# ─────────────────────────────────────────────────────────
# 4. DATA TRANSFORMS & LOADERS
# ─────────────────────────────────────────────────────────

# ImageNet normalisation (required for pre-trained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training: aggressive augmentation to prevent over-fitting
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.1),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation / Test: only normalize
val_transform = transforms.Compose([
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = SyntheticLeafDataset(CONFIG["train_samples"], train_transform)
val_dataset   = SyntheticLeafDataset(CONFIG["val_samples"],   val_transform)
test_dataset  = SyntheticLeafDataset(CONFIG["test_samples"],  val_transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"],
                          shuffle=False, num_workers=0, pin_memory=False)

print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# ─────────────────────────────────────────────────────────
# 5. MODEL ARCHITECTURE
#    Modify Pre-Trained ResNet50
# ─────────────────────────────────────────────────────────

def build_model(num_classes: int,
                unfreeze_from: str = "layer4",
                dropout_rate: float = 0.4) -> nn.Module:
    """
    Load ResNet50 pre-trained on ImageNet and modify it for
    our plant disease task.

    Freezing strategy
    -----------------
    - conv1, bn1, layer1, layer2, layer3 → FROZEN  (low-level features)
    - layer4                              → UNFROZEN (high-level, task-specific)
    - fc (new head)                       → UNFROZEN

    Architecture of the new classification head
    --------------------------------------------
    Global Average Pool → Flatten
    → Linear(2048, 512) → BatchNorm → ReLU → Dropout(0.4)
    → Linear(512, 128)  → BatchNorm → ReLU → Dropout(0.2)
    → Linear(128, num_classes)
    """
    # 5.1 Load backbone
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model   = models.resnet50(weights=weights)
    print(f"[INFO] Loaded ResNet50 (ImageNet pretrained)")

    # 5.2 Freeze ALL layers first
    for param in model.parameters():
        param.requires_grad = False

    # 5.3 Selectively unfreeze from `unfreeze_from` onward
    unfreeze = False
    for name, child in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in child.parameters():
                param.requires_grad = True
            print(f"  ✓ Unfrozen: {name}")

    # 5.4 Replace the classification head
    in_features = model.fc.in_features   # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),

        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate / 2),

        nn.Linear(128, num_classes),
    )
    print(f"  ✓ New head: 2048 → 512 → 128 → {num_classes}")

    # Count trainable parameters
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters — Total: {total:,} | Trainable: {train_:,} "
          f"({100*train_/total:.1f}%)")
    return model


model = build_model(
    num_classes   = CONFIG["num_classes"],
    unfreeze_from = CONFIG["unfreeze_layer"],
    dropout_rate  = CONFIG["dropout_rate"],
).to(DEVICE)


# ─────────────────────────────────────────────────────────
# 6. LOSS, OPTIMIZER & SCHEDULER
# ─────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Differential learning rates: backbone gets 100× smaller lr
backbone_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "fc" not in n]
head_params     = [p for n, p in model.named_parameters()
                   if p.requires_grad and "fc"     in n]

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
    {"params": head_params,     "lr": CONFIG["lr_head"]},
], weight_decay=CONFIG["weight_decay"])

scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-7)


# ─────────────────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    """Run one epoch. If optimizer is None → eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if is_train:
                loss.backward()
                # Gradient clipping to prevent instability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds        = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
best_acc   = 0.0
best_state = None
start_time = time.time()

print("\n" + "="*60)
print("  TRAINING")
print("="*60)

for epoch in range(1, CONFIG["num_epochs"] + 1):
    t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
    v_loss, v_acc = run_epoch(model, val_loader,   criterion, None,      "val")
    scheduler.step()

    history["train_loss"].append(t_loss)
    history["train_acc"].append(t_acc)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(v_acc)

    # Save best model
    if v_acc > best_acc:
        best_acc   = v_acc
        best_state = copy.deepcopy(model.state_dict())
        tag = " ← best"
    else:
        tag = ""

    print(f"Epoch [{epoch:02d}/{CONFIG['num_epochs']}]  "
          f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  |  "
          f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}{tag}")

elapsed = time.time() - start_time
print(f"\n[INFO] Training complete in {elapsed/60:.1f} min | Best Val Acc: {best_acc:.4f}")

# Restore best weights
model.load_state_dict(best_state)
torch.save(best_state, os.path.join(CONFIG["output_dir"], "best_model.pth"))
print("[INFO] Best model saved to outputs/best_model.pth")


# ─────────────────────────────────────────────────────────
# 8. EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  TEST SET EVALUATION")
print("="*60)

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print(classification_report(all_labels, all_preds,
                             target_names=CONFIG["class_names"]))


# ─────────────────────────────────────────────────────────
# 9. VISUALISATIONS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle("Plant Disease Detection – Fine-Tuned ResNet50", fontsize=15, fontweight="bold")

# 9.1 Loss curves
ax = axes[0]
ax.plot(history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
ax.plot(history["val_loss"],   label="Val Loss",   color="#3498db", linewidth=2, linestyle="--")
ax.set_title("Loss Curve"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(alpha=0.3)

# 9.2 Accuracy curves
ax = axes[1]
ax.plot(history["train_acc"], label="Train Acc", color="#e74c3c", linewidth=2)
ax.plot(history["val_acc"],   label="Val Acc",   color="#3498db", linewidth=2, linestyle="--")
ax.axhline(y=best_acc, color="grey", linestyle=":", label=f"Best Val={best_acc:.2%}")
ax.set_title("Accuracy Curve"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# 9.3 Confusion matrix
ax = axes[2]
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=CONFIG["class_names"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

plt.tight_layout()
plot_path = os.path.join(CONFIG["output_dir"], "training_results.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"[INFO] Training plot saved → {plot_path}")
plt.show()


# ─────────────────────────────────────────────────────────
# 10. ARCHITECTURE SUMMARY (frozen vs trainable)
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  LAYER SUMMARY (frozen / trainable)")
print("="*60)
for name, child in model.named_children():
    params       = list(child.parameters())
    n_total      = sum(p.numel() for p in params)
    n_trainable  = sum(p.numel() for p in params if p.requires_grad)
    status       = "TRAINABLE" if n_trainable > 0 else "frozen"
    print(f"  {name:<12} {status:<12} params={n_total:>8,}")


# ─────────────────────────────────────────────────────────
# 11. SAMPLE PREDICTIONS VISUALISATION
# ─────────────────────────────────────────────────────────

# Grab one batch from the test loader
model.eval()
sample_images, sample_labels = next(iter(test_loader))
with torch.no_grad():
    sample_preds = model(sample_images.to(DEVICE)).argmax(dim=1).cpu()

# Denormalize for display
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
display_imgs = sample_images[:8] * std + mean
display_imgs = display_imgs.clamp(0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Sample Predictions (Green border = Correct, Red = Wrong)",
             fontsize=13, fontweight="bold")

for i, ax in enumerate(axes.flat):
    img = display_imgs[i].permute(1, 2, 0).numpy()
    ax.imshow(img)
    pred  = sample_preds[i].item()
    truth = sample_labels[i].item()
    color = "#2ecc71" if pred == truth else "#e74c3c"
    ax.set_title(f"Pred: {CONFIG['class_names'][pred]}\n"
                 f"True: {CONFIG['class_names'][truth]}",
                 color=color, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
pred_path = os.path.join(CONFIG["output_dir"], "sample_predictions.png")
plt.savefig(pred_path, dpi=150, bbox_inches="tight")
print(f"[INFO] Predictions plot saved → {pred_path}")
plt.show()

print("\n✅ Case study complete! Check the 'outputs/' folder for saved model & plots.")


# ─────────────────────────────────────────────────────────
# 12. HOW TO USE YOUR OWN REAL DATASET
# ─────────────────────────────────────────────────────────
"""
REAL DATA INTEGRATION (PlantVillage / your own folder)
=======================================================

Organise your images like this:
    data/
      train/
        Healthy/          (*.jpg, *.png ...)
        Bacterial_Blight/
        Leaf_Spot/
        Rust/
      val/
        Healthy/ ...
      test/
        Healthy/ ...

Then replace Section 4 with:

    from torchvision.datasets import ImageFolder

    train_dataset = ImageFolder("data/train", transform=train_transform)
    val_dataset   = ImageFolder("data/val",   transform=val_transform)
    test_dataset  = ImageFolder("data/test",  transform=val_transform)

    # Update CONFIG["num_classes"] and CONFIG["class_names"] accordingly.

That's it — everything else stays the same!

Download PlantVillage from Kaggle:
    kaggle datasets download -d abdallahalidev/plantvillage-dataset
"""
