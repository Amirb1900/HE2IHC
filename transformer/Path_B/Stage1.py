
# ============================================================
# PATH B — STAGE 1
# Frozen TransPath Backbone + Trainable HER2 Classification Head
# Dataset: BCI H&E
# Classes: 0, 1+, 2+, 3+
# ============================================================

import os
import json
import random
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import timm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

SEED = 42

DATA_ROOT = Path(
    "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
)

OUT_DIR = Path(
    "/kaggle/working/transpath_path_b/stage1_frozen_backbone"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "hf_hub:1aurent/vit_small_patch16_224.transpath_mocov3"

CLASS_NAMES = ["0", "1+", "2+", "3+"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
PATIENCE = 6

HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("PATH B — STAGE 1: FROZEN BACKBONE")
print("=" * 72)
print("Device:", device)
print("Dataset:", DATA_ROOT)
print("Output:", OUT_DIR)

assert DATA_ROOT.exists(), f"Dataset root not found: {DATA_ROOT}"


# ------------------------------------------------------------
# 2. Discover split directories
# Supports either:
# root/train/HE
# or:
# root/train/train/HE
# ------------------------------------------------------------

def find_he_dir(split_name):
    candidates = [
        DATA_ROOT / split_name / split_name / "HE",
        DATA_ROOT / split_name / "HE",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    # Fallback: find an HE directory whose ancestors identify
    # the requested split.
    matches = []
    for candidate in DATA_ROOT.rglob("HE"):
        if not candidate.is_dir():
            continue

        parts = [part.lower() for part in candidate.parts]
        if split_name.lower() in parts:
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely identify {split_name}/HE under "
        f"{DATA_ROOT}. Candidates: {matches[:10]}"
    )


TRAIN_DIR = find_he_dir("train")
VAL_DIR = find_he_dir("val")

print("\nTrain H&E directory:", TRAIN_DIR)
print("Validation H&E directory:", VAL_DIR)


# ------------------------------------------------------------
# 3. Filename-to-label parsing
# Expected filename suffixes:
# ..._0.png, ..._1+.png, ..._2+.png, ..._3+.png
# ------------------------------------------------------------

def parse_label(filename):
    stem = Path(filename).stem
    match = re.search(r"_(0|1\+|2\+|3\+)$", stem)

    if match is None:
        raise ValueError(
            f"Could not parse HER2 label from filename: {filename}"
        )

    return match.group(1)


class BCIEHDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []

        image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        for path in image_paths:
            label_name = parse_label(path.name)
            self.samples.append(
                (path, LABEL_TO_ID[label_name], label_name)
            )

        print(
            f"\nLoaded {len(self.samples)} images from "
            f"{self.image_dir}"
        )

        counts = pd.Series(
            [sample[2] for sample in self.samples]
        ).value_counts().reindex(CLASS_NAMES, fill_value=0)

        print("Class distribution:")
        print(counts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_id, label_name = self.samples[idx]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_id, path.name


# ------------------------------------------------------------
# 4. Preprocessing
# Match the deterministic preprocessing used for Path A:
# 224x224, bicubic, mean/std = 0.5 per channel, crop_pct=0.9
# ------------------------------------------------------------

eval_transform = timm.data.create_transform(
    input_size=IMAGE_SIZE,
    is_training=False,
    interpolation="bicubic",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    crop_pct=0.9,
)

train_dataset = BCIEHDataset(TRAIN_DIR, transform=eval_transform)
val_dataset = BCIEHDataset(VAL_DIR, transform=eval_transform)

assert len(train_dataset) == 3396, (
    f"Expected 3396 Train images, found {len(train_dataset)}"
)
assert len(val_dataset) == 500, (
    f"Expected 500 Validation images, found {len(val_dataset)}"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    drop_last=False,
)


# ------------------------------------------------------------
# 5. Load pretrained TransPath and freeze the entire backbone
# ------------------------------------------------------------

print("\nLoading pretrained TransPath...")

backbone = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0,
)

backbone = backbone.to(device)

for param in backbone.parameters():
    param.requires_grad = False

backbone.eval()

feature_dim = getattr(backbone, "num_features", 384)
assert feature_dim == 384, (
    f"Expected feature dimension 384, got {feature_dim}"
)

print("Backbone:", MODEL_NAME)
print("Feature dimension:", feature_dim)
print("Backbone trainable parameters:",
      sum(p.numel() for p in backbone.parameters() if p.requires_grad))


# ------------------------------------------------------------
# 6. Classification head
# CLS token -> LayerNorm -> Dropout -> Linear(4)
# ------------------------------------------------------------

class HER2ClassificationHead(nn.Module):
    def __init__(self, feature_dim=384, num_classes=4):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.20),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, cls_features):
        return self.head(cls_features)


head = HER2ClassificationHead(
    feature_dim=feature_dim,
    num_classes=len(CLASS_NAMES),
).to(device)

print("Head trainable parameters:",
      sum(p.numel() for p in head.parameters() if p.requires_grad))


# ------------------------------------------------------------
# 7. Loss and optimizer
# Moderate class weights; no oversampling.
# ------------------------------------------------------------

train_labels = np.array([
    sample[1] for sample in train_dataset.samples
])

class_counts = np.bincount(
    train_labels, minlength=len(CLASS_NAMES)
).astype(np.float64)

# Square-root inverse-frequency weighting.
class_weights_np = np.sqrt(
    class_counts.sum() / (len(CLASS_NAMES) * class_counts)
)

# Normalize average weight to 1.
class_weights_np /= class_weights_np.mean()

class_weights = torch.tensor(
    class_weights_np,
    dtype=torch.float32,
    device=device,
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    head.parameters(),
    lr=HEAD_LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

print("\nClass weights:")
for name, weight in zip(CLASS_NAMES, class_weights_np):
    print(f"  {name}: {weight:.4f}")


# ------------------------------------------------------------
# 8. Extract CLS features from frozen backbone
# timm ViT forward_features output is expected to be [B,197,384].
# The first token is CLS; remaining 196 tokens are patches.
# ------------------------------------------------------------

@torch.no_grad()
def get_cls_features(images):
    features = backbone.forward_features(images)

    if isinstance(features, (tuple, list)):
        features = features[0]

    if features.ndim == 3:
        # [B, 197, 384] -> [B, 384]
        cls_features = features[:, 0, :]
    elif features.ndim == 2:
        # Some model implementations return pooled features.
        cls_features = features
    else:
        raise RuntimeError(
            f"Unexpected backbone output shape: {tuple(features.shape)}"
        )

    if cls_features.shape[-1] != feature_dim:
        raise RuntimeError(
            f"Unexpected CLS dimension: {cls_features.shape[-1]}"
        )

    return cls_features


# ------------------------------------------------------------
# 9. Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def evaluate():
    backbone.eval()
    head.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_filenames = []

    for images, labels, filenames in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        cls_features = get_cls_features(images)
        logits = head(cls_features)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())
        all_filenames.extend(list(filenames))

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    return metrics, y_true, y_pred, y_prob, all_filenames


# ------------------------------------------------------------
# 10. Training loop
# Only the head receives gradients.
# ------------------------------------------------------------

best_macro_f1 = -1.0
best_epoch = 0
best_head_state = None
epochs_without_improvement = 0
history = []

print("\n" + "=" * 72)
print("START TRAINING")
print("=" * 72)

for epoch in range(1, NUM_EPOCHS + 1):
    backbone.eval()
    head.train()

    running_loss = 0.0
    num_seen = 0

    for images, labels, _ in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Frozen backbone: no gradient graph is created.
        with torch.no_grad():
            cls_features = get_cls_features(images)

        optimizer.zero_grad(set_to_none=True)

        logits = head(cls_features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        num_seen += batch_size

    train_loss = running_loss / num_seen

    val_metrics, _, _, _, _ = evaluate()
    scheduler.step(val_metrics["macro_f1"])

    current_lr = optimizer.param_groups[0]["lr"]

    record = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_weighted_f1": float(val_metrics["weighted_f1"]),
        "learning_rate": float(current_lr),
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"loss={train_loss:.4f} | "
        f"val_acc={val_metrics['accuracy']:.4f} | "
        f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
        f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
        f"lr={current_lr:.2e}"
    )

    if val_metrics["macro_f1"] > best_macro_f1 + 1e-5:
        best_macro_f1 = float(val_metrics["macro_f1"])
        best_epoch = epoch
        best_head_state = copy.deepcopy(head.state_dict())
        epochs_without_improvement = 0

        torch.save(
            {
                "backbone_name": MODEL_NAME,
                "head_state_dict": best_head_state,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1,
                "class_names": CLASS_NAMES,
                "feature_dim": feature_dim,
                "image_size": IMAGE_SIZE,
                "backbone_frozen": True,
            },
            OUT_DIR / "best_stage1_checkpoint.pt",
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break


# ------------------------------------------------------------
# 11. Evaluate best checkpoint
# ------------------------------------------------------------

if best_head_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

head.load_state_dict(best_head_state)

final_metrics, y_true, y_pred, y_prob, filenames = evaluate()

print("\n" + "=" * 72)
print("BEST STAGE 1 CHECKPOINT")
print("=" * 72)
print("Best epoch:", best_epoch)
print(f"Accuracy:          {final_metrics['accuracy']:.4f}")
print(f"Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
print(f"Macro F1:          {final_metrics['macro_f1']:.4f}")
print(f"Weighted F1:       {final_metrics['weighted_f1']:.4f}")

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
)

print("Confusion matrix:")
print(cm_df)


# ------------------------------------------------------------
# 12. Save metrics, predictions, history
# ------------------------------------------------------------

metadata = {
    "experiment": "Path B Stage 1 — Frozen TransPath + HER2 Head",
    "backbone": MODEL_NAME,
    "backbone_frozen": True,
    "trainable_component": "classification_head_only",
    "train_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "best_epoch": int(best_epoch),
    "selection_metric": "validation_macro_f1",
    "class_names": CLASS_NAMES,
    "class_weights": class_weights_np.tolist(),
    **{k: float(v) for k, v in final_metrics.items()},
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

pd.DataFrame(history).to_csv(
    OUT_DIR / "training_history.csv", index=False
)

cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

report_dict = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

per_class_rows = []
for cls in CLASS_NAMES:
    row = report_dict[cls]
    per_class_rows.append({
        "class": cls,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1-score": row["f1-score"],
        "support": int(row["support"]),
    })

pd.DataFrame(per_class_rows).to_csv(
    OUT_DIR / "per_class_metrics.csv", index=False
)

pred_df = pd.DataFrame({
    "filename": filenames,
    "true_label": [CLASS_NAMES[i] for i in y_true],
    "predicted_label": [CLASS_NAMES[i] for i in y_pred],
    "correct": y_true == y_pred,
})

for i, cls in enumerate(CLASS_NAMES):
    pred_df[f"prob_{cls}"] = y_prob[:, i]

pred_df.to_csv(
    OUT_DIR / "validation_predictions.csv", index=False
)

print("\nSaved outputs:")
for path in sorted(OUT_DIR.iterdir()):
    print(" -", path)

print("\nStage 1 completed. Test set was not used.")
