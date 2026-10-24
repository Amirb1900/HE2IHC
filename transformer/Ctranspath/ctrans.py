# ============================================================
# CTransPath — BCI HER2 Classification
# Independent Kaggle experiment
# Pretrained CTransPath + end-to-end fine-tuning
# ============================================================

import os
import sys
import json
import time
import random
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# 1. INSTALL / IMPORT DEPENDENCIES
# ============================================================

def install_package(package):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        check=True
    )

install_package("gdown")
install_package("scikit-learn")
install_package("seaborn")
install_package("timm")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is not available. Enable GPU in Kaggle Settings > Accelerator."
    )

DEVICE = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# 3. PATHS AND CONFIGURATION
# ============================================================

DATA_DIR = Path(
    "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
)

WORK_DIR = Path("/kaggle/working/ctranspath_path_b")
REPO_DIR = WORK_DIR / "TransPath"
CHECKPOINT_PATH = WORK_DIR / "ctranspath.pth"
OUTPUT_DIR = WORK_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUTPUT_DIR / "best_ctranspath_checkpoint.pt"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

CLASS_NAMES = ["0", "1+", "2+", "3+"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

EPOCHS = 20
PATIENCE = 5

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4

# Keep this consistent with the previous Path B protocol.
# Set True only if Path B also used class-weighted CrossEntropy.
USE_CLASS_WEIGHTS = False

# Official CTransPath checkpoint from the repository README.
CHECKPOINT_URL = (
    "https://drive.google.com/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX"
)

print("\nOutput directory:", OUTPUT_DIR)


# ============================================================
# 4. DOWNLOAD OFFICIAL CTRanSPATH REPOSITORY AND CHECKPOINT
# ============================================================

WORK_DIR.mkdir(parents=True, exist_ok=True)

if not (REPO_DIR / "ctran.py").exists():
    print("\nCloning official TransPath repository...")

    subprocess.run(
        [
            "git", "clone",
            "https://github.com/Xiyue-Wang/TransPath.git",
            str(REPO_DIR)
        ],
        check=True
    )
else:
    print("Repository already exists.")

sys.path.insert(0, str(REPO_DIR))

if not (REPO_DIR / "ctran.py").exists():
    raise FileNotFoundError(
        f"ctran.py not found in {REPO_DIR}"
    )

if not CHECKPOINT_PATH.exists():
    print("\nDownloading pretrained CTransPath checkpoint...")

    subprocess.run(
        [
            sys.executable, "-m", "gdown",
            CHECKPOINT_URL,
            "-O", str(CHECKPOINT_PATH)
        ],
        check=True
    )

if (
    not CHECKPOINT_PATH.exists()
    or CHECKPOINT_PATH.stat().st_size < 1_000_000
):
    raise RuntimeError(
        "Checkpoint download failed or the file is unexpectedly small. "
        "Check Kaggle Internet access and the official download link."
    )

print(
    "Checkpoint size:",
    round(CHECKPOINT_PATH.stat().st_size / (1024 ** 2), 2),
    "MB"
)

from ctran import ctranspath


# ============================================================
# 5. LOAD DATASET
# ============================================================

def collect_split(split):
    image_dir = DATA_DIR / split / split / "HE"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    rows = []

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    if not image_paths:
        raise RuntimeError(
            f"No PNG images found in {image_dir}"
        )

    for path in image_paths:
        # Examples:
        # 00003_train_1+.png -> 1+
        # 00010_2+.png       -> 2+
        label_name = path.stem.rsplit("_", 1)[-1]

        if label_name not in LABEL_TO_ID:
            raise ValueError(
                f"Unexpected label '{label_name}' in {path.name}"
            )

        rows.append({
            "filename": str(path),
            "image_name": path.name,
            "label_name": label_name,
            "label": LABEL_TO_ID[label_name]
        })

    return pd.DataFrame(rows)


train_df = collect_split("train")
val_df = collect_split("val")
test_df = collect_split("test")

EXPECTED_COUNTS = {
    "train": 3396,
    "val": 500,
    "test": 977
}

actual_counts = {
    "train": len(train_df),
    "val": len(val_df),
    "test": len(test_df)
}

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

for split_name, df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    print(f"\n{split_name}: {len(df)} images")

    counts = (
        df["label_name"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    print("Class counts:", counts.to_dict())

if actual_counts != EXPECTED_COUNTS:
    raise RuntimeError(
        "Dataset counts differ from the expected counts. "
        f"Expected={EXPECTED_COUNTS}, actual={actual_counts}. "
        "Check dataset paths and label parsing before continuing."
    )

print("\nDataset count checks passed.")


# ============================================================
# 6. TRANSFORMS
# ============================================================

# Official CTransPath feature-extraction preprocessing uses
# Resize(224), ToTensor(), and ImageNet normalization.
# Train uses horizontal/vertical flips as mild augmentation.

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ============================================================
# 7. DATASET AND DATALOADERS
# ============================================================

class BCIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row["filename"]) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


train_ds = BCIDataset(train_df, train_transform)
val_ds = BCIDataset(val_df, eval_transform)
test_ds = BCIDataset(test_df, eval_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

sample_images, sample_labels = next(iter(train_loader))

print("\nBatch image shape:", tuple(sample_images.shape))
print("Batch label shape:", tuple(sample_labels.shape))


# ============================================================
# 8. LOAD PRETRAINED CTRanSPATH
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRETRAINED CTRanSPATH")
print("=" * 70)

# Build the official architecture.
backbone = ctranspath()

# Remove the original classification head.
backbone.head = nn.Identity()

# Explicit weights_only=False supports checkpoints containing
# metadata in addition to tensors. Only load this trusted official file.
try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )
except TypeError:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

if not isinstance(checkpoint, dict) or "model" not in checkpoint:
    raise KeyError(
        "Expected the official checkpoint to contain a 'model' key. "
        f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
    )

# Do not silently ignore missing or unexpected weights.
load_result = backbone.load_state_dict(
    checkpoint["model"],
    strict=True
)

print("Official pretrained weights loaded.")
print(load_result)

backbone = backbone.to(DEVICE)

# Determine feature dimension using a real forward pass.
backbone.eval()

with torch.no_grad():
    dummy = torch.zeros(
        2, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=DEVICE
    )

    dummy_features = backbone(dummy)

    if dummy_features.ndim > 2:
        dummy_features = torch.flatten(
            dummy_features, start_dim=1
        )

    FEATURE_DIM = int(dummy_features.shape[1])

print("Feature tensor shape:", tuple(dummy_features.shape))
print("Feature dimension:", FEATURE_DIM)


# ============================================================
# 9. CLASSIFICATION MODEL
# ============================================================

class CTransPathHER2(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=4):
        super().__init__()

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)

        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)

        logits = self.classifier(features)

        return logits


model = CTransPathHER2(
    backbone=backbone,
    feature_dim=FEATURE_DIM,
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

# Verify the complete model output.
model.eval()

with torch.no_grad():
    dummy_logits = model(
        torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    )

assert dummy_logits.shape == (2, 4), (
    f"Unexpected output shape: {tuple(dummy_logits.shape)}"
)

print("\nModel output shape:", tuple(dummy_logits.shape))
print("Backbone fine-tuning: YES")


# ============================================================
# 10. LOSS, OPTIMIZER, AND SCHEDULER
# ============================================================

if USE_CLASS_WEIGHTS:
    train_class_counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(len(CLASS_NAMES)), fill_value=0)
        .to_numpy(dtype=np.float64)
    )

    if np.any(train_class_counts == 0):
        raise RuntimeError(
            "At least one class has no training samples."
        )

    class_weights = (
        train_class_counts.sum()
        / (len(CLASS_NAMES) * train_class_counts)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE
        )
    )

    print("Class-weighted CrossEntropy enabled.")
    print("Class weights:", class_weights.tolist())

else:
    criterion = nn.CrossEntropyLoss()
    print("Standard CrossEntropy enabled.")

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# Mixed precision for GPU memory and speed.
scaler = torch.cuda.amp.GradScaler(enabled=True)


# ============================================================
# 11. EPOCH FUNCTION
# ============================================================

def run_epoch(model, loader, training):
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.detach().item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        all_labels.extend(
            labels.detach().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy().tolist()
        )

    average_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": float(average_loss),
        "accuracy": float(
            accuracy_score(all_labels, all_predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(all_labels, all_predictions)
        ),
        "macro_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0
            )
        )
    }

    return metrics


# ============================================================
# 12. TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING CTRanSPATH FINE-TUNING")
print("=" * 70)

history = []
best_val_macro_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    train_metrics = run_epoch(
        model,
        train_loader,
        training=True
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        training=False
    )

    scheduler.step(val_metrics["macro_f1"])

    record = {
        "epoch": epoch,
        **{
            f"train_{key}": value
            for key, value in train_metrics.items()
        },
        **{
            f"val_{key}": value
            for key, value in val_metrics.items()
        }
    }

    history.append(record)

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False
    )

    elapsed_epoch = time.time() - epoch_start

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS} | "
        f"Time: {elapsed_epoch:.1f}s"
    )

    print(
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Acc: {train_metrics['accuracy']:.4f} | "
        f"Train Macro F1: {train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f} | "
        f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
    )

    if val_metrics["macro_f1"] > best_val_macro_f1:
        best_val_macro_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "class_names": CLASS_NAMES,
                "seed": SEED,
                "feature_dim": FEATURE_DIM,
                "use_class_weights": USE_CLASS_WEIGHTS
            },
            BEST_CHECKPOINT
        )

        print("  -> Best checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping: no improvement for "
            f"{PATIENCE} consecutive epochs."
        )
        break

total_training_time = time.time() - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("Best epoch:", best_epoch)
print("Best Validation Macro F1:", round(best_val_macro_f1, 4))
print("Training time (minutes):", round(total_training_time / 60, 2))
print("Checkpoint:", BEST_CHECKPOINT)


# ============================================================
# 13. LOAD BEST CHECKPOINT
# ============================================================

if not BEST_CHECKPOINT.exists():
    raise FileNotFoundError(
        "Best checkpoint was not created. Training may have failed."
    )

try:
    best_saved = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )
except TypeError:
    best_saved = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE
    )

model.load_state_dict(
    best_saved["model_state_dict"],
    strict=True
)

model.eval()

print("\nLoaded best checkpoint from epoch:", best_saved["epoch"])


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

# Test set is evaluated only after checkpoint selection.
test_true = []
test_pred = []
test_prob = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images)

        probabilities = torch.softmax(
            logits.float(),
            dim=1
        )

        predictions = probabilities.argmax(dim=1)

        test_true.extend(labels.numpy().tolist())
        test_pred.extend(predictions.cpu().numpy().tolist())
        test_prob.extend(probabilities.cpu().numpy().tolist())


# Final metrics
test_metrics = {
    "best_epoch": int(best_saved["epoch"]),
    "accuracy": float(
        accuracy_score(test_true, test_pred)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(test_true, test_pred)
    ),
    "macro_f1": float(
        f1_score(
            test_true,
            test_pred,
            average="macro",
            zero_division=0
        )
    ),
    "weighted_f1": float(
        f1_score(
            test_true,
            test_pred,
            average="weighted",
            zero_division=0
        )
    )
}

report_text = classification_report(
    test_true,
    test_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0
)

report_dict = classification_report(
    test_true,
    test_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=[0, 1, 2, 3]
)

print("\n" + "=" * 70)
print("CTRanSPATH — FINAL TEST RESULTS")
print("=" * 70)

for key, value in test_metrics.items():
    if key == "best_epoch":
        print(f"{key:22s}: {value}")
    else:
        print(
            f"{key:22s}: {value:.4f} "
            f"({value * 100:.2f}%)"
        )

print("\nClassification report:")
print(report_text)

print("Confusion matrix (rows=true, columns=predicted):")

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES]
)

print(cm_df)


# ============================================================
# 15. SAVE METRICS, REPORT, CONFUSION MATRIX, PREDICTIONS
# ============================================================

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        test_metrics,
        f,
        indent=2,
        ensure_ascii=False
    )

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

cm_df.to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

# test_loader uses shuffle=False, so prediction order corresponds
# to the order of test_df.
predictions_df = pd.DataFrame({
    "image_name": test_df["image_name"].tolist(),
    "true_label_id": test_true,
    "true_label": [
        ID_TO_LABEL[x] for x in test_true
    ],
    "predicted_label_id": test_pred,
    "predicted_label": [
        ID_TO_LABEL[x] for x in test_pred
    ],
    "correct": [
        int(t == p)
        for t, p in zip(test_true, test_pred)
    ]
})

for i, class_name in enumerate(CLASS_NAMES):
    predictions_df[f"prob_{class_name}"] = [
        float(row[i]) for row in test_prob
    ]

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 16. SAVE CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("CTransPath — Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 17. SAVE TRAINING CURVES
# ============================================================

history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_PATH, index=False)

if not history_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train"
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation"
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history_df["epoch"],
        history_df["train_macro_f1"],
        label="Train"
    )

    axes[1].plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        label="Validation"
    )

    axes[1].set_title("Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_curves.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CTRanSPATH EXPERIMENT COMPLETED")
print("=" * 70)

print("\nFinal Test metrics:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

print("\nSaved files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(
            f"{path.name:40s} "
            f"{path.stat().st_size / 1024:.1f} KB"
        )

print("\nOutput directory:", OUTPUT_DIR)
print("Best checkpoint:", BEST_CHECKPOINT)
print("CTransPath Test evaluation completed.")
