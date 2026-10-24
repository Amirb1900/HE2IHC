# ============================================================
# CTransPath — Path B: End-to-End Fine-Tuning
# Dataset: BCI HER2 H&E images
# Classes: 0, 1+, 2+, 3+
# Environment: Kaggle / PyTorch / CUDA
# ============================================================

import os
import sys
import json
import time
import random
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

DATA_ROOT = Path(
    "/kaggle/input/datasets/amirb1900/"
    "resized-bci-512/512BCI_dataset"
)

OUTPUT_ROOT = Path(
    "/kaggle/working/ctranspath_path_b"
)

RESULTS_DIR = OUTPUT_ROOT / "results"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_ctranspath_path_b.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_ctranspath_path_b.pth"

HISTORY_PATH = RESULTS_DIR / "training_history.csv"
METRICS_PATH = RESULTS_DIR / "test_metrics.json"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

CLASS_NAMES = ["0", "1+", "2+", "3+"]

LABEL_TO_ID = {
    "0": 0,
    "1+": 1,
    "2+": 2,
    "3+": 3,
}

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_EPOCHS = 25
PATIENCE = 7

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# Keep the same unweighted CrossEntropyLoss protocol.
# No class weights or oversampling are applied.
USE_CLASS_WEIGHTS = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Device:", DEVICE)
print("Output directory:", OUTPUT_ROOT)
print("Results directory:", RESULTS_DIR)


# ============================================================
# 3. CHECK DATASET STRUCTURE
# ============================================================

TRAIN_HE_DIR = DATA_ROOT / "train" / "train" / "HE"
VAL_HE_DIR = DATA_ROOT / "val" / "val" / "HE"
TEST_HE_DIR = DATA_ROOT / "test" / "test" / "HE"

for directory in [TRAIN_HE_DIR, VAL_HE_DIR, TEST_HE_DIR]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

print("\n" + "=" * 70)
print("DATASET DIRECTORIES")
print("=" * 70)

print("Train:", TRAIN_HE_DIR)
print("Validation:", VAL_HE_DIR)
print("Test:", TEST_HE_DIR)


# ============================================================
# 4. READ FILENAMES AND LABELS
# ============================================================

def collect_images(image_dir):
    """
    Extract labels from the suffix after the final underscore.

    Examples:
        00003_train_1+.png -> 1+
        sample_2+.png      -> 2+
        sample_0.png       -> 0

    This parser does not assume that the filename contains
    only one underscore.
    """

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    records = []
    invalid_files = []

    for path in image_paths:
        label_name = path.stem.rsplit("_", 1)[-1]

        if label_name not in LABEL_TO_ID:
            invalid_files.append(
                (path.name, label_name)
            )
            continue

        records.append({
            "path": str(path),
            "filename": path.name,
            "label_name": label_name,
            "label": LABEL_TO_ID[label_name],
        })

    if invalid_files:
        examples = invalid_files[:10]

        raise ValueError(
            f"Could not parse labels for "
            f"{len(invalid_files)} files in {image_dir}.\n"
            f"Examples: {examples}"
        )

    if not records:
        raise RuntimeError(
            f"No valid PNG images found in {image_dir}"
        )

    return pd.DataFrame(records)


train_df = collect_images(TRAIN_HE_DIR)
val_df = collect_images(VAL_HE_DIR)
test_df = collect_images(TEST_HE_DIR)


def print_split_summary(name, df):
    counts = (
        df["label_name"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
    )

    print(f"\n{name}: {len(df)} images")
    print("Class counts:", counts.to_dict())


print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print_split_summary("Train", train_df)
print_split_summary("Validation", val_df)
print_split_summary("Test", test_df)

assert set(train_df["label"].unique()).issubset(
    set(range(NUM_CLASSES))
)

assert set(val_df["label"].unique()).issubset(
    set(range(NUM_CLASSES))
)

assert set(test_df["label"].unique()).issubset(
    set(range(NUM_CLASSES))
)

# Detect accidental filename overlap between splits.
train_paths = set(train_df["path"])
val_paths = set(val_df["path"])
test_paths = set(test_df["path"])

assert train_paths.isdisjoint(val_paths), (
    "Train and validation contain overlapping paths."
)

assert train_paths.isdisjoint(test_paths), (
    "Train and test contain overlapping paths."
)

assert val_paths.isdisjoint(test_paths), (
    "Validation and test contain overlapping paths."
)

print("\nDataset count and split checks passed.")


# ============================================================
# 5. IMAGE TRANSFORMS
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# 6. PYTORCH DATASET
# ============================================================

class BCIDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = row["path"]
        label = int(row["label"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


train_dataset = BCIDataset(
    train_df,
    transform=train_transform,
)

val_dataset = BCIDataset(
    val_df,
    transform=eval_transform,
)

test_dataset = BCIDataset(
    test_df,
    transform=eval_transform,
)


# ============================================================
# 7. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
    persistent_workers=NUM_WORKERS > 0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
    persistent_workers=NUM_WORKERS > 0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
    persistent_workers=NUM_WORKERS > 0,
)


# Verify image and label shapes before loading the model.
sample_images, sample_labels = next(iter(train_loader))

print("\n" + "=" * 70)
print("BATCH CHECK")
print("=" * 70)

print("Batch image shape:", tuple(sample_images.shape))
print("Batch label shape:", tuple(sample_labels.shape))
print("Batch label values:", sample_labels.tolist())

assert sample_images.ndim == 4
assert sample_images.shape[1] == 3
assert sample_images.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE)


# ============================================================
# 8. CTransPath CUSTOM CONVOLUTIONAL STEM
# ============================================================

# This implementation follows the CTransPath ConvStem design.
# It accepts additional keyword arguments used by newer timm
# versions and returns NHWC features expected by SwinTransformer.

try:
    import timm
    from timm.layers import to_2tuple
except ImportError as exc:
    raise ImportError(
        "Could not import timm or timm.layers.to_2tuple. "
        "Check the Kaggle environment before proceeding."
    ) from exc


print("\n" + "=" * 70)
print("TIMM VERSION")
print("=" * 70)

print("timm:", timm.__version__)


class ConvStem(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        **kwargs,
    ):
        super().__init__()

        if patch_size != 4:
            raise ValueError(
                "CTransPath ConvStem requires patch_size=4."
            )

        if embed_dim % 8 != 0:
            raise ValueError(
                "embed_dim must be divisible by 8."
            )

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        self.num_patches = (
            self.grid_size[0] * self.grid_size[1]
        )

        stem_layers = []

        input_dim = in_chans
        output_dim = embed_dim // 8

        for _ in range(2):
            stem_layers.extend([
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ])

            input_dim = output_dim
            output_dim *= 2

        stem_layers.append(
            nn.Conv2d(
                input_dim,
                embed_dim,
                kernel_size=1,
            )
        )

        self.proj = nn.Sequential(*stem_layers)

        self.norm = (
            norm_layer(embed_dim)
            if norm_layer is not None
            else nn.Identity()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        if (
            height != self.img_size[0]
            or width != self.img_size[1]
        ):
            raise ValueError(
                f"Expected image size {self.img_size}, "
                f"received {(height, width)}."
            )

        x = self.proj(x)

        # BCHW -> BHWC, as expected by SwinTransformer.
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        return x


# ============================================================
# 9. LOAD PRETRAINED CTransPath BACKBONE
# ============================================================

# This is the timm-compatible Hugging Face model.
# It is not the original Google Drive checkpoint.
#
# The model is loaded with pretrained=True BEFORE replacing
# its original classification head.

MODEL_ID = (
    "hf-hub:1aurent/"
    "swin_tiny_patch4_window7_224.CTransPath"
)

print("\n" + "=" * 70)
print("LOADING PRETRAINED CTransPath")
print("=" * 70)

print("Model ID:", MODEL_ID)
print("Downloading/loading pretrained weights if needed...")

try:
    backbone = timm.create_model(
        MODEL_ID,
        pretrained=True,
        embed_layer=ConvStem,
    )
except Exception as exc:
    raise RuntimeError(
        "CTransPath could not be loaded from Hugging Face. "
        "Check that Kaggle Internet is enabled and inspect "
        "the complete underlying exception. Do not start "
        "training with randomly initialized weights."
    ) from exc


# Remove the original classification head only after the
# pretrained model has loaded successfully.
if not hasattr(backbone, "head"):
    raise AttributeError(
        "The loaded model has no 'head' attribute. "
        "Inspect the model architecture before continuing."
    )

backbone.head = nn.Identity()

if not hasattr(backbone, "num_features"):
    raise AttributeError(
        "The model does not expose num_features."
    )

FEATURE_DIM = int(backbone.num_features)

print("Backbone loaded successfully.")
print("Feature dimension:", FEATURE_DIM)


# ============================================================
# 10. CLASSIFICATION MODEL
# ============================================================

class CTransPathClassifier(nn.Module):

    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()

        self.backbone = backbone

        # Swin feature maps: [B, H, W, C]
        # Global average pooling over spatial dimensions.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

    def forward(self, x):
        features = self.backbone(x)

        if features.ndim == 4:
            # NHWC -> NCHW
            features = features.permute(0, 3, 1, 2)

            # [B, C, H, W] -> [B, C, 1, 1]
            features = self.pool(features)

            # [B, C, 1, 1] -> [B, C]
            features = torch.flatten(features, start_dim=1)

        elif features.ndim != 2:
            raise RuntimeError(
                f"Unexpected backbone output shape: "
                f"{tuple(features.shape)}"
            )

        logits = self.classifier(features)

        return logits


model = CTransPathClassifier(
    backbone=backbone,
    feature_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
).to(DEVICE)


# ============================================================
# 11. VERIFY FORWARD PASS AND TRAINABLE PARAMETERS
# ============================================================

model.eval()

with torch.no_grad():
    dummy_input = torch.zeros(
        2, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=DEVICE,
    )

    dummy_logits = model(dummy_input)

assert tuple(dummy_logits.shape) == (2, NUM_CLASSES), (
    f"Unexpected output shape: {tuple(dummy_logits.shape)}"
)

model.train()

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\n" + "=" * 70)
print("MODEL CHECK")
print("=" * 70)

print("Output shape:", tuple(dummy_logits.shape))
print("Trainable parameters:", f"{trainable_params:,}")
print("Total parameters:", f"{total_params:,}")

assert trainable_params > 0


# ============================================================
# 12. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP,
)


# ============================================================
# 13. TRAINING AND EVALUATION FUNCTIONS
# ============================================================

def run_epoch(model, loader, optimizer=None):
    """
    If optimizer is provided, perform one training epoch.
    Otherwise, evaluate without gradient updates.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, targets in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=USE_AMP,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size

        predictions = torch.argmax(
            logits.detach(),
            dim=1,
        )

        all_targets.extend(
            targets.detach().cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    mean_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_targets,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(
        all_targets,
        all_predictions,
    )

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "targets": all_targets,
        "predictions": all_predictions,
    }


# ============================================================
# 14. TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

print("Epochs:", NUM_EPOCHS)
print("Backbone LR:", BACKBONE_LR)
print("Classifier LR:", CLASSIFIER_LR)
print("Batch size:", BATCH_SIZE)
print("Loss: unweighted CrossEntropyLoss")
print("Checkpoint criterion: Validation Macro F1")

best_val_macro_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

history = []

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):

    epoch_start = time.time()

    train_metrics = run_epoch(
        model,
        train_loader,
        optimizer=optimizer,
    )

    val_metrics = run_epoch(
        model,
        val_loader,
        optimizer=None,
    )

    scheduler.step(val_metrics["macro_f1"])

    backbone_lr_now = optimizer.param_groups[0]["lr"]
    classifier_lr_now = optimizer.param_groups[1]["lr"]

    epoch_duration = time.time() - epoch_start

    row = {
        "epoch": epoch,

        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": (
            train_metrics["balanced_accuracy"]
        ),
        "train_macro_f1": train_metrics["macro_f1"],
        "train_weighted_f1": train_metrics["weighted_f1"],

        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": (
            val_metrics["balanced_accuracy"]
        ),
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],

        "backbone_lr": backbone_lr_now,
        "classifier_lr": classifier_lr_now,
        "epoch_seconds": epoch_duration,
    }

    history.append(row)

    pd.DataFrame(history).to_csv(
        HISTORY_PATH,
        index=False,
    )

    print(
        f"\nEpoch [{epoch:02d}/{NUM_EPOCHS}] "
        f"| {epoch_duration:.1f}s"
    )

    print(
        f"Train: loss={train_metrics['loss']:.4f}, "
        f"acc={train_metrics['accuracy']:.4f}, "
        f"macro-F1={train_metrics['macro_f1']:.4f}"
    )

    print(
        f"Val:   loss={val_metrics['loss']:.4f}, "
        f"acc={val_metrics['accuracy']:.4f}, "
        f"balanced-acc={val_metrics['balanced_accuracy']:.4f}, "
        f"macro-F1={val_metrics['macro_f1']:.4f}, "
        f"weighted-F1={val_metrics['weighted_f1']:.4f}"
    )

    print(
        f"LR: backbone={backbone_lr_now:.2e}, "
        f"classifier={classifier_lr_now:.2e}"
    )

    # Always save the latest training state.
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "class_names": CLASS_NAMES,
            "model_id": MODEL_ID,
            "seed": SEED,
        },
        LAST_MODEL_PATH,
    )

    # Select the best checkpoint using validation macro-F1.
    if val_metrics["macro_f1"] > best_val_macro_f1:

        best_val_macro_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "class_names": CLASS_NAMES,
                "model_id": MODEL_ID,
                "feature_dim": FEATURE_DIM,
                "seed": SEED,
            },
            BEST_MODEL_PATH,
        )

        print(
            f"Best checkpoint updated: "
            f"val_macro_f1={best_val_macro_f1:.4f}"
        )

    else:
        epochs_without_improvement += 1

        print(
            "No validation Macro F1 improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


training_duration = time.time() - start_time

print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print("Best epoch:", best_epoch)
print("Best validation Macro F1:", f"{best_val_macro_f1:.4f}")
print("Training duration (minutes):", f"{training_duration / 60:.2f}")
print("Best checkpoint:", BEST_MODEL_PATH)


# ============================================================
# 15. LOAD THE BEST VALIDATION CHECKPOINT
# ============================================================

if not BEST_MODEL_PATH.is_file():
    raise FileNotFoundError(
        "Best model checkpoint was not created."
    )

try:
    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
except TypeError:
    # Compatibility fallback for PyTorch versions that do not
    # expose the weights_only argument.
    best_checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

model.load_state_dict(
    best_checkpoint["model_state_dict"],
    strict=True,
)

print("\nLoaded best checkpoint from epoch:", best_checkpoint["epoch"])
print(
    "Checkpoint validation Macro F1:",
    f"{best_checkpoint['best_val_macro_f1']:.4f}",
)


# ============================================================
# 16. FINAL TEST EVALUATION
# ============================================================

# Test is evaluated only after checkpoint selection.
# Do not change hyperparameters based on these results.

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_metrics = run_epoch(
    model,
    test_loader,
    optimizer=None,
)

y_true = np.asarray(test_metrics["targets"])
y_pred = np.asarray(test_metrics["predictions"])

test_accuracy = accuracy_score(
    y_true,
    y_pred,
)

test_balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred,
)

test_macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0,
)

test_weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    average="weighted",
    zero_division=0,
)

test_macro_precision = precision_score(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0,
)

test_macro_recall = recall_score(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    average="macro",
    zero_division=0,
)

print(f"Test Loss:              {test_metrics['loss']:.4f}")
print(f"Test Accuracy:          {test_accuracy:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
print(f"Test Macro Precision:   {test_macro_precision:.4f}")
print(f"Test Macro Recall:      {test_macro_recall:.4f}")
print(f"Test Macro F1:          {test_macro_f1:.4f}")
print(f"Test Weighted F1:       {test_weighted_f1:.4f}")


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:
    file.write("CTransPath — Path B\n")
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(
        f"Best validation Macro F1: "
        f"{best_val_macro_f1:.6f}\n\n"
    )

    file.write(report)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    values_format="d",
    colorbar=False,
)

ax.set_title("CTransPath — Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

fig.tight_layout()

fig.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

probabilities = []

model.eval()

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):
            logits = model(images)

        probs = torch.softmax(logits.float(), dim=1)

        probabilities.extend(
            probs.cpu().numpy().tolist()
        )

probabilities = np.asarray(probabilities)

prediction_df = test_df.reset_index(drop=True).copy()

prediction_df["true_label"] = [
    CLASS_NAMES[i] for i in y_true
]

prediction_df["predicted_label"] = [
    CLASS_NAMES[i] for i in y_pred
]

prediction_df["correct"] = (
    prediction_df["label"].to_numpy() == y_pred
)

for class_idx, class_name in enumerate(CLASS_NAMES):
    prediction_df[f"prob_{class_name}"] = (
        probabilities[:, class_idx]
    )

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 20. SAVE METRICS AS JSON
# ============================================================

metrics = {
    "model": "CTransPath",
    "experiment": "Path B — End-to-End Fine-Tuning",

    "pretrained_model_id": MODEL_ID,

    "dataset_root": str(DATA_ROOT),

    "num_train_images": int(len(train_df)),
    "num_validation_images": int(len(val_df)),
    "num_test_images": int(len(test_df)),

    "class_names": CLASS_NAMES,

    "best_epoch": int(best_epoch),
    "best_validation_macro_f1": float(best_val_macro_f1),

    "test_loss": float(test_metrics["loss"]),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_macro_precision": float(test_macro_precision),
    "test_macro_recall": float(test_macro_recall),
    "test_macro_f1": float(test_macro_f1),
    "test_weighted_f1": float(test_weighted_f1),

    "confusion_matrix": cm.tolist(),

    "backbone_learning_rate": BACKBONE_LR,
    "classifier_learning_rate": CLASSIFIER_LR,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "weight_decay": WEIGHT_DECAY,
    "class_weights_used": USE_CLASS_WEIGHTS,
    "seed": SEED,

    "training_duration_minutes": float(
        training_duration / 60
    ),

    "best_checkpoint": str(BEST_MODEL_PATH),
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Best epoch:              {best_epoch}")
print(f"Best validation Macro F1:{best_val_macro_f1: .4f}")
print(f"Test Accuracy:           {test_accuracy:.4f}")
print(f"Test Balanced Accuracy:  {test_balanced_accuracy:.4f}")
print(f"Test Macro F1:            {test_macro_f1:.4f}")
print(f"Test Weighted F1:         {test_weighted_f1:.4f}")

print("\nSaved files:")

for path in [
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    REPORT_PATH,
    PREDICTIONS_PATH,
    CONFUSION_MATRIX_PATH,
]:
    print(f"- {path}")

print("\nCTransPath Path B completed.")
