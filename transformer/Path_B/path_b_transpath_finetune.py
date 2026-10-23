
"""
Path B — TransPath Fine-tuning for HER2 Classification
======================================================

Stage 1: Frozen pretrained TransPath + trainable classifier head
Stage 2: Fine-tune the last Transformer blocks + classifier head
Stage 3: Extract CLS and patch features using the best Stage 2 model

Validation is used for model selection.
Test is not used for training or model selection.

Dataset labels are parsed from filename suffixes:
_0, _1+, _2+, _3+
"""

from pathlib import Path
import json
import random
import re

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import create_transform

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

DATA_ROOT = Path(
    "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
)

TRAIN_DIR = DATA_ROOT / "train" / "train" / "HE"
VAL_DIR = DATA_ROOT / "val" / "val" / "HE"

OUTPUT_DIR = Path("/kaggle/working/transpath_path_b")

STAGE1_DIR = OUTPUT_DIR / "stage1_frozen_backbone"
STAGE2_DIR = OUTPUT_DIR / "stage2_finetuned"
FEATURE_DIR = OUTPUT_DIR / "finetuned_features"

STAGE1_CKPT = STAGE1_DIR / "best_stage1_checkpoint.pt"
STAGE2_CKPT = STAGE2_DIR / "best_stage2_checkpoint.pt"

MODEL_NAME = "hf_hub:1aurent/vit_small_patch16_224.transpath_mocov3"

CLASS_NAMES = ["0", "1+", "2+", "3+"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

STAGE1_EPOCHS = 35
STAGE1_PATIENCE = 6
STAGE1_HEAD_LR = 1e-3

STAGE2_EPOCHS = 20
STAGE2_PATIENCE = 6
STAGE2_BACKBONE_LR = 1e-5
STAGE2_HEAD_LR = 1e-4
UNFREEZE_LAST_N_BLOCKS = 2

WEIGHT_DECAY = 1e-4
DROPOUT = 0.20

# False means only Train and Validation features are extracted.
# Do not enable until the model and all decisions are finalized.
EXTRACT_TEST_FEATURES = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LABEL_PATTERN = re.compile(r"_(0|1\+|2\+|3\+)$")

for directory in (STAGE1_DIR, STAGE2_DIR, FEATURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Dataset
# ============================================================

def parse_label(path):
    match = LABEL_PATTERN.search(Path(path).stem)

    if match is None:
        raise ValueError(
            f"Cannot parse HER2 label from filename: {Path(path).name}"
        )

    return CLASS_NAMES.index(match.group(1))


class BCIDataset(Dataset):
    def __init__(self, directory, transform):
        self.directory = Path(directory)
        self.transform = transform

        if not self.directory.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.directory}"
            )

        image_paths = sorted(
            p for p in self.directory.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_paths:
            raise RuntimeError(
                f"No supported images found in {self.directory}"
            )

        self.samples = [
            (path, parse_label(path))
            for path in image_paths
        ]

        self.targets = [label for _, label in self.samples]

        counts = {
            CLASS_NAMES[i]: self.targets.count(i)
            for i in range(NUM_CLASSES)
        }

        print(f"\nDataset: {self.directory}")
        print(f"Number of images: {len(self.samples)}")
        print(f"Class counts: {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        image = self.transform(image)

        return image, label, path.name


def build_transform():
    # Consistent with the previous Path A and Stage 1 experiments.
    return create_transform(
        input_size=IMAGE_SIZE,
        is_training=False,
        interpolation="bicubic",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=0.9,
    )


def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )


# ============================================================
# 4. Model
# ============================================================

class HER2Head(nn.Module):
    def __init__(self, feature_dim, num_classes=NUM_CLASSES):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(DROPOUT),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, features):
        return self.layers(features)


def build_model():
    backbone = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
    )

    feature_dim = int(getattr(backbone, "num_features", 384))

    head = HER2Head(
        feature_dim=feature_dim,
        num_classes=NUM_CLASSES,
    )

    return backbone.to(DEVICE), head.to(DEVICE)


def get_tokens(backbone, images):
    """
    For the expected ViT architecture, forward_features returns
    [batch, 197, 384]: one CLS token plus 196 patch tokens.
    """
    features = backbone.forward_features(images)

    if features.ndim not in (2, 3):
        raise RuntimeError(
            f"Unexpected TransPath output shape: {tuple(features.shape)}"
        )

    return features


def get_cls_features(backbone, images):
    features = get_tokens(backbone, images)

    if features.ndim == 3:
        return features[:, 0, :]

    return features


# ============================================================
# 5. Evaluation and reporting
# ============================================================

@torch.no_grad()
def evaluate(backbone, head, loader):
    backbone.eval()
    head.eval()

    labels_all = []
    predictions_all = []
    probabilities_all = []
    filenames_all = []

    for images, labels, filenames in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            features = get_cls_features(backbone, images)
            logits = head(features)

        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = logits.argmax(dim=1)

        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().numpy().tolist())
        filenames_all.extend(list(filenames))

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(predictions_all)
    y_prob = np.asarray(probabilities_all)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    return {
        "metrics": metrics,
        "labels": y_true,
        "predictions": y_pred,
        "probabilities": y_prob,
        "filenames": filenames_all,
    }


def save_evaluation(result, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = result["labels"]
    y_pred = result["predictions"]

    with open(
        output_dir / f"{prefix}_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result["metrics"], file, indent=2)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES],
    ).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        output_dir / f"{prefix}_classification_report.csv"
    )

    prediction_data = {
        "filename": result["filenames"],
        "true_label": [CLASS_NAMES[i] for i in y_true],
        "predicted_label": [CLASS_NAMES[i] for i in y_pred],
    }

    for i, class_name in enumerate(CLASS_NAMES):
        prediction_data[f"prob_{class_name}"] = result["probabilities"][:, i]

    pd.DataFrame(prediction_data).to_csv(
        output_dir / f"{prefix}_predictions.csv",
        index=False,
    )

    print(f"\n{prefix} metrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion matrix:\n", cm)


# ============================================================
# 6. Training loop
# ============================================================

def train_model(
    backbone,
    head,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    class_weights,
    epochs,
    patience,
    checkpoint_path,
    stage_name,
    train_backbone,
):
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Compatible with CUDA mixed-precision training.
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        backbone.train(train_backbone)
        head.train()

        total_loss = 0.0
        total_examples = 0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                if train_backbone:
                    features = get_cls_features(backbone, images)
                else:
                    with torch.no_grad():
                        features = get_cls_features(backbone, images)

                logits = head(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / max(total_examples, 1)

        val_result = evaluate(backbone, head, val_loader)
        val_metrics = val_result["metrics"]

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])

        learning_rates = [
            float(group["lr"]) for group in optimizer.param_groups
        ]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rates": json.dumps(learning_rates),
        })

        print(
            f"\n[{stage_name}] Epoch {epoch}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={learning_rates}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "stage": stage_name,
                "epoch": epoch,
                "model_name": MODEL_NAME,
                "class_names": CLASS_NAMES,
                "feature_dim": int(
                    getattr(backbone, "num_features", 384)
                ),
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict": head.state_dict(),
                "metrics": val_metrics,
                "unfreeze_last_n_blocks": (
                    UNFREEZE_LAST_N_BLOCKS if train_backbone else 0
                ),
                "config": {
                    "image_size": IMAGE_SIZE,
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                },
            }

            torch.save(checkpoint, checkpoint_path)
            print("Saved best checkpoint:", checkpoint_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        Path(checkpoint_path).parent / "training_history.csv",
        index=False,
    )

    # Restore the best validation checkpoint before returning.
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    backbone.load_state_dict(best_checkpoint["backbone_state_dict"])
    head.load_state_dict(best_checkpoint["head_state_dict"])

    print(f"\n{stage_name} best epoch: {best_epoch}")
    print(f"{stage_name} best Macro F1: {best_macro_f1:.4f}")

    return best_checkpoint


# ============================================================
# 7. Fine-tuned feature extraction
# ============================================================

@torch.no_grad()
def extract_features(backbone, dataset, output_path):
    loader = make_loader(dataset, shuffle=False)
    backbone.eval()

    cls_batches = []
    patch_batches = []
    labels_all = []
    filenames_all = []

    for images, labels, filenames in loader:
        images = images.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            tokens = get_tokens(backbone, images)

        tokens = tokens.float()

        if tokens.ndim == 3:
            cls_features = tokens[:, 0, :]
            patch_features = tokens[:, 1:, :]
            patch_batches.append(patch_features.cpu())
        else:
            cls_features = tokens

        cls_batches.append(cls_features.cpu())
        labels_all.extend(labels.tolist())
        filenames_all.extend(list(filenames))

    output = {
        "cls_features": torch.cat(cls_batches, dim=0),
        "labels": torch.tensor(labels_all, dtype=torch.long),
        "filenames": filenames_all,
        "class_names": CLASS_NAMES,
        "model_name": MODEL_NAME,
        "source_checkpoint": str(STAGE2_CKPT),
    }

    if patch_batches:
        output["patch_features"] = torch.cat(patch_batches, dim=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    print(f"\nSaved features: {output_path}")
    print("CLS shape:", tuple(output["cls_features"].shape))

    if "patch_features" in output:
        print("Patch shape:", tuple(output["patch_features"].shape))


# ============================================================
# 8. Main pipeline
# ============================================================

def main():
    set_seed()

    print("Device:", DEVICE)
    print("Mixed precision:", USE_AMP)
    print("Model:", MODEL_NAME)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    transform = build_transform()

    train_dataset = BCIDataset(TRAIN_DIR, transform)
    val_dataset = BCIDataset(VAL_DIR, transform)

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(val_dataset, shuffle=False)

    counts = np.bincount(
        np.asarray(train_dataset.targets),
        minlength=NUM_CLASSES,
    ).astype(np.float32)

    # Same square-root inverse-frequency weighting strategy.
    class_weights = np.sqrt(counts.sum() / counts)
    class_weights = class_weights / class_weights.mean()

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("\nClass weights:", class_weights.cpu().numpy())

    # --------------------------------------------------------
    # Stage 1: Frozen backbone
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("STAGE 1 — Frozen pretrained TransPath")
    print("=" * 65)

    backbone1, head1 = build_model()

    for parameter in backbone1.parameters():
        parameter.requires_grad = False

    for parameter in head1.parameters():
        parameter.requires_grad = True

    optimizer1 = torch.optim.AdamW(
        head1.parameters(),
        lr=STAGE1_HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    train_model(
        backbone=backbone1,
        head=head1,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer1,
        scheduler=scheduler1,
        class_weights=class_weights,
        epochs=STAGE1_EPOCHS,
        patience=STAGE1_PATIENCE,
        checkpoint_path=STAGE1_CKPT,
        stage_name="stage1",
        train_backbone=False,
    )

    stage1_result = evaluate(backbone1, head1, val_loader)
    save_evaluation(stage1_result, STAGE1_DIR, "stage1")

    del backbone1, head1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Stage 2: Fine-tune last Transformer blocks
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("STAGE 2 — Fine-tune final Transformer blocks")
    print("=" * 65)

    # Stage 1 did not update the frozen backbone. Reload the original
    # pretrained backbone and restore the trained Stage 1 head.
    backbone2, head2 = build_model()

    stage1_checkpoint = torch.load(
        STAGE1_CKPT,
        map_location=DEVICE,
    )
    head2.load_state_dict(
        stage1_checkpoint["head_state_dict"],
        strict=True,
    )

    for parameter in backbone2.parameters():
        parameter.requires_grad = False

    if not hasattr(backbone2, "blocks"):
        raise AttributeError(
            "TransPath does not expose .blocks; inspect the model "
            "architecture before unfreezing Transformer blocks."
        )

    if UNFREEZE_LAST_N_BLOCKS > len(backbone2.blocks):
        raise ValueError(
            f"Requested {UNFREEZE_LAST_N_BLOCKS} blocks, but the model "
            f"has {len(backbone2.blocks)} blocks."
        )

    for block in backbone2.blocks[-UNFREEZE_LAST_N_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Unfreeze final normalization layer, if present.
    if getattr(backbone2, "norm", None) is not None:
        for parameter in backbone2.norm.parameters():
            parameter.requires_grad = True

    for parameter in head2.parameters():
        parameter.requires_grad = True

    backbone_parameters = [
        p for p in backbone2.parameters() if p.requires_grad
    ]
    head_parameters = list(head2.parameters())

    print(
        "Trainable backbone parameters:",
        f"{sum(p.numel() for p in backbone_parameters):,}",
    )
    print("Unfrozen final Transformer blocks:", UNFREEZE_LAST_N_BLOCKS)

    optimizer2 = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": STAGE2_BACKBONE_LR,
            },
            {
                "params": head_parameters,
                "lr": STAGE2_HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    train_model(
        backbone=backbone2,
        head=head2,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer2,
        scheduler=scheduler2,
        class_weights=class_weights,
        epochs=STAGE2_EPOCHS,
        patience=STAGE2_PATIENCE,
        checkpoint_path=STAGE2_CKPT,
        stage_name="stage2",
        train_backbone=True,
    )

    stage2_result = evaluate(backbone2, head2, val_loader)
    save_evaluation(stage2_result, STAGE2_DIR, "stage2")

    # --------------------------------------------------------
    # Compare Stage 1 vs Stage 2
    # --------------------------------------------------------
    comparison = pd.DataFrame({
        "Stage 1 — Frozen Backbone": stage1_result["metrics"],
        "Stage 2 — Fine-tuned Backbone": stage2_result["metrics"],
    }).T

    comparison.to_csv(OUTPUT_DIR / "stage1_vs_stage2.csv")

    print("\n" + "=" * 65)
    print("STAGE 1 vs STAGE 2")
    print("=" * 65)
    print(comparison)

    # --------------------------------------------------------
    # Extract features using the selected Stage 2 backbone
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXTRACT FEATURES USING THE BEST STAGE 2 BACKBONE")
    print("=" * 65)

    extract_features(
        backbone2,
        train_dataset,
        FEATURE_DIR / "train_features.pt",
    )

    extract_features(
        backbone2,
        val_dataset,
        FEATURE_DIR / "val_features.pt",
    )

    # Test is intentionally excluded by default.
    if EXTRACT_TEST_FEATURES:
        test_dir = DATA_ROOT / "test" / "test" / "HE"

        if not test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}"
            )

        test_dataset = BCIDataset(test_dir, transform)

        extract_features(
            backbone2,
            test_dataset,
            FEATURE_DIR / "test_features.pt",
        )

    print("\nPath B completed.")
    print("Stage 1 checkpoint:", STAGE1_CKPT)
    print("Stage 2 checkpoint:", STAGE2_CKPT)
    print("Metrics comparison:", OUTPUT_DIR / "stage1_vs_stage2.csv")
    print("Feature directory:", FEATURE_DIR)
    print("Test features extracted:", EXTRACT_TEST_FEATURES)


if __name__ == "__main__":
    main()
