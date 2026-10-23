
# ============================================================
# PATH B — FINAL TEST EVALUATION
# Load best Stage 2 checkpoint; no training
# ============================================================

import re
import json
import numpy as np
import pandas as pd
import torch
import timm

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from timm.data import create_transform

# -------------------------
# 1. Paths and configuration
# -------------------------
DATA_ROOT = Path(
    "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
)

TEST_DIR = DATA_ROOT / "test" / "test" / "HE"

CHECKPOINT_PATH = Path(
    "/kaggle/working/transpath_path_b/stage2_finetuned/"
    "best_stage2_checkpoint.pt"
)

OUT_DIR = Path("/kaggle/working/transpath_path_b/test_evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["0", "1+", "2+", "3+"]
NUM_CLASSES = 4
BATCH_SIZE = 16
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Test directory:", TEST_DIR)
print("Checkpoint:", CHECKPOINT_PATH)

assert TEST_DIR.exists(), f"Test directory not found: {TEST_DIR}"
assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"


# --------------------------------------------------
# 2. Label parser
# IMPORTANT: must match the training Dataset exactly
# --------------------------------------------------
def parse_label(path):
    """
    Assumes the class label is at the end of the filename,
    e.g. sample_0.png, sample_1+.png, sample_2+.png,
    sample_3+.png.

    If the training script uses another naming convention,
    replace this function with its exact label-parsing logic.
    """
    stem = Path(path).stem
    match = re.search(r"(?:^|[_\-\s])(0|1\+|2\+|3\+)$", stem)

    if match is None:
        raise ValueError(
            f"Cannot determine label from filename: {Path(path).name}\n"
            "Use the exact label parser from "
            "path_b_transpath_finetune.py."
        )

    return CLASS_NAMES.index(match.group(1))


# -------------------------
# 3. Test dataset
# -------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

test_paths = sorted(
    p for p in TEST_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
)

assert len(test_paths) > 0, f"No images found in {TEST_DIR}"

test_labels = [parse_label(p) for p in test_paths]

print("\nTest samples:", len(test_paths))
print("Test class counts:")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {test_labels.count(idx)}")

# Expected counts from the previously inspected dataset
expected_counts = [38, 235, 446, 258]
actual_counts = [test_labels.count(i) for i in range(NUM_CLASSES)]

if actual_counts != expected_counts:
    raise ValueError(
        f"Test class counts do not match expected counts.\n"
        f"Expected: {expected_counts}\n"
        f"Actual:   {actual_counts}\n"
        "Stop here and verify the label parser before evaluating."
    )


# -------------------------
# 4. Preprocessing
# Must match training/evaluation preprocessing
# -------------------------
transform = create_transform(
    input_size=224,
    is_training=False,
    interpolation="bicubic",
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    crop_pct=0.9,
)

class BCITestDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
            image = self.transform(image)

        return image, self.labels[idx]


test_dataset = BCITestDataset(test_paths, test_labels, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)


# -------------------------
# 5. Load checkpoint
# -------------------------
# weights_only=False is used because this is your own trusted checkpoint.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

print("\nCheckpoint keys:", list(checkpoint.keys()))
print("Checkpoint epoch:", checkpoint.get("epoch", "not stored"))
print(
    "Saved validation Macro F1:",
    checkpoint.get("best_val_macro_f1",
                   checkpoint.get("val_macro_f1", "not stored"))
)


# -------------------------
# 6. Rebuild Stage 2 model
# -------------------------
backbone = timm.create_model(
    "hf_hub:1aurent/vit_small_patch16_224.transpath_mocov3",
    pretrained=False,
    num_classes=0,
)

feature_dim = backbone.num_features
assert feature_dim == 384, f"Unexpected feature dimension: {feature_dim}"

head = torch.nn.Sequential(
    torch.nn.LayerNorm(feature_dim),
    torch.nn.Dropout(0.20),
    torch.nn.Linear(feature_dim, NUM_CLASSES),
)

# Expected checkpoint format from the Stage 2 training code.
if "backbone_state_dict" not in checkpoint:
    raise KeyError(
        "backbone_state_dict not found in checkpoint. "
        "Inspect checkpoint keys and adapt loading to the training script."
    )

backbone.load_state_dict(checkpoint["backbone_state_dict"])

if "head_state_dict" in checkpoint:
    head.load_state_dict(checkpoint["head_state_dict"])
elif "classifier_state_dict" in checkpoint:
    head.load_state_dict(checkpoint["classifier_state_dict"])
else:
    raise KeyError(
        "Could not find head_state_dict or classifier_state_dict. "
        f"Available keys: {list(checkpoint.keys())}. "
        "Use the exact head key saved by the training script."
    )

backbone = backbone.to(device).eval()
head = head.to(device).eval()


# -------------------------
# 7. Inference — no training
# -------------------------
all_preds = []
all_labels = []
all_probs = []

with torch.inference_mode():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=True):
                features = backbone(images)
                logits = head(features)
        else:
            features = backbone(images)
            logits = head(features)

        probs = torch.softmax(logits.float(), dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")


y_true = np.asarray(all_labels)
y_pred = np.asarray(all_preds)
y_prob = np.asarray(all_probs)


# -------------------------
# 8. Metrics
# -------------------------
metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
}

print("\n" + "=" * 60)
print("PATH B — FINAL TEST RESULTS")
print("=" * 60)

for name, value in metrics.items():
    print(f"{name:20s}: {value:.4f} ({value * 100:.2f}%)")

print("\nClassification report:")
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)
print(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
)

print("Confusion matrix (rows=true, columns=predicted):")
print(cm)


# -------------------------
# 9. Save outputs
# -------------------------
with open(OUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open(OUT_DIR / "test_classification_report.txt", "w") as f:
    f.write(report)

pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
).to_csv(OUT_DIR / "test_confusion_matrix.csv")

np.save(OUT_DIR / "test_predictions.npy", y_pred)
np.save(OUT_DIR / "test_labels.npy", y_true)
np.save(OUT_DIR / "test_probabilities.npy", y_prob)

print("\nSaved test results to:", OUT_DIR)
print("Evaluation complete. No training was performed.")
