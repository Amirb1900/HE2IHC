# ============================================================
# TransPath HER2 Classification + Feature Extraction
# ============================================================

import os
import re
import math
import copy
import random
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as transforms

import timm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Paths
# ============================================================

data_dir = "/kaggle/input/resized-bci-512/512BCI_dataset"

RESULT_DIR = "/kaggle/working/transpath_results"
FEATURE_DIR = RESULT_DIR

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 3. Fixed Label Mapping
# ============================================================

LABEL_TO_IDX = {
    "0": 0,
    "1+": 1,
    "2+": 2,
    "3+": 3
}

IDX_TO_LABEL = {
    v: k for k, v in LABEL_TO_IDX.items()
}


# ============================================================
# 4. Collect Images
# ============================================================

def get_label_from_filename(filename):

    stem = os.path.splitext(filename)[0]

    match = re.search(r"_(0|1\+|2\+|3\+)$", stem)

    if match is None:
        raise ValueError(
            f"Could not determine label from filename: {filename}"
        )

    return match.group(1)


def collect_split(split_name):

    split_dir = os.path.join(
        data_dir,
        split_name,
        split_name,
        "HE"
    )

    records = []

    for filename in sorted(os.listdir(split_dir)):

        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ):
            continue

        label_name = get_label_from_filename(filename)

        records.append({
            "filename": filename,
            "path": os.path.join(split_dir, filename),
            "label": label_name,
            "label_idx": LABEL_TO_IDX[label_name],
            "split": split_name
        })

    return pd.DataFrame(records)


train_df = collect_split("train")
val_df = collect_split("val")
test_df = collect_split("test")

df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

print("Dataset summary:")
print(df.groupby(["split", "label"]).size())
print()
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")


# ============================================================
# 5. Model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


MODEL_NAME = (
    "hf_hub:1aurent/"
    "vit_small_patch16_224.transpath_mocov3"
)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=4
)

model = model.to(device)

print("\nModel loaded.")
print("Embedding dimension: 384")
print("Expected tokens: 197 (1 CLS + 196 patch tokens)")


# ============================================================
# 6. Model-specific preprocessing
# ============================================================

data_config = timm.data.resolve_model_data_config(model)

eval_tfms = timm.data.create_transform(
    **data_config,
    is_training=False
)

train_tfms = timm.data.create_transform(
    input_size=data_config["input_size"],
    interpolation=data_config["interpolation"],
    mean=data_config["mean"],
    std=data_config["std"],
    crop_pct=data_config.get("crop_pct", 1.0),
    is_training=True
)


# ============================================================
# 7. Dataset
# ============================================================

class MyDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            row["label_idx"],
            dtype=torch.long
        )

        return image, label


train_dataset = MyDataset(
    train_df,
    transform=train_tfms
)

val_dataset = MyDataset(
    val_df,
    transform=eval_tfms
)

test_dataset = MyDataset(
    test_df,
    transform=eval_tfms
)


# ============================================================
# 8. Class Imbalance
#    KEEPING BOTH:
#    WeightedRandomSampler + class-weighted FocalLoss
# ============================================================

class_counts = (
    train_df["label_idx"]
    .value_counts()
    .sort_index()
    .values
)

print("\nTraining class counts:")
for i, count in enumerate(class_counts):
    print(f"Class {IDX_TO_LABEL[i]}: {count}")


class_weights = (
    1.0 /
    torch.tensor(
        class_counts,
        dtype=torch.float32
    )
)

sample_weights = [
    class_weights[label]
    for label in train_df["label_idx"]
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ============================================================
# 9. Focal Loss
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean"
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        elif self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


total = sum(class_counts)

alpha = [
    total / class_counts[i]
    for i in range(len(class_counts))
]

alpha = torch.tensor(
    alpha,
    dtype=torch.float32
).to(device)


criterion = FocalLoss(
    alpha=alpha,
    gamma=2.0
)


# ============================================================
# 10. Optimizer + Scheduler
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

warmup_epochs = 5
total_epochs = 100


def lr_lambda(current_epoch):

    if current_epoch < warmup_epochs:

        return (
            float(current_epoch + 1)
            / float(warmup_epochs)
        )

    progress = (
        current_epoch - warmup_epochs
    ) / (
        total_epochs - warmup_epochs
    )

    return 0.5 * (
        1 + math.cos(math.pi * progress)
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_lambda
)


# ============================================================
# 11. Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        preds = outputs.argmax(dim=1)

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_acc = accuracy_score(
        all_labels,
        all_preds
    )

    return epoch_loss, epoch_acc


# ============================================================
# 12. Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            preds = outputs.argmax(dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    qwk = cohen_kappa_score(
        all_labels,
        all_preds,
        weights="quadratic"
    )

    return (
        epoch_loss,
        acc,
        balanced_acc,
        macro_f1,
        qwk
    )


# ============================================================
# 13. Training Loop
# ============================================================

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_balanced_acc": [],
    "val_macro_f1": [],
    "val_qwk": [],
    "lr": []
}


best_val_loss = float("inf")
best_model_wts = copy.deepcopy(
    model.state_dict()
)

patience = 10
patience_counter = 0


for epoch in range(total_epochs):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    (
        val_loss,
        val_acc,
        val_balanced_acc,
        val_macro_f1,
        val_qwk
    ) = validate(
        model,
        val_loader,
        criterion,
        device
    )

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_balanced_acc"].append(val_balanced_acc)
    history["val_macro_f1"].append(val_macro_f1)
    history["val_qwk"].append(val_qwk)
    history["lr"].append(current_lr)

    print(
        f"Epoch [{epoch+1:03d}/{total_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Bal Acc: {val_balanced_acc:.4f} | "
        f"Val Macro-F1: {val_macro_f1:.4f} | "
        f"Val QWK: {val_qwk:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_wts = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1

    scheduler.step()

    if patience_counter >= patience:

        print(
            f"\nEarly stopping at epoch "
            f"{epoch + 1}."
        )

        break


# ============================================================
# 14. Restore Best Model
# ============================================================

model.load_state_dict(best_model_wts)

print("\nBest model restored.")


# ============================================================
# 15. Save Training History
# ============================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

print(
    f"Training history saved to:\n"
    f"{history_path}"
)


# ============================================================
# 16. Test Evaluation
# ============================================================

def evaluate_test(
    model,
    loader,
    device
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    qwk = cohen_kappa_score(
        all_labels,
        all_preds,
        weights="quadratic"
    )

    return (
        all_labels,
        all_preds,
        acc,
        balanced_acc,
        macro_f1,
        weighted_f1,
        qwk
    )


(
    test_labels,
    test_preds,
    test_acc,
    test_balanced_acc,
    test_macro_f1,
    test_weighted_f1,
    test_qwk
) = evaluate_test(
    model,
    test_loader,
    device
)


# ============================================================
# 17. Display Final Test Metrics
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy:           {test_acc:.4f}")
print(f"Balanced Accuracy:  {test_balanced_acc:.4f}")
print(f"Macro F1:           {test_macro_f1:.4f}")
print(f"Weighted F1:        {test_weighted_f1:.4f}")
print(f"Quadratic QWK:      {test_qwk:.4f}")


# ============================================================
# 18. Save Test Metrics
# ============================================================

test_metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Weighted F1",
        "Quadratic Weighted Kappa"
    ],
    "Value": [
        test_acc,
        test_balanced_acc,
        test_macro_f1,
        test_weighted_f1,
        test_qwk
    ]
})

test_metrics_path = os.path.join(
    RESULT_DIR,
    "test_metrics.csv"
)

test_metrics.to_csv(
    test_metrics_path,
    index=False
)


# ============================================================
# 19. Classification Report
# ============================================================

report_dict = classification_report(
    test_labels,
    test_preds,
    labels=[0, 1, 2, 3],
    target_names=["0", "1+", "2+", "3+"],
    output_dict=True,
    zero_division=0
)

classification_report_df = pd.DataFrame(
    report_dict
).transpose()

classification_report_path = os.path.join(
    RESULT_DIR,
    "classification_report.csv"
)

classification_report_df.to_csv(
    classification_report_path
)


print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        test_preds,
        labels=[0, 1, 2, 3],
        target_names=["0", "1+", "2+", "3+"],
        zero_division=0
    )
)


# ============================================================
# 20. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_preds,
    labels=[0, 1, 2, 3]
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual 0",
        "Actual 1+",
        "Actual 2+",
        "Actual 3+"
    ],
    columns=[
        "Predicted 0",
        "Predicted 1+",
        "Predicted 2+",
        "Predicted 3+"
    ]
)

cm_csv_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(
    cm_csv_path
)


# ============================================================
# 21. Confusion Matrix Figure
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("TransPath Test Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    range(4),
    ["0", "1+", "2+", "3+"]
)

plt.yticks(
    range(4),
    ["0", "1+", "2+", "3+"]
)

for i in range(4):

    for j in range(4):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_png_path = os.path.join(
    RESULT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 22. Save Best Model
# ============================================================

checkpoint_path = os.path.join(
    RESULT_DIR,
    "best_transpath_model.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "label_to_idx": LABEL_TO_IDX,
        "idx_to_label": IDX_TO_LABEL,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_balanced_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_qwk": test_qwk
    },
    checkpoint_path
)

print(
    f"\nBest model saved to:\n"
    f"{checkpoint_path}"
)


# ============================================================
# 23. Feature Extraction Dataset
# ============================================================

class FeatureDataset(Dataset):

    def __init__(self, dataframe, transform):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["path"]
        ).convert("RGB")

        image = self.transform(image)

        return (
            image,
            int(row["label_idx"]),
            row["label"],
            row["filename"],
            row["path"],
            row["split"]
        )


def extract_features(
    model,
    dataframe,
    split_name,
    transform,
    device,
    batch_size=32
):

    dataset = FeatureDataset(
        dataframe,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    cls_features = []
    patch_features = []

    labels = []
    label_names = []
    filenames = []
    paths = []
    splits = []

    model.eval()

    with torch.no_grad():

        for (
            images,
            batch_labels,
            batch_label_names,
            batch_filenames,
            batch_paths,
            batch_splits
        ) in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Extract features BEFORE classification head
            # ------------------------------------------------

            tokens = model.forward_features(
                images
            )

            # Expected:
            # [B, 197, 384]
            #
            # 1 CLS token + 196 patch tokens

            if tokens.ndim != 3:

                raise RuntimeError(
                    f"Unexpected feature shape: "
                    f"{tokens.shape}"
                )

            if tokens.shape[1] == 197:

                cls_token = tokens[:, 0, :]
                patch_tokens = tokens[:, 1:, :]

            else:

                raise RuntimeError(
                    f"Unexpected number of tokens: "
                    f"{tokens.shape}"
                )

            cls_features.append(
                cls_token.cpu()
            )

            patch_features.append(
                patch_tokens.cpu()
            )

            labels.extend(
                batch_labels.numpy().tolist()
            )

            label_names.extend(
                list(batch_label_names)
            )

            filenames.extend(
                list(batch_filenames)
            )

            paths.extend(
                list(batch_paths)
            )

            splits.extend(
                list(batch_splits)
            )

    cls_features = torch.cat(
        cls_features,
        dim=0
    )

    patch_features = torch.cat(
        patch_features,
        dim=0
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    feature_dict = {

        "cls_features": cls_features,

        "patch_features": patch_features,

        "labels": labels,

        "label_names": label_names,

        "filenames": filenames,

        "paths": paths,

        "splits": splits,

        "feature_dim": cls_features.shape[-1],

        "num_patch_tokens": patch_features.shape[1],

        "patch_grid_size": (14, 14),

        "model_name": MODEL_NAME,

        "feature_source": (
            "fine-tuned TransPath backbone "
            "before classification head"
        )
    }

    print(
        f"\n{split_name} feature extraction:"
    )

    print(
        "CLS features:",
        tuple(cls_features.shape)
    )

    print(
        "Patch features:",
        tuple(patch_features.shape)
    )

    return feature_dict


# ============================================================
# 24. Extract Train / Val / Test Features
# ============================================================

train_features = extract_features(
    model=model,
    dataframe=train_df,
    split_name="train",
    transform=eval_tfms,
    device=device
)

val_features = extract_features(
    model=model,
    dataframe=val_df,
    split_name="val",
    transform=eval_tfms,
    device=device
)

test_features = extract_features(
    model=model,
    dataframe=test_df,
    split_name="test",
    transform=eval_tfms,
    device=device
)


# ============================================================
# 25. Save Split Features as .pt
# ============================================================

torch.save(
    train_features,
    os.path.join(
        FEATURE_DIR,
        "train_features.pt"
    )
)

torch.save(
    val_features,
    os.path.join(
        FEATURE_DIR,
        "val_features.pt"
    )
)

torch.save(
    test_features,
    os.path.join(
        FEATURE_DIR,
        "test_features.pt"
    )
)


# ============================================================
# 26. Save Split Features as .npz
# ============================================================

def save_npz_features(
    feature_dict,
    output_path
):

    np.savez_compressed(
        output_path,

        cls_features=
            feature_dict["cls_features"]
            .numpy(),

        patch_features=
            feature_dict["patch_features"]
            .numpy(),

        labels=
            feature_dict["labels"]
            .numpy()
    )


save_npz_features(
    train_features,
    os.path.join(
        FEATURE_DIR,
        "train_features.npz"
    )
)

save_npz_features(
    val_features,
    os.path.join(
        FEATURE_DIR,
        "val_features.npz"
    )
)

save_npz_features(
    test_features,
    os.path.join(
        FEATURE_DIR,
        "test_features.npz"
    )
)


# ============================================================
# 27. Combine All Features
# ============================================================

all_features = {

    "cls_features": torch.cat(
        [
            train_features["cls_features"],
            val_features["cls_features"],
            test_features["cls_features"]
        ],
        dim=0
    ),

    "patch_features": torch.cat(
        [
            train_features["patch_features"],
            val_features["patch_features"],
            test_features["patch_features"]
        ],
        dim=0
    ),

    "labels": torch.cat(
        [
            train_features["labels"],
            val_features["labels"],
            test_features["labels"]
        ],
        dim=0
    ),

    "label_names":
        train_features["label_names"]
        + val_features["label_names"]
        + test_features["label_names"],

    "filenames":
        train_features["filenames"]
        + val_features["filenames"]
        + test_features["filenames"],

    "paths":
        train_features["paths"]
        + val_features["paths"]
        + test_features["paths"],

    "splits":
        train_features["splits"]
        + val_features["splits"]
        + test_features["splits"],

    "feature_dim":
        train_features["feature_dim"],

    "num_patch_tokens":
        train_features["num_patch_tokens"],

    "patch_grid_size":
        (14, 14),

    "model_name":
        MODEL_NAME,

    "feature_source":
        "fine-tuned TransPath backbone before classification head"
}


# ============================================================
# 28. Add Sample Index
# ============================================================

num_samples = len(
    all_features["labels"]
)

sample_indices = np.arange(
    num_samples
)

all_features["sample_index"] = sample_indices


# ============================================================
# 29. Save All Features
# ============================================================

all_pt_path = os.path.join(
    FEATURE_DIR,
    "all_features.pt"
)

torch.save(
    all_features,
    all_pt_path
)


np.savez_compressed(
    os.path.join(
        FEATURE_DIR,
        "all_features.npz"
    ),

    cls_features=
        all_features["cls_features"].numpy(),

    patch_features=
        all_features["patch_features"].numpy(),

    labels=
        all_features["labels"].numpy(),

    sample_index=
        sample_indices
)


# ============================================================
# 30. Save Metadata
# ============================================================

metadata_df = pd.DataFrame({

    "sample_index":
        sample_indices,

    "filename":
        all_features["filenames"],

    "path":
        all_features["paths"],

    "split":
        all_features["splits"],

    "label":
        all_features["labels"].numpy(),

    "label_name":
        all_features["label_names"]
})


metadata_path = os.path.join(
    FEATURE_DIR,
    "metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# 31. README
# ============================================================

README_TEXT = f"""
TransPath HER2 Feature Extraction
=================================

Model:
{MODEL_NAME}

Backbone:
TransPath ViT-S/16

Input:
224 x 224 RGB

Embedding dimension:
384

Number of tokens:
197

Token structure:
1 CLS token + 196 patch tokens

Patch grid:
14 x 14

Saved features
--------------

cls_features:
Shape = [N, 384]

Global image-level representation.

patch_features:
Shape = [N, 196, 384]

Patch-level representation.
The 196 tokens correspond to a 14 x 14 spatial patch grid.

labels:
Integer labels:
0  -> 0
1  -> 1+
2  -> 2+
3  -> 3+

Important:
Features were extracted from the fine-tuned TransPath
backbone BEFORE the classification head.

Preprocessing during feature extraction:
Deterministic evaluation preprocessing.
No random augmentation.

Files
-----

train_features.pt
val_features.pt
test_features.pt

train_features.npz
val_features.npz
test_features.npz

all_features.pt
all_features.npz

metadata.csv

The row/sample correspondence is defined by sample_index.

The .pt files preserve metadata and are recommended for
PyTorch/LDM workflows.

The .npz files provide a compact NumPy-compatible format.

Test features should remain isolated when evaluating
downstream models to avoid information leakage.
"""


readme_path = os.path.join(
    FEATURE_DIR,
    "README.txt"
)

with open(
    readme_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        README_TEXT
    )


# ============================================================
# 32. Save Complete Results Summary
# ============================================================

summary_df = pd.DataFrame({

    "item": [
        "Train samples",
        "Validation samples",
        "Test samples",
        "Feature dimension",
        "Patch tokens",
        "Patch grid",
        "Test Accuracy",
        "Test Balanced Accuracy",
        "Test Macro F1",
        "Test Weighted F1",
        "Test QWK"
    ],

    "value": [
        len(train_df),
        len(val_df),
        len(test_df),
        all_features["feature_dim"],
        all_features["num_patch_tokens"],
        "14 x 14",
        test_acc,
        test_balanced_acc,
        test_macro_f1,
        test_weighted_f1,
        test_qwk
    ]
})

summary_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "results_summary.csv"
    ),
    index=False
)


# ============================================================
# 33. ZIP Everything
# ============================================================

zip_path = (
    "/kaggle/working/"
    "transpath_results.zip"
)

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as zipf:

    for root, dirs, files in os.walk(
        RESULT_DIR
    ):

        for file in files:

            file_path = os.path.join(
                root,
                file
            )

            arcname = os.path.relpath(
                file_path,
                RESULT_DIR
            )

            zipf.write(
                file_path,
                arcname
            )


# ============================================================
# 34. Final Output Summary
# ============================================================

print("\n")
print("=" * 70)
print("ALL RESULTS SAVED")
print("=" * 70)

print(
    f"\nResults directory:\n"
    f"{RESULT_DIR}"
)

print(
    f"\nDownloadable ZIP:\n"
    f"{zip_path}"
)

print("\nSaved files:")

for filename in sorted(
    os.listdir(RESULT_DIR)
):

    path = os.path.join(
        RESULT_DIR,
        filename
    )

    size_mb = (
        os.path.getsize(path)
        / (1024 ** 2)
    )

    print(
        f"  {filename:<35} "
        f"{size_mb:>10.2f} MB"
    )

print("\n" + "=" * 70)
print("Feature shapes")
print("=" * 70)

print(
    "Train CLS:",
    tuple(
        train_features["cls_features"].shape
    )
)

print(
    "Train Patch:",
    tuple(
        train_features["patch_features"].shape
    )
)

print(
    "Val CLS:",
    tuple(
        val_features["cls_features"].shape
    )
)

print(
    "Val Patch:",
    tuple(
        val_features["patch_features"].shape
    )
)

print(
    "Test CLS:",
    tuple(
        test_features["cls_features"].shape
    )
)

print(
    "Test Patch:",
    tuple(
        test_features["patch_features"].shape
    )
)

print("\nDone.")
