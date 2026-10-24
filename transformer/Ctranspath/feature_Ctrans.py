
# ============================================================
# CTransPath Feature Extraction for BCI H&E -> IHC / LDM
# Single-cell implementation for Kaggle
#
# Loads the fine-tuned checkpoint (best epoch 22).
# Extracts spatial backbone features and global pooled features.
# Processes train / val / test H&E images.
# Saves feature arrays and image metadata.
# ============================================================

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
)

CHECKPOINT_PATH = Path(
    "/kaggle/working/ctranspath_path_b/checkpoints/"
    "best_ctranspath_path_b.pth"
)

OUTPUT_ROOT = Path("/kaggle/working/ctranspath_features")

MODEL_ID = (
    "hf-hub:1aurent/"
    "swin_tiny_patch4_window7_224.CTransPath"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
FEATURE_DIM = 768
SEED = 42

CLASS_NAMES = ["0", "1+", "2+", "3+"]

SPLITS = {
    "train": DATA_ROOT / "train" / "train" / "HE",
    "val": DATA_ROOT / "val" / "val" / "HE",
    "test": DATA_ROOT / "test" / "test" / "HE",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("CTRANSPATH FEATURE EXTRACTION")
print("=" * 70)
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)
print("Device:", DEVICE)
print("Dataset exists:", DATA_ROOT.exists())
print("Checkpoint exists:", CHECKPOINT_PATH.exists())
print("Output:", OUTPUT_ROOT)

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )


# ============================================================
# 2. LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

if "model_state_dict" not in checkpoint:
    raise KeyError("Checkpoint has no model_state_dict.")

saved_state = checkpoint["model_state_dict"]

print("\nCheckpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint model ID:", checkpoint.get("model_id"))
print("Checkpoint feature dimension:", checkpoint.get("feature_dim"))
print("Checkpoint best validation Macro F1:",
      checkpoint.get("best_val_macro_f1"))

if checkpoint.get("model_id") != MODEL_ID:
    raise ValueError(
        "Checkpoint model ID does not match configured MODEL_ID."
    )

if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
    raise ValueError(
        f"Expected feature_dim={FEATURE_DIM}, "
        f"got {checkpoint.get('feature_dim')}"
    )


# ============================================================
# 3. CREATE MODEL AND RECONSTRUCT CHECKPOINT STEM
# ============================================================

# Create the timm model without its classification head.
backbone = timm.create_model(
    MODEL_ID,
    pretrained=False,
    num_classes=0
)

# The checkpoint has:
# proj.0: Conv2d(3, 12, 3x3)
# proj.1: BatchNorm2d(12), no bias
# proj.2: ReLU
# proj.3: Conv2d(12, 24, 3x3)
# proj.4: BatchNorm2d(24)
# proj.5: ReLU
# proj.6: Conv2d(24, 96, 1x1)
#
# Strides and padding below are inferred, not verified against
# the original training source. Verify them before scientific use.

backbone.patch_embed.proj = nn.Sequential(
    nn.Conv2d(
        3, 12,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    ),
    nn.BatchNorm2d(12, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(
        12, 24,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    ),
    nn.BatchNorm2d(24, affine=True),
    nn.ReLU(inplace=True),

    nn.Conv2d(
        24, 96,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True
    )
)

# Extract backbone-only parameters from the classifier checkpoint.
backbone_state = {
    key[len("backbone."):]: value
    for key, value in saved_state.items()
    if key.startswith("backbone.")
}

if not backbone_state:
    raise RuntimeError(
        "No backbone.* parameters found in checkpoint."
    )

# Strict loading is intentional: never silently ignore mismatches.
load_result = backbone.load_state_dict(
    backbone_state,
    strict=True
)

print("\nBackbone checkpoint loaded.")
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)

backbone = backbone.to(DEVICE)
backbone.eval()

for parameter in backbone.parameters():
    parameter.requires_grad = False


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

# These settings must match the preprocessing used during
# fine-tuning. Verify the original training transform.
# If it used different normalization or resizing, change this
# section to match the training code exactly.

image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# 5. DATASET
# ============================================================

def parse_label(filename):
    """
    Examples:
        00003_train_1+.png -> 1+
        sample_2+.png      -> 2+
    """
    suffix = Path(filename).stem.rsplit("_", 1)[-1]
    return suffix if suffix in CLASS_NAMES else None


class HEDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        self.paths = sorted([
            path for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in extensions
        ])

        if not self.paths:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return tensor, index


# ============================================================
# 6. CONVERT BACKBONE OUTPUT TO SPATIAL FEATURE MAP
# ============================================================

def extract_spatial_map(features, expected_channels=768):
    """
    Accepted outputs:
      [B, H, W, C] -> [B, C, H, W]
      [B, C, H, W] -> unchanged
      [B, N, C]    -> [B, C, sqrt(N), sqrt(N)] only if N is square

    The function deliberately rejects ambiguous outputs.
    """

    if isinstance(features, (tuple, list)):
        if len(features) != 1:
            raise RuntimeError(
                "Multiple feature outputs returned. "
                "Select the intended backbone stage explicitly."
            )
        features = features[0]

    if not torch.is_tensor(features):
        raise TypeError(
            f"Unexpected output type: {type(features)}"
        )

    if features.ndim == 4:
        if features.shape[-1] == expected_channels:
            # NHWC -> NCHW
            features = features.permute(0, 3, 1, 2).contiguous()

        elif features.shape[1] == expected_channels:
            # Already NCHW
            pass

        else:
            raise RuntimeError(
                "Cannot identify channel dimension: "
                f"{tuple(features.shape)}"
            )

    elif features.ndim == 3:
        if features.shape[-1] != expected_channels:
            raise RuntimeError(
                f"Expected last dimension {expected_channels}; "
                f"received {tuple(features.shape)}"
            )

        batch, tokens, channels = features.shape
        side = int(tokens ** 0.5)

        if side * side != tokens:
            raise RuntimeError(
                f"Token count {tokens} is not square. "
                "Cannot safely reshape to a spatial map."
            )

        features = features.transpose(1, 2).reshape(
            batch, channels, side, side
        )

    else:
        raise RuntimeError(
            f"Unsupported feature shape: {tuple(features.shape)}"
        )

    if features.shape[1] != expected_channels:
        raise RuntimeError(
            f"Expected {expected_channels} channels; "
            f"got {features.shape[1]}"
        )

    return features


# ============================================================
# 7. TEST MODEL FORWARD PASS BEFORE FULL EXTRACTION
# ============================================================

print("\nTesting model forward pass...")

with torch.inference_mode():
    dummy = torch.zeros(
        2, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=DEVICE
    )

    raw_output = backbone.forward_features(dummy)

    print("Raw backbone output shape:", tuple(raw_output.shape))

    test_spatial = extract_spatial_map(
        raw_output,
        expected_channels=FEATURE_DIM
    )

    test_pooled = test_spatial.mean(dim=(2, 3))

    print("Spatial map shape:", tuple(test_spatial.shape))
    print("Pooled feature shape:", tuple(test_pooled.shape))

    if not torch.isfinite(test_spatial).all():
        raise RuntimeError(
            "Non-finite values detected in test features."
        )

del dummy, raw_output, test_spatial, test_pooled

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 8. EXTRACT FEATURES FOR TRAIN / VAL / TEST
# ============================================================

summaries = []

for split_name, image_dir in SPLITS.items():

    print("\n" + "=" * 70)
    print("Processing split:", split_name)
    print("Image directory:", image_dir)
    print("=" * 70)

    dataset = HEDataset(
        image_dir=image_dir,
        transform=image_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False
    )

    spatial_batches = []
    pooled_batches = []
    metadata_rows = []

    with torch.inference_mode():
        for images, indices in tqdm(
            loader,
            desc=f"Extracting {split_name}"
        ):
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            # Backbone output; classification head is not used.
            raw_features = backbone.forward_features(images)

            spatial = extract_spatial_map(
                raw_features,
                expected_channels=FEATURE_DIM
            ).float()

            # Global average pooling:
            # [B, C, Hf, Wf] -> [B, C]
            pooled = spatial.mean(dim=(2, 3))

            if not torch.isfinite(spatial).all():
                raise RuntimeError(
                    f"Non-finite spatial features in {split_name}."
                )

            spatial_batches.append(
                spatial.cpu().numpy().astype(np.float32)
            )

            pooled_batches.append(
                pooled.cpu().numpy().astype(np.float32)
            )

            for idx in indices.tolist():
                path = dataset.paths[idx]
                label_name = parse_label(path.name)

                metadata_rows.append({
                    "split": split_name,
                    "index": idx,
                    "filename": path.name,
                    "path": str(path),
                    "label_name": label_name,
                    "label": (
                        CLASS_NAMES.index(label_name)
                        if label_name is not None
                        else -1
                    )
                })

    spatial_array = np.concatenate(
        spatial_batches, axis=0
    )

    pooled_array = np.concatenate(
        pooled_batches, axis=0
    )

    metadata = pd.DataFrame(metadata_rows).sort_values(
        "index"
    ).reset_index(drop=True)

    if not (
        len(metadata) == len(spatial_array) == len(pooled_array)
    ):
        raise RuntimeError(
            f"{split_name}: feature/metadata count mismatch."
        )

    split_output = OUTPUT_ROOT / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    spatial_file = split_output / "spatial_features.npy"
    pooled_file = split_output / "pooled_features.npy"
    metadata_file = split_output / "metadata.csv"

    np.save(spatial_file, spatial_array)
    np.save(pooled_file, pooled_array)
    metadata.to_csv(metadata_file, index=False)

    summary = {
        "split": split_name,
        "num_images": int(len(metadata)),
        "spatial_shape": list(spatial_array.shape),
        "pooled_shape": list(pooled_array.shape),
        "spatial_file": str(spatial_file),
        "pooled_file": str(pooled_file),
        "metadata_file": str(metadata_file),
        "unparsed_labels": int((metadata["label"] == -1).sum())
    }

    summaries.append(summary)

    print("\nSaved spatial features:", spatial_file)
    print("Saved pooled features:", pooled_file)
    print("Saved metadata:", metadata_file)
    print("Spatial array shape:", spatial_array.shape)
    print("Pooled array shape:", pooled_array.shape)
    print("Unparsed labels:", summary["unparsed_labels"])

    del spatial_batches, pooled_batches
    del spatial_array, pooled_array, metadata

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 9. SAVE EXTRACTION CONFIGURATION
# ============================================================

config = {
    "model_id": MODEL_ID,
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_epoch": checkpoint.get("epoch"),
    "best_validation_macro_f1": checkpoint.get(
        "best_val_macro_f1"
    ),
    "feature_dim": FEATURE_DIM,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "verification_required": True
    },
    "stem_reconstruction": {
        "verification_required": True,
        "note": (
            "Conv stem was reconstructed from checkpoint tensor "
            "shapes. Confirm stride/padding and architecture against "
            "the original training model before scientific use."
        )
    },
    "feature_types": [
        "spatial_features: [N, C, Hf, Wf]",
        "pooled_features: [N, C]"
    ],
    "splits": summaries
}

config_path = OUTPUT_ROOT / "extraction_config.json"

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION FINISHED")
print("=" * 70)
print("Output root:", OUTPUT_ROOT)
print("Configuration:", config_path)

for item in summaries:
    print(
        f"{item['split']}: "
        f"N={item['num_images']}, "
        f"spatial={item['spatial_shape']}, "
        f"pooled={item['pooled_shape']}"
    )

print("\nFiles are saved under:", OUTPUT_ROOT)
