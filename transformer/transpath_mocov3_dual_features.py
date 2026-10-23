#!/usr/bin/env python3
"""Extract TransPath CLS and patch-token features for paired BCI H&E/IHC data.

This script intentionally does NOT train a classifier, save classifier checkpoints,
write duplicate .npz files, concatenate all splits, create plots, or build a ZIP.
It saves one .pt file per split, containing features and pairing metadata needed by
later conditional-LDM training.

Default: extract features from the original pretrained TransPath MoCo v3 backbone.
Optional: pass --finetuned-checkpoint to ALSO extract a second feature set from a
previously fine-tuned TransPath checkpoint. Only use that option once the
fine-tuned checkpoint has been validated.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_DIR = Path(
    "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"
)
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/transpath_features")
MODEL_NAME = "hf_hub:1aurent/vit_small_patch16_224.transpath_mocov3"
LABEL_TO_IDX = {"0": 0, "1+": 1, "2+": 2, "3+": 3}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label(filename: str) -> str:
    stem = Path(filename).stem
    match = re.search(r"_(0|1\+|2\+|3\+)$", stem)
    if match is None:
        raise ValueError(f"Cannot parse HER2 label from H&E filename: {filename}")
    return match.group(1)


def make_pair_index(ihc_dir: Path) -> Dict[str, Path]:
    if not ihc_dir.is_dir():
        raise FileNotFoundError(f"IHC directory not found: {ihc_dir}")
    index: Dict[str, Path] = {}
    for path in sorted(ihc_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            # Exact filename first; case-insensitive filename/stem fallback below.
            index[path.name] = path
            index[path.name.lower()] = path
            index[path.stem] = path
            index[path.stem.lower()] = path
    return index


def collect_split(data_dir: Path, split: str) -> pd.DataFrame:
    he_dir = data_dir / split / split / "HE"
    if split in {"train", "val"}:
        ihc_dir = data_dir / split / split / "IHC"
    elif split == "test":
        ihc_dir = data_dir / "groundtruth" / "groundtruth"
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not he_dir.is_dir():
        raise FileNotFoundError(f"H&E directory not found: {he_dir}")

    ihc_index = make_pair_index(ihc_dir)
    records: List[dict] = []
    unmatched: List[str] = []

    for he_path in sorted(he_dir.iterdir()):
        if not he_path.is_file() or he_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Prefer exact basename, then case-insensitive basename, then stem.
        ihc_path = (
            ihc_index.get(he_path.name)
            or ihc_index.get(he_path.name.lower())
            or ihc_index.get(he_path.stem)
            or ihc_index.get(he_path.stem.lower())
        )
        if ihc_path is None:
            unmatched.append(he_path.name)
            continue

        label_name = parse_label(he_path.name)
        records.append(
            {
                "filename": he_path.name,
                "he_path": str(he_path.resolve()),
                "ihc_path": str(ihc_path.resolve()),
                "label_name": label_name,
                "label": LABEL_TO_IDX[label_name],
                "split": split,
            }
        )

    if unmatched:
        preview = ", ".join(unmatched[:10])
        raise RuntimeError(
            f"Could not find paired IHC files for {len(unmatched)} H&E files "
            f"in split '{split}'. Examples: {preview}. Check filename mapping."
        )

    if not records:
        raise RuntimeError(f"No paired images found for split '{split}' in {he_dir}")

    return pd.DataFrame.from_records(records)


class PairedHEDataset(Dataset):
    """Loads H&E for feature extraction; IHC path is retained as target metadata."""

    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row["he_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return (
            image,
            int(row["label"]),
            str(row["label_name"]),
            str(row["filename"]),
            str(row["he_path"]),
            str(row["ihc_path"]),
            str(row["split"]),
        )


def load_checkpoint_state(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")


def extract_split(
    model: torch.nn.Module,
    dataframe: pd.DataFrame,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_dtype: torch.dtype,
    model_source: str,
) -> dict:
    split = str(dataframe.iloc[0]["split"])
    dataset = PairedHEDataset(dataframe, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    cls_chunks = []
    patch_chunks = []
    labels: List[int] = []
    label_names: List[str] = []
    filenames: List[str] = []
    he_paths: List[str] = []
    ihc_paths: List[str] = []
    splits: List[str] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images, batch_labels, batch_label_names, batch_filenames, batch_he, batch_ihc, batch_splits = batch
            images = images.to(device, non_blocking=True)
            tokens = model.forward_features(images)

            if tokens.ndim != 3:
                raise RuntimeError(f"Expected [B, tokens, dim], got {tuple(tokens.shape)}")
            if tokens.shape[1] != 197:
                raise RuntimeError(
                    f"Expected 197 tokens (CLS + 196 patches), got {tokens.shape[1]}"
                )
            if tokens.shape[2] != 384:
                raise RuntimeError(f"Expected embedding dimension 384, got {tokens.shape[2]}")

            cls_chunks.append(tokens[:, 0, :].to(dtype=feature_dtype).cpu())
            patch_chunks.append(tokens[:, 1:, :].to(dtype=feature_dtype).cpu())
            labels.extend(torch.as_tensor(batch_labels).tolist())
            label_names.extend(list(batch_label_names))
            filenames.extend(list(batch_filenames))
            he_paths.extend(list(batch_he))
            ihc_paths.extend(list(batch_ihc))
            splits.extend(list(batch_splits))

    return {
        "cls_features": torch.cat(cls_chunks, dim=0),
        "patch_features": torch.cat(patch_chunks, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_names": label_names,
        "filenames": filenames,
        "he_paths": he_paths,
        "ihc_paths": ihc_paths,
        "splits": splits,
        "model_name": MODEL_NAME,
        "feature_source": model_source,
        "feature_dim": 384,
        "num_patch_tokens": 196,
        "patch_grid_size": (14, 14),
        "feature_dtype": str(feature_dtype).replace("torch.", ""),
    }


def save_source_features(
    *,
    model: torch.nn.Module,
    source_name: str,
    dataframes: dict,
    transform,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_dtype: torch.dtype,
) -> None:
    source_dir = output_dir / f"{source_name}_features"
    source_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()

    for split in ("train", "val", "test"):
        print(f"\nExtracting {source_name} features: {split}")
        feature_data = extract_split(
            model=model,
            dataframe=dataframes[split],
            transform=transform,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_dtype=feature_dtype,
            model_source=(
                f"{source_name} TransPath MoCo v3 backbone; classifier head not used"
            ),
        )
        output_path = source_dir / f"{split}_features.pt"
        torch.save(feature_data, output_path)
        print(f"  CLS:   {tuple(feature_data['cls_features'].shape)}")
        print(f"  Patch: {tuple(feature_data['patch_features'].shape)}")
        print(f"  Saved: {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB)")
        del feature_data


def run_extraction(
    *,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_dtype: torch.dtype,
    checkpoint_path: Optional[Path],
) -> None:
    dataframes = {split: collect_split(data_dir, split) for split in ("train", "val", "test")}

    print("Paired dataset summary:")
    summary = pd.concat(dataframes.values(), ignore_index=True)
    print(summary.groupby(["split", "label_name"]).size().to_string())
    print("\nSplit sizes:")
    for split, frame in dataframes.items():
        print(f"  {split}: {len(frame)} paired H&E/IHC samples")

    # The classifier head is retained only so a previously trained checkpoint
    # can be loaded strictly. It is never called; forward_features() is used.
    pretrained_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=4)
    data_config = timm.data.resolve_model_data_config(pretrained_model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Always create the primary baseline from original pretrained weights.
    save_source_features(
        model=pretrained_model,
        source_name="pretrained",
        dataframes=dataframes,
        transform=transform,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_dtype=feature_dtype,
    )
    del pretrained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Optional second run for a validated fine-tuned checkpoint.
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")
        finetuned_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=4)
        state = load_checkpoint_state(checkpoint_path)
        finetuned_model.load_state_dict(state, strict=True)
        print(f"\nLoaded fine-tuned checkpoint: {checkpoint_path}")
        save_source_features(
            model=finetuned_model,
            source_name="finetuned",
            dataframes=dataframes,
            transform=transform,
            output_dir=output_dir,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            feature_dtype=feature_dtype,
        )
        del finetuned_model

    print("\nFeature extraction complete.")
    print(f"Output directory: {output_dir}")
    print("Each .pt file includes CLS/Patch features, labels, filenames, H&E paths, and paired IHC paths.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="float16 halves disk usage; cast features to float32 when feeding the LDM if needed.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=Path,
        default=None,
        help="Optional validated TransPath checkpoint. If omitted, only pretrained features are extracted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dtype = torch.float16 if args.feature_dtype == "float16" else torch.float32

    print("=" * 72)
    print("TransPath Feature Extraction for Conditional H&E-to-IHC LDM")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Feature storage dtype: {args.feature_dtype}")
    print(f"Batch size: {args.batch_size}")

    run_extraction(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_dtype=feature_dtype,
        checkpoint_path=args.finetuned_checkpoint,
    )


if __name__ == "__main__":
    main()
