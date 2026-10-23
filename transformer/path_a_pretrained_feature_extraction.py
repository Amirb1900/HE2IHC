#!/usr/bin/env python3
"""PATH A: extract CLS and patch features from pretrained TransPath, without HER2 fine-tuning."""
import argparse, json, random, re
from pathlib import Path
import pandas as pd
from PIL import Image
import timm, torch
from torch.utils.data import Dataset, DataLoader

DATA = Path("/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset")
OUT = Path("/kaggle/working/transpath_path_a")
MODEL = "hf_hub:1aurent/vit_small_patch16_224.transpath_mocov3"
LABELS = {"0": 0, "1+": 1, "2+": 2, "3+": 3}
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def pairs(folder):
    if not folder.is_dir(): raise FileNotFoundError(folder)
    d = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTS:
            for k in (p.name, p.name.lower(), p.stem, p.stem.lower()): d[k] = p
    return d

def collect(root, split):
    he_dir = root / split / split / "HE"
    ihc_dir = root / split / split / "IHC" if split != "test" else root / "groundtruth" / "groundtruth"
    lookup, rows, missing = pairs(ihc_dir), [], []
    if not he_dir.is_dir(): raise FileNotFoundError(he_dir)
    for he in sorted(he_dir.iterdir()):
        if not he.is_file() or he.suffix.lower() not in EXTS: continue
        ihc = lookup.get(he.name) or lookup.get(he.name.lower()) or lookup.get(he.stem) or lookup.get(he.stem.lower())
        if ihc is None: missing.append(he.name); continue
        m = re.search(r"_(0|1\+|2\+|3\+)$", he.stem)
        if not m: raise ValueError(f"HER2 suffix not found: {he.name}")
        label = m.group(1)
        rows.append(dict(filename=he.name, he_path=str(he.resolve()), ihc_path=str(ihc.resolve()),
                         label_name=label, label=LABELS[label], split=split))
    if missing: raise RuntimeError(f"{len(missing)} unmatched H&E files; examples: {missing[:8]}")
    if not rows: raise RuntimeError(f"No pairs in {split}")
    return pd.DataFrame(rows)

class HEDataset(Dataset):
    def __init__(self, df, tf): self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r.he_path) as im: x = self.tf(im.convert("RGB"))
        return x, int(r.label), r.label_name, r.filename, r.he_path, r.ihc_path, r.split

def extract(model, df, tf, device, batch, workers, dtype):
    loader = DataLoader(HEDataset(df, tf), batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    cls, patch, labels, names, files, hp, ip, splits = [], [], [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y, yn, fn, h, i, s in loader:
            tokens = model.forward_features(x.to(device, non_blocking=True))
            if tuple(tokens.shape[1:]) != (197, 384):
                raise RuntimeError(f"Expected [B,197,384] tokens; got {tuple(tokens.shape)}")
            cls.append(tokens[:, 0].to(dtype=dtype, device="cpu"))
            patch.append(tokens[:, 1:].to(dtype=dtype, device="cpu"))
            labels.extend(torch.as_tensor(y).tolist()); names.extend(list(yn)); files.extend(list(fn))
            hp.extend(list(h)); ip.extend(list(i)); splits.extend(list(s))
    return dict(cls_features=torch.cat(cls), patch_features=torch.cat(patch),
                labels=torch.tensor(labels, dtype=torch.long), label_names=names, filenames=files,
                he_paths=hp, ihc_paths=ip, splits=splits, model_name=MODEL,
                feature_source="pretrained TransPath MoCo v3; no HER2 fine-tuning",
                feature_dim=384, num_patch_tokens=196, patch_grid_size=(14, 14),
                feature_dtype=str(dtype).replace("torch.", ""))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DATA); p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--feature-dtype", choices=["float16","float32"], default="float16")
    a = p.parse_args()
    if a.batch_size < 1: p.error("--batch-size must be positive")
    random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if a.feature_dtype == "float16" else torch.float32
    a.output_dir.mkdir(parents=True, exist_ok=True)
    dfs = {s: collect(a.data_dir, s) for s in ("train","val","test")}
    print("Pair counts:", {s: len(d) for s,d in dfs.items()})
    print(pd.concat(dfs.values()).groupby(["split","label_name"]).size().to_string())
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(device).eval()
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    dest = a.output_dir / "pretrained_features"; dest.mkdir(parents=True, exist_ok=True)
    for split, df in dfs.items():
        data = extract(model, df, tf, device, a.batch_size, a.num_workers, dtype)
        path = dest / f"{split}_features.pt"; torch.save(data, path)
        print(split, tuple(data["cls_features"].shape), tuple(data["patch_features"].shape), path, f"{path.stat().st_size/2**30:.2f} GiB")
    manifest = dict(path="A", model=MODEL, data_dir=str(a.data_dir), output_dir=str(a.output_dir),
                    split_sizes={s:len(d) for s,d in dfs.items()}, labels=LABELS, feature_dtype=a.feature_dtype)
    (a.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PATH A done:", a.output_dir)

if __name__ == "__main__": main()
