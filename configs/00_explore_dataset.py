"""
اسکریپت شناسایی ساختار دیتاست BCI در کگل
==========================================
این اسکریپت رو اول از همه توی کگل اجرا کن (توی یک سلول نوت‌بوک).
خروجیش رو کامل کپی کن و برام بفرست تا دقیقاً بدونم ساختار پوشه‌ها،
نام فایل‌ها، فرمت‌ها و تعدادشون چیه. بعدش می‌تونم dataset.py رو
دقیقاً منطبق با واقعیت بنویسم/اصلاح کنم.

مسیر پایه‌ای که دادی:
/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset
"""

import os
from pathlib import Path
from collections import Counter

ROOT = Path("/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset")

print("=" * 70)
print("۱) آیا مسیر ریشه وجود دارد؟")
print("=" * 70)
print(f"مسیر: {ROOT}")
print(f"وجود دارد: {ROOT.exists()}")
print()

if not ROOT.exists():
    print("!! مسیر پیدا نشد. لیست /kaggle/input رو نشون میدم تا مسیر درست رو پیدا کنیم:")
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_dir():
            print(f"  DIR : {p}")
    raise SystemExit("مسیر بالا رو اصلاح کن و دوباره اجرا کن.")

print("=" * 70)
print("۲) ساختار کامل درختی پوشه‌ها (فقط پوشه‌ها، تا عمق ۴)")
print("=" * 70)


def print_tree(path: Path, prefix: str = "", max_depth: int = 4, depth: int = 0):
    if depth > max_depth:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return
    dirs = [e for e in entries if e.is_dir()]
    for d in dirs:
        print(f"{prefix}{d.name}/")
        print_tree(d, prefix + "  ", max_depth, depth + 1)


print_tree(ROOT)
print()

print("=" * 70)
print("۳) همه‌ی زیرپوشه‌ها به همراه تعداد فایل داخلشون")
print("=" * 70)
all_dirs = sorted([p for p in ROOT.rglob("*") if p.is_dir()])
for d in all_dirs:
    files = list(d.glob("*"))
    file_count = sum(1 for f in files if f.is_file())
    print(f"{d.relative_to(ROOT)}  ->  {file_count} فایل")
print()

print("=" * 70)
print("۴) پسوند فایل‌ها (چند تا از هر نوع در کل دیتاست)")
print("=" * 70)
ext_counter = Counter()
for f in ROOT.rglob("*"):
    if f.is_file():
        ext_counter[f.suffix.lower()] += 1
for ext, count in ext_counter.most_common():
    print(f"  {ext or '(بدون پسوند)'}: {count}")
print()

print("=" * 70)
print("۵) نمونه نام فایل از هر پوشه‌ی برگ (leaf directory) که فایل تصویر داره")
print("=" * 70)
image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
leaf_dirs_with_images = []
for d in all_dirs:
    files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
    if files:
        leaf_dirs_with_images.append((d, files))

for d, files in leaf_dirs_with_images:
    print(f"\n--- {d.relative_to(ROOT)} ({len(files)} تصویر) ---")
    for f in sorted(files)[:5]:
        print(f"   {f.name}")
    if len(files) > 5:
        print(f"   ... و {len(files) - 5} فایل دیگر")
print()

print("=" * 70)
print("۶) بررسی ابعاد چند تصویر نمونه (برای تایید resize به 512x512)")
print("=" * 70)
try:
    from PIL import Image

    checked = 0
    for d, files in leaf_dirs_with_images:
        if checked >= 6:
            break
        sample_file = sorted(files)[0]
        with Image.open(sample_file) as img:
            print(f"  {sample_file.relative_to(ROOT)}: {img.size} mode={img.mode}")
        checked += 1
except ImportError:
    print("  PIL نصب نیست، این بخش رد شد.")
print()

print("=" * 70)
print("۷) اگر پوشه‌های train/test (یا مشابه) و HE/IHC (یا مشابه) پیدا شدند،")
print("   بررسی تطابق نام فایل بین دو پوشه")
print("=" * 70)
# حدس می‌زنیم اسم پوشه‌ها چی ممکنه باشه (حساس به بزرگی/کوچکی حروف نیست)
possible_split_names = ["train", "test", "val", "valid", "validation"]
possible_stain_names = ["he", "h&e", "ihc"]

for d in all_dirs:
    name_lower = d.name.lower()
    if name_lower in possible_stain_names:
        sibling_dirs = [
            s for s in all_dirs
            if s.parent == d.parent and s != d and s.name.lower() in possible_stain_names
        ]
        for sib in sibling_dirs:
            files_d = {f.name for f in d.glob("*") if f.is_file()}
            files_sib = {f.name for f in sib.glob("*") if f.is_file()}
            common = files_d & files_sib
            print(f"\n  مقایسه: {d.relative_to(ROOT)}  <-->  {sib.relative_to(ROOT)}")
            print(f"    تعداد در {d.name}: {len(files_d)}")
            print(f"    تعداد در {sib.name}: {len(files_sib)}")
            print(f"    فایل مشترک (نام یکسان): {len(common)}")
            only_d = files_d - files_sib
            only_sib = files_sib - files_d
            if only_d:
                print(f"    فقط در {d.name} (نمونه): {list(only_d)[:3]}")
            if only_sib:
                print(f"    فقط در {sib.name} (نمونه): {list(only_sib)[:3]}")

print()
print("=" * 70)
print("۸) نمونه‌ی کامل چند نام فایل خام (برای بررسی الگوی نام‌گذاری سطح HER2)")
print("=" * 70)
sample_names = []
for d, files in leaf_dirs_with_images:
    sample_names.extend([f.name for f in sorted(files)[:10]])
for name in sample_names[:30]:
    print(f"  {name}")

print()
print("=" * 70)
print("پایان اسکریپت شناسایی. خروجی بالا رو کامل کپی کن و بفرست.")
print("=" * 70)
