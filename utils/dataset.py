"""
BCI Dataset — H&E به IHC
=========================
این ماژول تصاویر جفت‌شده‌ی H&E و IHC از مجموعه‌داده BCI را لود می‌کند.

نسخه‌ی این فایل بر اساس ساختار واقعی کشف‌شده در کگل نوشته شده
(نه یک فرض تئوریک) — به configs/paths.py مراجعه کن که توضیح کامل
ساختار پوشه‌ها آنجا آمده است.

نکته‌ی مهم درباره‌ی split «test»:
    برخلاف train و val که تصاویر HE و IHC در دو پوشه‌ی خواهر کنار هم
    هستند، برای test:
        - تصاویر HE در:           test/test/HE
        - تصاویر IHC (ground truth) در: groundtruth/groundtruth
    این دو مسیر جدا از هم هستند اما نام فایل‌هایشان یکی است، پس هنوز
    می‌توان بر اساس نام فایل جفت‌سازی کرد.

نکته درباره‌ی نام فایل در val:
    فایل‌های داخل val/val/HE و val/val/IHC نام‌هایی مثل
    "00003_train_1+.png" دارند (یعنی همچنان کلمه‌ی "train" در اسمشان
    هست، چون این زیرمجموعه‌ای از داده‌ی train اصلی است که برای
    validation کنار گذاشته شده). این تفاوتی در منطق کد ایجاد نمی‌کند
    چون parse_her2_level_from_filename فقط دنبال آخرین عدد/علامت +
    می‌گردد، نه کلمه‌ی split.

فرمت نام فایل: {index}_{split_tag}_{her2_level}.png
مثال: 00000_train_3+.png -> her2_level = "3+"
(split_tag می‌تواند train/test باشد صرف‌نظر از اینکه فایل واقعاً
 در کدام پوشه split قرار دارد)
"""

import os
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.paths import DATASET_ROOT, SPLIT_PATHS, IMG_SIZE, EXPECTED_COUNTS


# نگاشت سطوح HER2 به عدد صحیح برای دسته‌بندی
HER2_LEVEL_TO_IDX = {
    "0": 0,
    "1+": 1,
    "2+": 2,
    "3+": 3,
}
IDX_TO_HER2_LEVEL = {v: k for k, v in HER2_LEVEL_TO_IDX.items()}


def parse_her2_level_from_filename(filename: str) -> str:
    """
    از نام فایل مثل '00000_train_3+.png' سطح HER2 را استخراج می‌کند.
    خروجی یکی از: '0', '1+', '2+', '3+'
    """
    stem = Path(filename).stem  # 00000_train_3+
    match = re.search(r"_(\d\+?)$", stem)
    if match is None:
        raise ValueError(
            f"نتوانستم سطح HER2 را از نام فایل استخراج کنم: '{filename}'. "
            f"فرمت مورد انتظار: {{index}}_{{split_tag}}_{{level}}.png "
            f"مثل 00000_train_3+.png"
        )
    level = match.group(1)
    if level not in HER2_LEVEL_TO_IDX:
        raise ValueError(f"سطح HER2 نامعتبر '{level}' از فایل '{filename}'")
    return level


class BCIDataset(Dataset):
    """
    Dataset برای جفت تصاویر H&E -> IHC مجموعه‌داده BCI.

    هر آیتم شامل:
        he_img: تنسور H&E نرمال‌شده در بازه [-1, 1]، شکل (3, H, W)
        ihc_img: تنسور IHC نرمال‌شده در بازه [-1, 1]، شکل (3, H, W)
        her2_level: رشته‌ی سطح HER2 ('0', '1+', '2+', '3+')
        her2_idx: اندیس صحیح متناظر (0 تا 3)
        filename: نام فایل (برای دیباگ و ردیابی)
    """

    def __init__(
        self,
        root_dir: str = None,
        split: str = "train",
        img_size: int = None,
        augment: bool = False,
        verify_count: bool = True,
    ):
        """
        Args:
            root_dir: مسیر ریشه دیتاست. اگر None باشد از configs.paths.DATASET_ROOT
                      استفاده می‌شود.
            split: 'train', 'val' یا 'test'
            img_size: ابعاد نهایی تصویر. اگر None باشد از configs.paths.IMG_SIZE
                      استفاده می‌شود (که 512 است).
            augment: اگر True باشد، augmentation های هندسی ساده اعمال می‌شود
                     (فقط برای train توصیه می‌شود، نه val/test)
            verify_count: اگر True باشد، تعداد فایل‌های پیداشده را با
                          EXPECTED_COUNTS در configs/paths.py مقایسه می‌کند
                          و در صورت مغایرت هشدار می‌دهد.
        """
        if split not in SPLIT_PATHS:
            raise ValueError(
                f"split نامعتبر: '{split}'. باید یکی از {list(SPLIT_PATHS.keys())} باشد."
            )

        self.root_dir = Path(root_dir) if root_dir is not None else Path(DATASET_ROOT)
        self.split = split
        self.img_size = img_size if img_size is not None else IMG_SIZE
        self.augment = augment

        he_rel_path = SPLIT_PATHS[split]["he"]
        ihc_rel_path = SPLIT_PATHS[split]["ihc"]

        self.he_dir = self.root_dir / he_rel_path
        self.ihc_dir = self.root_dir / ihc_rel_path

        if not self.he_dir.exists():
            raise FileNotFoundError(f"پوشه H&E پیدا نشد: {self.he_dir}")
        if not self.ihc_dir.exists():
            raise FileNotFoundError(f"پوشه IHC پیدا نشد: {self.ihc_dir}")

        # فقط فایل‌هایی را نگه می‌داریم که هم در HE و هم در IHC وجود دارند
        he_files = {f.name for f in self.he_dir.glob("*.png")}
        ihc_files = {f.name for f in self.ihc_dir.glob("*.png")}
        common_files = sorted(he_files & ihc_files)

        if len(common_files) == 0:
            raise RuntimeError(
                f"هیچ فایل مشترکی بین {self.he_dir} و {self.ihc_dir} پیدا نشد. "
                f"تعداد فایل HE: {len(he_files)}, تعداد فایل IHC: {len(ihc_files)}"
            )

        missing_in_ihc = he_files - ihc_files
        missing_in_he = ihc_files - he_files
        if missing_in_ihc:
            print(
                f"[هشدار split={split}] {len(missing_in_ihc)} فایل در HE هست ولی "
                f"در IHC نیست (نادیده گرفته می‌شود). نمونه: {list(missing_in_ihc)[:3]}"
            )
        if missing_in_he:
            print(
                f"[هشدار split={split}] {len(missing_in_he)} فایل در IHC هست ولی "
                f"در HE نیست (نادیده گرفته می‌شود). نمونه: {list(missing_in_he)[:3]}"
            )

        self.filenames = common_files

        if verify_count and split in EXPECTED_COUNTS:
            expected = EXPECTED_COUNTS[split]
            actual = len(self.filenames)
            if actual != expected:
                print(
                    f"[هشدار sanity-check] split='{split}': تعداد نمونه‌های یافت‌شده "
                    f"({actual}) با تعداد مورد انتظار ({expected}) یکی نیست. "
                    f"این می‌تواند طبیعی باشد (مثلاً اگر برخی فایل‌ها فیلتر شده‌اند) "
                    f"یا نشانه‌ی یک مشکل در مسیرها باشد — لطفاً بررسی کن."
                )

        # --- تبدیل‌های پایه (resize احتیاطی + تبدیل به تنسور + نرمال‌سازی) ---
        base_transforms = [
            T.Resize(
                (self.img_size, self.img_size),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),  # مقادیر را به [0, 1] می‌برد
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # به [-1, 1]
        ]
        self.base_transform = T.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        filename = self.filenames[idx]

        he_path = self.he_dir / filename
        ihc_path = self.ihc_dir / filename

        he_img = Image.open(he_path).convert("RGB")
        ihc_img = Image.open(ihc_path).convert("RGB")

        # --- augmentation هندسی هم‌زمان (چرخش/فلیپ یکسان روی هر دو تصویر) ---
        if self.augment:
            he_img, ihc_img = self._apply_paired_augment(he_img, ihc_img)

        he_tensor = self.base_transform(he_img)
        ihc_tensor = self.base_transform(ihc_img)

        her2_level = parse_her2_level_from_filename(filename)
        her2_idx = HER2_LEVEL_TO_IDX[her2_level]

        return {
            "he_img": he_tensor,
            "ihc_img": ihc_tensor,
            "her2_level": her2_level,
            "her2_idx": her2_idx,
            "filename": filename,
        }

    def _apply_paired_augment(self, he_img: Image.Image, ihc_img: Image.Image):
        """
        Augmentation هندسی که باید دقیقاً یکسان روی هر دو تصویر (H&E و IHC)
        اعمال شود تا تطابق مکانی/پیکسلی بین آن‌ها از بین نرود.
        """
        import random

        if random.random() < 0.5:
            he_img = he_img.transpose(Image.FLIP_LEFT_RIGHT)
            ihc_img = ihc_img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            he_img = he_img.transpose(Image.FLIP_TOP_BOTTOM)
            ihc_img = ihc_img.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            k = random.choice([90, 180, 270])
            he_img = he_img.rotate(k, expand=True)
            ihc_img = ihc_img.rotate(k, expand=True)

        return he_img, ihc_img

    def get_class_counts(self) -> dict:
        """
        تعداد نمونه‌های هر کلاس HER2 را برمی‌گرداند.
        برای طراحی WeightedRandomSampler و بررسی عدم توازن داده استفاده می‌شود.
        """
        counts = {level: 0 for level in HER2_LEVEL_TO_IDX}
        for filename in self.filenames:
            level = parse_her2_level_from_filename(filename)
            counts[level] += 1
        return counts

    def get_sample_weights(self) -> torch.Tensor:
        """
        برای هر نمونه، وزنی متناسب با معکوس فراوانی کلاسش برمی‌گرداند.
        خروجی مستقیماً قابل استفاده در torch.utils.data.WeightedRandomSampler است.
        """
        counts = self.get_class_counts()
        weights = []
        for filename in self.filenames:
            level = parse_her2_level_from_filename(filename)
            class_count = counts[level]
            weight = 1.0 / class_count if class_count > 0 else 0.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.double)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    تبدیل تنسور نرمال‌شده [-1, 1] به بازه [0, 1] برای نمایش/ذخیره تصویر.
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)
