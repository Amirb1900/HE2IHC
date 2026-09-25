"""
تنظیمات مسیر پروژه — تک منبع حقیقت برای همه‌ی مسیرها
========================================================
این فایل بر اساس خروجی واقعی اسکریپت scripts/00_explore_dataset.py
که در کگل اجرا شد، نوشته شده است (نه فرض اولیه).

ساختار واقعی کشف‌شده در کگل:

512BCI_dataset/
├── train/
│   └── train/
│       ├── HE/      (3396 تصویر)
│       └── IHC/     (3396 تصویر)
├── val/
│   └── val/
│       ├── HE/      (500 تصویر)   <- نام فایل‌ها داخلش هنوز "train" دارند
│       └── IHC/     (500 تصویر)      (چون این زیرمجموعه‌ای از train اصلی است)
├── test/
│   └── test/
│       └── HE/      (977 تصویر)   <- IHC اینجا نیست!
└── groundtruth/
    └── groundtruth/ (977 تصویر)   <- این‌ها همان IHC واقعی مجموعه test هستند

نکته مهم: برای split «test»، تصاویر HE از test/test/HE خوانده می‌شوند
ولی تصاویر IHC (ground truth) از یک مسیر کاملاً متفاوت یعنی
groundtruth/groundtruth خوانده می‌شوند. این تفاوت ساختاری نسبت به
train/val در کد utils/dataset.py مدیریت شده است.
"""

# مسیر ریشه‌ی دیتاست در کگل
DATASET_ROOT = "/kaggle/input/datasets/amirb1900/resized-bci-512/512BCI_dataset"

# --- مسیرهای هر split نسبت به DATASET_ROOT ---
# فرمت: (پوشه‌ی HE, پوشه‌ی IHC) — هر دو نسبت به DATASET_ROOT

SPLIT_PATHS = {
    "train": {
        "he": "train/train/HE",
        "ihc": "train/train/IHC",
    },
    "val": {
        "he": "val/val/HE",
        "ihc": "val/val/IHC",
    },
    "test": {
        "he": "test/test/HE",
        "ihc": "groundtruth/groundtruth",  # نکته: مسیر متفاوت با HE!
    },
}

# مسیر خروجی برای ذخیره چک‌پوینت‌ها، لاگ‌ها و نتایج
OUTPUT_ROOT = "/kaggle/working/outputs"

# ابعاد تصویر (طبق بررسی، همه تصاویر واقعاً 512x512 هستند - تایید شد)
IMG_SIZE = 512

# تعداد نمونه‌های مورد انتظار هر split (برای sanity check در ابتدای اجرا)
EXPECTED_COUNTS = {
    "train": 3396,
    "val": 500,
    "test": 977,
}
