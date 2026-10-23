
# ============================================================
# Path A — Final Test Evaluation
# Frozen TransPath MoCo v3 + CLS Features + Logistic Regression
# No backbone fine-tuning
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

ROOT = Path("/kaggle/working/transpath_path_a")
FEATURE_DIR = ROOT / "pretrained_features"
OUT = ROOT / "test_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASS_IDS = [0, 1, 2, 3]
CLASS_NAMES = ["0", "1+", "2+", "3+"]

# Fit the final linear probe on Train + Validation.
# Test is never used for fitting or model selection.
FIT_ON_TRAIN_AND_VAL = True

# ------------------------------------------------------------
# 2. Load saved feature files
# ------------------------------------------------------------

def load_features(split):
    path = FEATURE_DIR / f"{split}_features.pt"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {split} feature file: {path}\n"
            "Run the Path A feature-extraction cell first."
        )

    try:
        pack = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        pack = torch.load(path, map_location="cpu")

    required = {"cls_features", "labels"}
    missing = required - set(pack.keys())
    if missing:
        raise KeyError(f"{split}: missing keys {missing}")

    X = pack["cls_features"]
    y = pack["labels"]

    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    if X.ndim != 2 or X.shape[1] != 384:
        raise ValueError(
            f"{split}: expected CLS shape (N, 384), got {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split}: feature/label count mismatch: "
            f"{len(X)} vs {len(y)}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{split}: features contain NaN or Inf")

    if not np.isin(y, CLASS_IDS).all():
        raise ValueError(
            f"{split}: unexpected labels {np.unique(y).tolist()}"
        )

    filenames = pack.get(
        "filenames", [f"{split}_{i}" for i in range(len(y))]
    )

    if len(filenames) != len(y):
        raise ValueError(f"{split}: filename count mismatch")

    return {
        "X": X,
        "y": y,
        "filenames": list(filenames),
    }


train = load_features("train")
val = load_features("val")
test = load_features("test")

for name, data in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    print(f"\n{name}:")
    print("  Features:", data["X"].shape)
    print("  Samples:", len(data["y"]))
    print(
        "  Class counts:",
        {
            CLASS_NAMES[c]: int((data["y"] == c).sum())
            for c in CLASS_IDS
        },
    )

# ------------------------------------------------------------
# 3. Prepare final fitting data
# ------------------------------------------------------------

if FIT_ON_TRAIN_AND_VAL:
    X_fit = np.concatenate([train["X"], val["X"]], axis=0)
    y_fit = np.concatenate([train["y"], val["y"]], axis=0)
    fit_description = "Train + Validation"
else:
    X_fit = train["X"]
    y_fit = train["y"]
    fit_description = "Train only"

if set(np.unique(y_fit)) != set(CLASS_IDS):
    raise ValueError(
        "The fitting data must contain all four HER2 classes."
    )

print("\nFinal fitting data:", fit_description)
print("Fitting samples:", len(y_fit))

# ------------------------------------------------------------
# 4. Train the frozen-feature linear probe
# ------------------------------------------------------------
# Same classifier configuration as the Path A CV analysis.
# StandardScaler is fitted only on the selected fitting data.

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=SEED,
    )),
])

model.fit(X_fit, y_fit)

print("Linear probe fitted successfully.")
print("Backbone fine-tuning: NO")

# ------------------------------------------------------------
# 5. Predict the held-out Test set
# ------------------------------------------------------------

X_test = test["X"]
y_test = test["y"]

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Explicitly ensure the model's class order matches class IDs.
fitted_classes = model.named_steps["classifier"].classes_
if not np.array_equal(fitted_classes, np.array(CLASS_IDS)):
    raise ValueError(
        f"Unexpected classifier class order: {fitted_classes}"
    )

# ------------------------------------------------------------
# 6. Calculate final metrics
# ------------------------------------------------------------

metrics = {
    "split": "test",
    "feature_representation": "CLS",
    "feature_dim": 384,
    "backbone": (
        "hf_hub:1aurent/"
        "vit_small_patch16_224.transpath_mocov3"
    ),
    "backbone_fine_tuned": False,
    "classifier": "LogisticRegression",
    "classifier_C": 1.0,
    "class_weight": "balanced",
    "max_iter": 3000,
    "fitting_data": fit_description,
    "n_fit": int(len(y_fit)),
    "n_test": int(len(y_test)),
    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, y_pred)
    ),
    "macro_f1": float(
        f1_score(
            y_test, y_pred,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )
    ),
    "weighted_f1": float(
        f1_score(
            y_test, y_pred,
            labels=CLASS_IDS,
            average="weighted",
            zero_division=0,
        )
    ),
}

print("\n" + "=" * 65)
print("PATH A — FINAL TEST RESULTS")
print("=" * 65)

for key in [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]:
    print(f"{key:20s}: {metrics[key]:.4f} ({metrics[key] * 100:.2f}%)")

# ------------------------------------------------------------
# 7. Per-class classification report
# ------------------------------------------------------------

report_dict = classification_report(
    y_test,
    y_pred,
    labels=CLASS_IDS,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).T

print("\nClassification report:")
print(report_df.round(4).to_string())

report_path = OUT / "classification_report.csv"
report_df.to_csv(report_path, index=True)

# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test, y_pred, labels=CLASS_IDS
)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(cm_df)

cm_path = OUT / "confusion_matrix.csv"
cm_df.to_csv(cm_path, index=True)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)
disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)
ax.set_title("Path A — Test Confusion Matrix")
fig.tight_layout()

cm_plot_path = OUT / "confusion_matrix.png"
fig.savefig(cm_plot_path, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

# ------------------------------------------------------------
# 9. Save per-image predictions
# ------------------------------------------------------------

predictions_df = pd.DataFrame({
    "filename": test["filenames"],
    "true_label_id": y_test,
    "true_label": [CLASS_NAMES[int(y)] for y in y_test],
    "predicted_label_id": y_pred,
    "predicted_label": [CLASS_NAMES[int(y)] for y in y_pred],
})

for class_id, class_name in enumerate(CLASS_NAMES):
    predictions_df[f"prob_{class_name}"] = y_prob[:, class_id]

predictions_df["correct"] = y_test == y_pred

predictions_path = OUT / "test_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

# ------------------------------------------------------------
# 10. Save machine-readable summary
# ------------------------------------------------------------

metrics["class_counts_test"] = {
    CLASS_NAMES[c]: int((y_test == c).sum())
    for c in CLASS_IDS
}
metrics["per_class_metrics"] = {
    CLASS_NAMES[c]: {
        "precision": float(report_dict[CLASS_NAMES[c]]["precision"]),
        "recall": float(report_dict[CLASS_NAMES[c]]["recall"]),
        "f1": float(report_dict[CLASS_NAMES[c]]["f1-score"]),
        "support": int(report_dict[CLASS_NAMES[c]]["support"]),
    }
    for c in CLASS_IDS
}
metrics["outputs"] = {
    "classification_report": str(report_path),
    "confusion_matrix_csv": str(cm_path),
    "confusion_matrix_plot": str(cm_plot_path),
    "test_predictions": str(predictions_path),
}

summary_path = OUT / "test_metrics.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\nSaved outputs:")
for path in [
    summary_path,
    report_path,
    cm_path,
    cm_plot_path,
    predictions_path,
]:
    print(" ", path)

print("\nPath A Test evaluation completed.")
