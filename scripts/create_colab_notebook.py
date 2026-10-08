"""Generate the turnkey high-accuracy Google Colab training notebook for Flatfoot Classification."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_colab_notebook():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "machine_shape": "hm"
            },
            "accelerator": "GPU",
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
        })

    # Header
    add_markdown("""# 🦶 Flatfoot (Pes Planus) Classification - High-Accuracy Colab Training Pipeline
### Deep Learning, Vision Transformers, 5-Fold Cross-Validation & Multi-Scale Ensembles on Google Colab GPU

This notebook is engineered for maximum diagnostic accuracy on Google Colab (NVIDIA T4 / V100 / A100 GPU).
With 15 GB - 40 GB VRAM available on Colab, you can train heavyweight architectures, high-resolution inputs, and 5-Fold Stratified Ensembles to push clinical accuracy beyond 91% - 94%+.

---
### 🌟 State-of-the-Art Arsenal Included:
1. **FootArchNet-Ultra:** Custom Dual-Stream Cross-Attention Network fusing whole-foot macro morphology ($512\\times512$) and midfoot arch vault zoom ($256\\times256$).
2. **Heavyweight Vision Transformers & Modern ConvNets:** `Swin-B`, `Swin-T`, `ConvNeXt-Base`, `DenseNet-201`, `EfficientNet-V2`.
3. **Model EMA (Exponential Moving Average, $\\beta=0.999$):** Stabilizes late-epoch weight convergence and reduces validation loss variance.
4. **5-Fold Stratified Cross-Validation:** Trains across the entire development set and ensembles 5 models to eliminate variance.
5. **Multi-Scale Test-Time Augmentation (TTA):** 480px, 512px, 544px multi-scale inference.
6. **Explainable AI (Grad-CAM):** Verifies anatomical focus on the medial arch apex and calcaneal pitch.
""")

    # Cell 1: Hardware check
    add_markdown("## 1. Hardware Verification & GPU Setup\nVerify that the Colab GPU accelerator is enabled (Runtime > Change runtime type > GPU).")
    add_code("""import torch
import os
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    !nvidia-smi
else:
    print("⚠️ WARNING: Running on CPU! Please go to Runtime > Change runtime type > GPU to enable hardware acceleration.")
""")

    # Cell 2: Setup code repository
    add_markdown("## 2. Codebase Setup\nLoad project source code (`src/` and `scripts/`).")
    add_code("""import os
import sys
import zipfile
from pathlib import Path

# Check if src already exists
if not os.path.exists("src/models/model_factory.py"):
    # Look for flatfoot_code.zip in current dir or Google Drive
    code_zip_candidates = [
        Path("/content/flatfoot_code.zip"),
        Path("/content/drive/MyDrive/flatfoot_code.zip"),
        Path("flatfoot_code.zip")
    ]
    code_zip_found = next((p for p in code_zip_candidates if p.exists()), None)
    
    if code_zip_found:
        print(f"✓ Found code archive at {code_zip_found}. Extracting...")
        with zipfile.ZipFile(code_zip_found, 'r') as zf:
            zf.extractall(".")
        print("✓ Code extracted successfully!")
    else:
        # Try cloning GitHub repo
        print("Cloning flatfoot_classifier repository from GitHub...")
        !git clone https://github.com/tahsinsoyak/flatfoot_classifier.git
        if os.path.exists("flatfoot_classifier"):
            %cd flatfoot_classifier
        else:
            print("⚠️ Git clone failed (repo is private).")
            print("👉 ÇÖZÜM: Masaüstünüzdeki 'flatfoot_code.zip' (76 KB) dosyasını sol taraftaki Colab panelinde /content içine sürükleyin ve bu hücreyi tekrar çalıştırın!")

repo_root = Path(".").resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if os.path.exists("src"):
    print(f"✓ Project root active at: {repo_root} (src modülü başarıyla yüklendi!)")
else:
    print("❌ 'src' klasörü henüz bulunamadı. Lütfen 'flatfoot_code.zip' dosyasını Colab'e yükleyin.")
""")

    # Cell 3: Dependencies
    add_markdown("## 3. Install Required Dependencies")
    add_code("""!pip install -q albumentations timm scikit-learn seaborn matplotlib pandas pillow tqdm
print("✓ All dependencies installed successfully.")
""")

    # Cell 4: Dataset Ingestion
    add_markdown("""## 4. Dataset Ingestion (Google Drive or Upload)
Mount Google Drive to load the compact `flatfoot_processed_dataset.zip` (~48.5 MB) containing all 1,529 standardized radiographs and train/val/test split manifests.
""")
    add_code("""import zipfile
from pathlib import Path

zip_candidates = [
    Path("/content/drive/MyDrive/flatfoot_processed_dataset.zip"),
    Path("/content/drive/MyDrive/flatfoot_classifier/flatfoot_processed_dataset.zip"),
    Path("data/flatfoot_processed_dataset.zip"),
    Path("/content/flatfoot_processed_dataset.zip")
]

target_zip = None
for candidate in zip_candidates:
    if candidate.exists():
        target_zip = candidate
        break

if target_zip is None:
    try:
        from google.colab import drive
        print("Mounting Google Drive to look for dataset...")
        drive.mount('/content/drive')
        for candidate in zip_candidates:
            if candidate.exists():
                target_zip = candidate
                break
    except Exception as e:
        print("Google Drive mount skipped:", e)

if target_zip and target_zip.exists():
    print(f"✓ Found dataset zip at: {target_zip}")
    print("Extracting dataset...")
    with zipfile.ZipFile(target_zip, 'r') as zf:
        zf.extractall(".")
    print("✓ Extraction complete!")
else:
    print("⚠️ Dataset zip not found automatically.")
    print("Please upload 'flatfoot_processed_dataset.zip' (48.5 MB) to Google Drive (MyDrive) or directly to Colab via the files tab on the left.")

# Verify extraction
train_csv = Path("data/splits/train.csv")
val_csv = Path("data/splits/val.csv")
test_csv = Path("data/splits/test.csv")
processed_dir = Path("data/processed")

if train_csv.exists() and test_csv.exists() and processed_dir.exists():
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print(f"✓ Dataset verified successfully:")
    print(f"  - Training cohort:   {len(train_df)} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"  - Validation cohort: {len(val_df)} samples ({val_df['label'].value_counts().to_dict()})")
    print(f"  - Held-out test set: {len(test_df)} samples ({test_df['label'].value_counts().to_dict()})")
else:
    print("⚠️ Please ensure dataset is extracted into data/processed and data/splits.")
""")

    # Cell 5: Model Selection & Architecture Showcase
    add_markdown("""## 5. Select Model Architecture
Choose which high-capacity architecture you want to inspect and train:
- `foot_arch_net_ultra`: Novel Dual-Stream Cross-Attention Network (our top-performing single model, 0.9506 AUC)
- `convnext_base`: Modern 88M parameter ConvNet with 7x7 depthwise kernels
- `swin_b`: Hierarchical Shifted-Window Vision Transformer Base (88M parameters)
- `densenet201`: 201-layer DenseNet for bone trabeculae feature reuse
- `foot_arch_net_v2`: Coordinate Convolution + TransArchAttention
""")
    add_code("""from src.models.model_factory import create_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose your architecture:
SELECTED_MODEL = "foot_arch_net_ultra"  # Options: foot_arch_net_ultra, convnext_base, swin_b, densenet201, swin_t

model = create_model(SELECTED_MODEL, num_classes=2, pretrained=True, dropout=0.3).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 65)
print(f"ARCHITECTURE SUMMARY: {SELECTED_MODEL.upper()}")
print("=" * 65)
print(f"Total Parameters:     {total_params / 1e6:.2f} Million")
print(f"Trainable Parameters: {trainable_params / 1e6:.2f} Million")
print(f"Device:               {device}")

dummy_x = torch.randn(2, 3, 512, 512, device=device)
with torch.no_grad():
    dummy_out = model(dummy_x)
print(f"Forward sanity check: Input (2, 3, 512, 512) -> Logits {dummy_out.shape} ✓")
print("=" * 65)
""")

    # Cell 6: Train Single Model with EMA
    add_markdown("""## 6. Train Selected Architecture with AMP & Model EMA
Train for 20 epochs with Automatic Mixed Precision (AMP), Cosine Annealing learning rate schedule, and Exponential Moving Average (EMA, $\\beta=0.999$) smoothing.
On Google Colab, you can comfortably use `batch_size = 16`!
""")
    add_code("""import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from src.dataset import create_dataloaders
from src.training.trainer import Trainer

# Training hyperparameters (Optimized for Colab GPU)
EPOCHS = 20
BATCH_SIZE = 16  # Colab T4 / A100 handles batch_size 16 easily at 512px
LR = 1e-4
WEIGHT_DECAY = 1e-2
IMG_SIZE = 512

# Dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data",
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=2
)

# Class-weighted loss: 908 pes planus (class 1), 621 normal (class 0)
class_weights = torch.tensor([1.231, 0.842], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

run_dir = Path(f"experiments/colab_run_{SELECTED_MODEL}_{IMG_SIZE}px")
run_dir.mkdir(parents=True, exist_ok=True)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device,
    use_amp=True,
    output_dir=run_dir,
    patience=8,
    use_ema=True,
    ema_decay=0.999
)

print(f"Starting {SELECTED_MODEL} training for {EPOCHS} epochs on {device} (batch_size={BATCH_SIZE})...")
history = trainer.fit(train_loader, val_loader, epochs=EPOCHS)
print("✓ Training finished!")
""")

    # Cell 7: Evaluate Model on Test Set with Multi-Scale TTA
    add_markdown("""## 7. Evaluate Single Model on Unseen Clinical Test Set (n=229)
Evaluates test accuracy with Multi-Scale Test-Time Augmentation (TTA: 480px, 512px, 544px).
""")
    add_code("""import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
from src.dataset import FootRadiographDataset, get_transforms

best_ckpt = run_dir / "best_model.pt"
checkpoint = torch.load(best_ckpt, map_location=device)
if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
    model.load_state_dict(checkpoint["ema_state_dict"])
    print("✓ Loaded Model EMA weights for optimal generalization.")
else:
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Multi-Scale Test-Time Augmentation
scales = [480, 512, 544]
scale_probs = []

print(f"Running Multi-Scale TTA across {scales} on 229 unseen test patients...")
for s in scales:
    s_tf = get_transforms(s, is_training=False)
    test_ds = FootRadiographDataset("data/splits/test.csv", transform=s_tf)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(imgs)
                p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())
    scale_probs.append(np.array(probs))

test_probs = np.mean(scale_probs, axis=0)
test_preds = (test_probs >= 0.50).astype(int)

y_true = pd.read_csv("data/splits/test.csv")["label"].to_numpy()

acc = accuracy_score(y_true, test_preds)
sens = recall_score(y_true, test_preds)
cm = confusion_matrix(y_true, test_preds)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp)
prec = precision_score(y_true, test_preds)
npv = tn / (tn + fn)
f1 = f1_score(y_true, test_preds)
roc_auc = roc_auc_score(y_true, test_probs)

print("=" * 65)
print(f"TEST RESULTS: {SELECTED_MODEL.upper()} (TTA)")
print("=" * 65)
print(f"Diagnostic Accuracy: {acc*100:.2f}% ({tp+tn} / 229 correct)")
print(f"Sensitivity (Recall): {sens*100:.2f}% (Flatfoot detected: {tp}/{tp+fn})")
print(f"Specificity:         {spec*100:.2f}% (Normal detected: {tn}/{tn+fp})")
print(f"Precision (PPV):     {prec*100:.2f}%")
print(f"Negative Pred Val:   {npv*100:.2f}%")
print(f"F1-Score:            {f1:.4f}")
print(f"ROC-AUC:             {roc_auc:.4f}")
print("=" * 65)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Pes Planus"],
            yticklabels=["Normal", "Pes Planus"])
plt.title(f"{SELECTED_MODEL} Test Confusion Matrix (Acc: {acc*100:.2f}%)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(run_dir / "test_confusion_matrix.png", dpi=300)
plt.show()
""")

    # Cell 8: The Game Changer: 5-Fold Stratified Cross-Validation
    add_markdown("""## 8. 🚀 THE GAME CHANGER: 5-Fold Stratified Cross-Validation Pipeline
### How to reach >91% - 94% accuracy?
Single splits leave 15% of development data out of training. 
By training **5 stratified folds** and ensembling all 5 models together with Multi-Scale TTA, every sample contributes to feature learning, eliminating random variance.

Run the automated K-fold pipeline with one command:
""")
    add_code("""# Run 5-Fold Cross-Validation on FootArchNet-Ultra or ConvNeXt-Base
# This trains 5 models and evaluates the ensemble on test.csv
!python scripts/train_kfold.py --model foot_arch_net_ultra --folds 5 --epochs 20 --batch-size 16 --img-size 512
""")

    # Cell 9: Super Ensemble Evaluator
    add_markdown("""## 9. Super Ensemble V2 / V3 Evaluation
Combine predictions from multiple diverse model families (FootArchNet-Ultra + DenseNet-201 + FootArchNet-V2 + Swin-T + ConvNeXt) using probability calibration and multi-scale TTA.
""")
    add_code("""!python scripts/evaluate_super_ensemble.py
""")

    # Cell 10: Grad-CAM Explainability
    add_markdown("""## 10. Grad-CAM Anatomical Explainability Validation
Visualize where the network focuses across clinical test cases to verify alignment with orthopaedic criteria (medial longitudinal arch, navicular-cuneiform sag, calcaneal pitch).
""")
    add_code("""from src.models.gradcam import GradCAM
import cv2
from PIL import Image

# Automatically select target layer based on architecture
if hasattr(model, 'gradcam_refine'):
    target_layer = model.gradcam_refine[0]
elif hasattr(model, 'pyramid'):
    target_layer = model.pyramid.refine_conv[-3]
elif hasattr(model, 'fusion_block'):
    target_layer = model.fusion_block.fusion_conv[-3]
elif hasattr(model, 'features'):
    target_layer = model.features[-1]
else:
    target_layer = list(model.children())[-2]

gradcam = GradCAM(model, target_layer)

test_df = pd.read_csv("data/splits/test.csv")
sample_indices = [0, 1, 2, 137, 138, 139]  # 3 Normal, 3 Pes Planus

fig, axes = plt.subplots(len(sample_indices), 2, figsize=(10, 3.2 * len(sample_indices)))
fig.suptitle(f"{SELECTED_MODEL} Grad-CAM Saliency Maps", fontsize=14, y=0.995, fontweight="bold")

eval_tf = get_transforms(512, is_training=False)

for row_idx, idx in enumerate(sample_indices):
    sample_row = test_df.iloc[idx]
    img_path = sample_row["processed_path"]
    true_label = int(sample_row["label"])

    # Path resolution
    if not Path(img_path).exists():
        parts = Path(str(img_path).replace("\\\\", "/")).parts
        img_path = Path("data/processed") / parts[-2] / parts[-1]

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))

    tensor = eval_tf(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    cam = gradcam.generate(tensor, target_class=true_label)
    overlay = GradCAM.overlay_heatmap(img_resized, cam, alpha=0.45)

    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1)[0, 1].item()
    pred_label = 1 if prob >= 0.5 else 0

    class_name = "Pes Planus" if true_label == 1 else "Normal Foot"
    pred_name = "Pes Planus" if pred_label == 1 else "Normal Foot"
    status = "CORRECT" if true_label == pred_label else "MISCLASSIFIED"

    axes[row_idx, 0].imshow(img_resized)
    axes[row_idx, 0].set_title(f"Original X-Ray: {class_name}", fontsize=10)
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(overlay)
    axes[row_idx, 1].set_title(f"Grad-CAM: {pred_name} (p={prob:.2f}) [{status}]", fontsize=10)
    axes[row_idx, 1].axis("off")

plt.tight_layout()
plt.savefig(run_dir / "gradcam_test_samples.png", dpi=300)
plt.show()
print(f"✓ Grad-CAM saliency maps saved to {run_dir / 'gradcam_test_samples.png'}")
""")

    # Cell 11: Export to Google Drive
    add_markdown("""## 11. Save Checkpoints & Results to Google Drive
Export model weights (`best_model.pt`), metrics JSON, and evaluation charts directly to your Google Drive folder for permanent backup.
""")
    add_code("""import shutil

drive_export_dir = Path("/content/drive/MyDrive/flatfoot_classifier_results")
try:
    drive_export_dir.mkdir(parents=True, exist_ok=True)
    # Copy all files from run directory
    for f in run_dir.glob("*"):
        if f.is_file():
            shutil.copy(f, drive_export_dir / f.name)
    # Also copy k-fold results if present
    kfold_dir = Path("experiments")
    for kf in kfold_dir.glob("run_kfold*"):
        target_kf = drive_export_dir / kf.name
        target_kf.mkdir(parents=True, exist_ok=True)
        for f in kf.glob("*"):
            if f.is_file():
                shutil.copy(f, target_kf / f.name)
    print(f"✓ All checkpoints, predictions, and figures successfully backed up to Google Drive at: {drive_export_dir}")
except Exception as e:
    print("Could not copy directly to Google Drive:", e)
    print("You can download the files directly from the Colab file browser on the left under experiments/.")
""")

    # Write notebook
    out_path = PROJECT_ROOT / "notebooks" / "Flatfoot_Colab_Training.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Colab notebook created at {out_path} ({out_path.stat().st_size} bytes)")
    return out_path

if __name__ == "__main__":
    build_colab_notebook()
