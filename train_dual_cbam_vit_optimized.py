import os
import sys
import json
import time
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, cohen_kappa_score
from PIL import Image
from tqdm import tqdm

# Import plotting suite from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.plot_dashboard import (
    plot_training_dashboard,
    plot_confusion_matrix_heatmap,
    plot_per_class_metrics,
    plot_roc_auc_curves
)

# Seed reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS (READY FOR HYPERPARAMETER TUNING ALGORITHMS)
# ==============================================================================
DATA_DIR    = r"D:\test_2\processed_dataset"
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")

# Isolated Separate Output Directory
OUT_DIR     = r"D:\test_2\outputs_dual_cbam_vit_optimized"
CKPT_DIR    = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR    = os.path.join(OUT_DIR, "plots")

# ------------------------------------------------------------------------------
# 12 GB RTX 4080 VRAM OPTIMIZED PARAMETERS
# ------------------------------------------------------------------------------
BATCH_SIZE         = 16  # Reduced micro-batch size for VRAM safety
ACCUMULATION_STEPS = 2   # Effective Batch Size = 16 * 2 = 32
IMAGE_SIZE         = 224
EPOCHS_P1          = 15
EPOCHS_P2          = 15

# Hyperparameters (Exposed for Tuning Algorithms: Optuna / Grid Search / Bayesian)
HYPERPARAMS = {
    "lr_phase1": 1e-4,
    "lr_phase2": 1e-5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "cbam_reduction": 16,
    "dropout_p1": 0.4,
    "dropout_p2": 0.3
}

NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in [OUT_DIR, CKPT_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# Helper function for atomic checkpoint saves
def save_atomic_checkpoint(state, target_path):
    tmp_path = target_path + ".tmp"
    torch.save(state, tmp_path)
    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(tmp_path, target_path)

# ==============================================================================
# ATTENTION MODULES: CHANNEL ATTENTION + SPATIAL ATTENTION (CBAM)
# ==============================================================================
class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) for ResNet-50 Feature Recalibration."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) for Highlighting X-Ray Bone Regions."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """CBAM: Sequential Channel Attention followed by Spatial Attention."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ==============================================================================
# PARALLEL DUAL-BRANCH FUSION ARCHITECTURE (RESNET-50 CBAM + ViT-B/16)
# ==============================================================================
class DualBranchOsteoporosisFusion(nn.Module):
    def __init__(self, num_classes: int, cbam_reduction: int = 16):
        super().__init__()
        print("[ARCHITECTURE] Initializing Parallel Dual-Branch Fusion (ResNet50 + CBAM & ViT-B/16 Transformer)...")
        
        # 1. Parallel Branch A: ResNet-50 Feature Extractor
        self.resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_conv = nn.Sequential(*list(self.resnet_base.children())[:-2])  # (B, 2048, H, W)
        
        # ResNet-50 CBAM Attention
        self.resnet_cbam = CBAMBlock(channels=2048, reduction=cbam_reduction, kernel_size=7)
        self.resnet_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. Parallel Branch B: Vision Transformer (ViT-B/16)
        self.vit_branch = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit_in_features = self.vit_branch.heads.head.in_features  # 768
        self.vit_branch.heads = nn.Identity()
        
        # 3. Parallel Feature Fusion Layer (2048 + 768 = 2816 Dims)
        fusion_dim = 2048 + vit_in_features
        
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(p=HYPERPARAMS["dropout_p1"]),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=HYPERPARAMS["dropout_p2"]),
            nn.Linear(512, num_classes)
        )

    def _freeze_backbone(self):
        for param in self.resnet_conv.parameters():
            param.requires_grad = False
        for param in self.resnet_cbam.parameters():
            param.requires_grad = True
        for param in self.vit_branch.parameters():
            param.requires_grad = False
        print("[MODEL] ResNet-50 and ViT backbones frozen for Phase 1 classifier head training.")

    def _unfreeze_backbone_vram_optimized(self):
        """
        VRAM Optimization for 12 GB Laptop GPU:
        - Freezes lower ResNet stem (conv1, bn1, layer1, layer2)
        - Unfreezes deep representation layers (layer3, layer4) + CBAM + ViT encoder blocks
        - Cuts gradient parameter memory footprint in half to prevent OOM crashes!
        """
        for name, param in self.resnet_conv.named_parameters():
            if "6" in name or "7" in name:  # Unfreeze layer3 and layer4 only
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        for param in self.resnet_cbam.parameters():
            param.requires_grad = True
            
        for param in self.vit_branch.parameters():
            param.requires_grad = True
            
        print("[MODEL] VRAM-Optimized Selective Unfreezing: ResNet Layer3/4 + CBAM + ViT unfrozen for Phase 2.")

    def forward(self, x):
        # Parallel Stream A: ResNet-50 + CBAM Attention
        resnet_map = self.resnet_conv(x)
        resnet_cbam_map = self.resnet_cbam(resnet_map)
        resnet_feats = self.resnet_pool(resnet_cbam_map).flatten(1)
        
        # Parallel Stream B: ViT-B/16 Vision Transformer
        vit_feats = self.vit_branch(x)
        
        # Parallel Feature Fusion
        fused_feats = torch.cat((resnet_feats, vit_feats), dim=1)
        out = self.fusion_classifier(fused_feats)
        return out

# ==============================================================================
# DATA LOADERS & TRANSFORMS
# ==============================================================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_datasets():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train'])
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=data_transforms['val_test'])
    test_dataset  = datasets.ImageFolder(TEST_DIR, transform=data_transforms['val_test'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# ==============================================================================
# AMP MIXED PRECISION & GRADIENT ACCUMULATION TRAINING ENGINE
# ==============================================================================
def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device, epoch_idx=1, total_epochs=15, last_val_loss=None, last_val_acc=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad()
    
    desc_str = f"Epoch {epoch_idx:02d}/{total_epochs:02d} [Train]"
    pbar = tqdm(dataloader, desc=desc_str, leave=False)
    
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # AMP Mixed Precision Autocast (reduces VRAM consumption by 50%)
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = criterion(out, y) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        
        # Gradient Accumulation Step
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * ACCUMULATION_STEPS * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        postfix = {
            'train_loss': f"{running_loss/total:.4f}",
            'train_acc': f"{100*correct/total:.2f}%"
        }
        if last_val_loss is not None and last_val_acc is not None:
            postfix['val_loss'] = f"{last_val_loss:.4f}"
            postfix['val_acc']  = f"{100*last_val_acc:.2f}%"
            
        pbar.set_postfix(postfix)
        
    if scheduler is not None:
        scheduler.step()
        
    # Clear VRAM cache after epoch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return running_loss / total, correct / total


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    return running_loss / total, correct / total


def benchmark_latency(model, device, num_samples=100):
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    
    for _ in range(10):
        _ = model(dummy_input)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            with torch.amp.autocast('cuda'):
                _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    total_time = time.time() - start_time
    latency_ms = (total_time / num_samples) * 1000
    fps = num_samples / total_time
    return latency_ms, fps

# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    print("=" * 85)
    print("DUAL-BRANCH CBAM + ViT-B/16 - VRAM OPTIMIZED & REALTIME FLUSHED AI PIPELINE")
    print("=" * 85)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset Path: {DATA_DIR}")
    print(f"Separate Output Folder: {OUT_DIR}")
    print(f"VRAM Safety: Micro-batch {BATCH_SIZE} | Accumulation Steps {ACCUMULATION_STEPS} | AMP Mixed Precision Enabled")
    print("-" * 85)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = load_datasets()
    class_names = train_ds.classes
    num_classes = len(class_names)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Sample Count -> Train (5x Augmented): {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    targets = [y for _, y in train_ds.samples]
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=targets)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print("Calculated Class Weights:", {class_names[i]: round(w, 3) for i, w in enumerate(weights)})

    model = DualBranchOsteoporosisFusion(num_classes=num_classes, cbam_reduction=HYPERPARAMS["cbam_reduction"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=HYPERPARAMS["label_smoothing"])
    scaler = torch.amp.GradScaler('cuda')

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "lr": [],         "phase1_epochs": EPOCHS_P1
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(CKPT_DIR, "best_model.pth")
    log_csv_path = os.path.join(OUT_DIR, "realtime_training_log.csv")

    # Initialize CSV Log File with Header
    with open(log_csv_path, "w") as f:
        f.write("phase,epoch,train_loss,train_acc_pct,val_loss,val_acc_pct,lr,timestamp\n")
        f.flush()

    last_val_loss, last_val_acc = None, None

    # --------------------------------------------------------------------------
    # PHASE 1: FROZEN DUAL-BRANCH BACKBONES TRAINING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("STARTING PHASE 1: FROZEN DUAL-BRANCH BACKBONES (CLASSIFIER HEAD WARMUP)")
    print("=" * 85)
    model._freeze_backbone()
    optimizer_p1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HYPERPARAMS["lr_phase1"], weight_decay=HYPERPARAMS["weight_decay"])
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=EPOCHS_P1)

    start_time_all = time.time()

    for epoch in range(1, EPOCHS_P1 + 1):
        curr_lr = optimizer_p1.param_groups[0]['lr']
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer_p1, scheduler_p1, scaler, DEVICE,
                                      epoch_idx=epoch, total_epochs=EPOCHS_P1,
                                      last_val_loss=last_val_loss, last_val_acc=last_val_acc)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)

        last_val_loss, last_val_acc = val_loss, val_acc

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(curr_lr)

        # Real-time console log
        print(f"[P1 Epoch {epoch:02d}/{EPOCHS_P1:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {curr_lr:.6f}")

        # Real-Time Flushed CSV Writing (Prevents Data Loss on Crash)
        curr_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_csv_path, "a") as f:
            f.write(f"1,{epoch},{tr_loss:.4f},{tr_acc*100:.2f},{val_loss:.4f},{val_acc*100:.2f},{curr_lr:.6f},{curr_time_str}\n")
            f.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_atomic_checkpoint({"model_state": model.state_dict(), "class_names": class_names, "val_acc": val_acc}, best_model_path)
            print(f"  --> Saved new best Dual-Branch CBAM model checkpoint! (Val Acc: {val_acc*100:.2f}%)")

    # --------------------------------------------------------------------------
    # PHASE 2: FULL DUAL-BRANCH VRAM-OPTIMIZED FINE-TUNING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("STARTING PHASE 2: FULL DUAL-BRANCH VRAM-OPTIMIZED FINE-TUNING")
    print("=" * 85)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model._unfreeze_backbone_vram_optimized()

    optimizer_p2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=HYPERPARAMS["lr_phase2"], weight_decay=HYPERPARAMS["weight_decay"])
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=EPOCHS_P2)

    for epoch in range(1, EPOCHS_P2 + 1):
        curr_lr = optimizer_p2.param_groups[0]['lr']
        total_ep = EPOCHS_P1 + epoch
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer_p2, scheduler_p2, scaler, DEVICE,
                                      epoch_idx=epoch, total_epochs=EPOCHS_P2,
                                      last_val_loss=last_val_loss, last_val_acc=last_val_acc)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)

        last_val_loss, last_val_acc = val_loss, val_acc

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(curr_lr)

        print(f"[P2 Epoch {epoch:02d}/{EPOCHS_P2:02d} (Total {total_ep:02d})] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {curr_lr:.6f}")

        curr_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_csv_path, "a") as f:
            f.write(f"2,{total_ep},{tr_loss:.4f},{tr_acc*100:.2f},{val_loss:.4f},{val_acc*100:.2f},{curr_lr:.6f},{curr_time_str}\n")
            f.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_atomic_checkpoint({"model_state": model.state_dict(), "class_names": class_names, "val_acc": val_acc}, best_model_path)
            print(f"  --> Saved new best Dual-Branch CBAM model checkpoint! (Val Acc: {val_acc*100:.2f}%)")

    total_training_time = time.time() - start_time_all
    print(f"\nTotal Training Elapsed Time: {total_training_time/60:.2f} minutes")

    with open(os.path.join(OUT_DIR, "metrics_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_dashboard(history, os.path.join(PLOT_DIR, "training_dashboard.png"))

    print("\n" + "=" * 85)
    print("INDEPENDENT TEST SET CLINICAL EVALUATION")
    print("=" * 85)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, DEVICE)

    y_true, y_pred, y_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(x)
                probs = F.softmax(out, dim=1)
            y_probs.append(probs.cpu().numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            y_true.extend(y.numpy())

    y_probs = np.vstack(y_probs)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str  = classification_report(y_true, y_pred, target_names=class_names)
    kappa_score = cohen_kappa_score(y_true, y_pred)

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Cohen's Kappa : {kappa_score:.4f}")
    print("\nFull Classification Report:\n")
    print(report_str)

    plot_confusion_matrix_heatmap(y_true, y_pred, class_names, os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plot_per_class_metrics(report_dict, class_names, os.path.join(PLOT_DIR, "per_class_metrics.png"))
    plot_roc_auc_curves(y_true, y_probs, class_names, os.path.join(PLOT_DIR, "roc_curves.png"))

    latency_ms, fps = benchmark_latency(model, DEVICE)
    print(f"\nInference Benchmark -> Latency: {latency_ms:.2f} ms/image | Throughput: {fps:.1f} FPS")

    benchmark_summary = {
        "model_architecture": "Patent-Grade Dual-Branch Fusion (ResNet50 + CBAM Attention & ViT-B/16)",
        "vram_optimization": "AMP Mixed Precision (FP16) + Gradient Accumulation + Selective Layer Unfreezing",
        "clinical_task": "Multi-Class Osteoporosis Screening (Normal, Osteopenia, Osteoporosis)",
        "dataset_summary": {
            "train_samples_5x_augmented": len(train_ds),
            "val_samples_clean": len(val_ds),
            "test_samples_clean": len(test_ds),
            "classes": class_names
        },
        "performance_metrics": {
            "test_accuracy_pct": round(test_acc * 100, 2),
            "test_loss": round(test_loss, 4),
            "cohens_kappa": round(kappa_score, 4),
            "macro_precision": round(report_dict['macro avg']['precision'], 4),
            "macro_recall": round(report_dict['macro avg']['recall'], 4),
            "macro_f1": round(report_dict['macro avg']['f1-score'], 4),
            "inference_latency_ms": round(latency_ms, 2),
            "throughput_fps": round(fps, 1)
        },
        "hyperparameters": HYPERPARAMS
    }

    summary_path = os.path.join(OUT_DIR, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    shutil.make_archive("dual_cbam_vit_optimized_outputs", "zip", OUT_DIR)
    print("\n" + "=" * 85)
    print("TRAINING & CLINICAL EVALUATION COMPLETE!")
    print("=" * 85)

if __name__ == "__main__":
    main()
