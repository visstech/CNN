"""
train_resnet_pneumonia.py

Pretrained ResNet18 pipeline for Chest X-ray Pneumonia classification.
Features:
 - ImageFolder data loading (train/val/test)
 - Data augmentation (train) and ImageNet normalization
 - Transfer learning with pretrained ResNet18
 - Mixed precision training (torch.amp)
 - ReduceLROnPlateau scheduler (based on val accuracy)
 - Save best model (by val accuracy)
 - Final test evaluation (accuracy, confusion matrix, classification report)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# USER CONFIGURATION
# ----------------------------
DATA_DIR = r"C:\ML\CNN\chest_xray\chest_xray"  # <-- set this to your dataset folder
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 4       # set to 0 if you run into Windows spawn issues
PIN_MEMORY = True
MODEL_SAVE_PATH = "best_resnet18_chest_xray.pth"

# ----------------------------
# DEVICE SETUP
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Optional: speed up cudnn for fixed-size inputs
torch.backends.cudnn.benchmark = True

# ----------------------------
# TRANSFORMS
# ----------------------------
# Train transforms — include augmentations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation / Test transforms — deterministic
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# DATASETS & DATALOADERS
# ----------------------------
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")
test_dir  = os.path.join(DATA_DIR, "test")

# Verify directories exist
for p in (train_dir, val_dir, test_dir):
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Required folder not found: {p}")

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(root=val_dir, transform=val_transforms)
test_dataset  = datasets.ImageFolder(root=test_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print(f"Classes: {train_dataset.classes}")
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# ----------------------------
# MODEL (Pretrained ResNet18)
# ----------------------------
# Use torchvision's weights API (works for modern torchvision). If yours differs, replace accordingly.
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except Exception:
    # fallback for older torchvision versions:
    model = models.resnet18(pretrained=True)

# Replace final fully-connected layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

# ----------------------------
# LOSS, OPTIMIZER, SCHEDULER, SCALER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Scheduler will reduce LR if validation accuracy stops improving (use 'max' since we monitor accuracy)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
scaler = torch.amp.GradScaler()  # For mixed precision training

# ----------------------------
# UTILS: train_one_epoch, validate
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # backward + optimize with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=running_loss / total, acc=running_corrects / total)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):

                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

            pbar.set_postfix(loss=running_loss / total, acc=running_corrects / total)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return epoch_loss, epoch_acc, all_preds, all_targets

# ----------------------------
# MAIN TRAINING LOOP
# ----------------------------
def train_and_evaluate(num_epochs=NUM_EPOCHS):
    best_val_acc = 0.0
    best_epoch = -1
    # Lists to store metrics for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Scheduler step on validation accuracy (we want to maximize accuracy)
        scheduler.step(val_acc)

        epoch_time = time.time() - t0
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, MODEL_SAVE_PATH)
            print(f"--> New best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining finished. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    # ===== Plot curves after training =====
    epochs_range = range(1, num_epochs + 1)

    # Loss plot
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_losses, 'b', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(10,5))
    plt.plot(epochs_range, train_accs, 'b', label='Training Accuracy')
    plt.plot(epochs_range, val_accs, 'r', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_val_acc, best_epoch


# ----------------------------
# FINAL TEST EVALUATION
# ----------------------------
def test_final(model, weights_path=MODEL_SAVE_PATH):
    # Load best weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion)

    print("\n===== Test Results =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Print class-wise report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=train_dataset.classes, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    best_val_acc, best_epoch = train_and_evaluate(num_epochs=NUM_EPOCHS)
    test_final(model, MODEL_SAVE_PATH)
