# train_pneumonia.py
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- settings --------
DATA_DIR = "C:\\ML\\CNN\\chest_xray\\chest_xray"                # should contain train/val/test folders
BATCH_SIZE = 32
NUM_EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IMG_SIZE = 224
MODEL_SAVE = "weights/best_model.pth"
os.makedirs("weights", exist_ok=True)

# -------- transforms --------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]) if True else transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]) if True else transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# If images are single-channel, torchvision datasets will convert; adjust normalization as needed.

# -------- datasets & loaders --------
train_ds = datasets.ImageFolder(Path(DATA_DIR)/"train", transform=train_tfms)
val_ds = datasets.ImageFolder(Path(DATA_DIR)/"val", transform=val_tfms)
test_ds = datasets.ImageFolder(Path(DATA_DIR)/"test", transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -------- model (small CNN from scratch) --------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)

# Optional: try transfer learning (uncomment)
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
# model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

# -------- training & evaluation helpers --------
def train_one_epoch(model, loader, opt, crit):
    model.train()
    running_loss = 0.0
    preds = []
    targets = []
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        out = model(imgs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item() * imgs.size(0)
        preds.append(out.detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    preds = np.argmax(np.vstack(preds), axis=1)
    targets = np.concatenate(targets)
    return epoch_loss, preds, targets

def evaluate(model, loader, crit):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = crit(out, labels)
            running_loss += loss.item() * imgs.size(0)
            preds.append(out.cpu().numpy())
            targets.append(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    preds = np.argmax(np.vstack(preds), axis=1)
    targets = np.concatenate(targets)
    return epoch_loss, preds, targets

if __name__ == "__main__":
    best_val_loss = 1e9
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        tr_loss, tr_preds, tr_targets = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion)
        print(f" train_loss: {tr_loss:.4f} | val_loss: {val_loss:.4f}")
        # metrics
        print(" Val classification report:")
        print(classification_report(val_targets, val_preds, target_names=train_ds.classes, digits=4))
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            print(" Saved best model.")

    # final test eval
    model.load_state_dict(torch.load(MODEL_SAVE))
    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion)
    print("Test report:")
    print(classification_report(test_targets, test_preds, target_names=test_ds.classes, digits=4))
    print("Confusion matrix:\n", confusion_matrix(test_targets, test_preds))
