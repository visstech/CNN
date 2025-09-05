import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import numpy as np 
import os
import seaborn as sns 
from PIL import Image

data_dir ='C://ML//CNN//chest_xray'
# Transformations: resize, tensor, normalize
transform = transforms.Compose([
                                transforms.Grayscale(),             # Convert to 1 channel (grayscale)
                                transforms.Resize((28,28)),           # Resize to 28x28
                                transforms.ToTensor(),              # Convert to tensor
                                transforms.Normalize((0.5),(0.5))    # Normalize
])

train_data = datasets.ImageFolder(os.path.join(data_dir, "train"),transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, "test"),transform=transform)

#Data Loader
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True)

print('Classes:',train_data.classes)


# -------------------------
# 1. Improved CNN Model
# -------------------------
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 🔹 Use dummy input to auto-calc flattened size
        dummy_input = torch.zeros(1, 1, 28, 28)   # (batch, channels, H, W)
        out = self._forward_conv(dummy_input)
        n_features = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(n_features, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------------
# 2. Training Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)

criterion = nn.CrossEntropyLoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------------
# 3. Training Loop (example)
# -------------------------
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%")

    return model

trained_model = train_model(model, train_loader, test_loader, epochs=10)

def test_model(model, test_loader):
        model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

       # Compute accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"\nTest Accuracy: {acc*100:.2f}%")

    # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=test_loader.dataset.classes))
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

test_model(model, test_loader) 

#Prediction on single image 
def predict_and_plot(image_path, model, transform, class_names):
    # Load image
    img = Image.open(image_path).convert("L")   # "L" = grayscale

    # Keep a copy for display (before transforms)
    img_display = img.copy()

    # Apply same transforms as training
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    class_id = predicted.item()
    label = class_names[class_id]

    # Plot the image with prediction
    plt.imshow(img_display, cmap="gray")
    plt.title(f"Prediction: {label}", fontsize=14, color="blue")
    plt.axis("off")
    plt.show()

    return label 

class_names = train_data.classes  # ["NORMAL", "PNEUMONIA"]

result = predict_and_plot(
    "C://ML//CNN//chest_xray//chest_xray//val//PNEUMONIA//person1946_bacteria_4874.jpeg",
    model,
    transform,
    class_names
)

print("Final Prediction:", result)

