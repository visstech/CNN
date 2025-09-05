#CNN on MNIST (handwritten digits)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),              # Convert image → Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

trainset = torchvision.datasets.MNIST(root='C://ML//CNN//data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='C://ML//CNN//data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# -------------------------------
# 2. Define CNN Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # input=1ch, output=32ch
        self.pool = nn.MaxPool2d(2, 2)                          # downsample by 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after pooling
        self.fc2 = nn.Linear(128, 10)          # 10 classes (digits)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # conv1 -> relu -> pool
        x = self.pool(torch.relu(self.conv2(x)))  # conv2 -> relu -> pool
        x = x.view(-1, 64 * 7 * 7)                # flatten
        x = torch.relu(self.fc1(x))               # FC layer
        x = self.fc2(x)                           # logits
        return x

model = SimpleCNN()

# -------------------------------
# 3. Training Setup
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 4. Train the Model
# -------------------------------
epochs = 2
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)           # forward pass
        loss = criterion(outputs, labels) # compute loss
        loss.backward()                   # backprop
        optimizer.step()                  # update weights
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

print("Training done ✅")

# -------------------------------
# 5. Evaluate on Test Data
# -------------------------------
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)   # <-- same as in tabular case
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# -------------------------------
# 6. Predict a Single Image
# -------------------------------
import random
image, label = random.choice(testset)   # pick a random test image

# Add batch dimension [1, 1, 28, 28]
image_tensor = image.unsqueeze(0)

# Model prediction
with torch.no_grad():
    output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True: {label}, Predicted: {predicted_class.item()}")
plt.show()
