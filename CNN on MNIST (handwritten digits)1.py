#CNN on MNIST (handwritten digits)
'''Works for any input image size (e.g. 28×28 MNIST or 32×32 CIFAR10).

If you add/remove conv/pool layers, you don’t need to recalc 64*7*7.

Much safer for experimenting'''

import torch
from torch import nn,optim
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# -------------------
# 1. Dataset + Loader
# -------------------

transform = transforms.Compose([transforms.ToTensor(),  # Convert image → Tensor [0,1]
                                transforms.Normalize((0.5),(0.5))] # Normalize (mean=0.5, std=0.5)
                               )

train_dataset = datasets.MNIST(root='C://ML//CNN//data',train=True,transform=transform,download=True)
test_dataset  = datasets.MNIST(root='C://ML//CNN//data',train=False,transform=transform,download=True)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)

test_loader  = DataLoader(test_dataset,batch_size=64,shuffle=True)
image,labels = test_dataset[3]
print('Image shapme:',image.shape,'Label is:',labels)
# -------------------
# 2. CNN Model
# -------------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # (1 → 32)
        self.pool = nn.MaxPool2d(2, 2)                # halve H,W
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32 → 64)

        # auto-calc flattened size
        dummy_input = torch.zeros(1, 1, 28, 28)
        print(dummy_input)
        out = self._forward_conv(dummy_input)
        print('out:\n',out)
        n_features = out.view(1, -1).size(1)
        print('n_features:',n_features)

        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 → 7x7
        return x
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)              # flatten       
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
'''H_in = 28   (height = width = 28)
After Conv1 (kernel=3, stride=1, padding=1):
H_out = (H_in - kernel + 2*padding)/stride + 1
       = (28 - 3 + 2*1)/1 + 1
       = 28
So output height/width = 28
(meaning size doesn’t change when padding=1 and stride=1).

After MaxPool2d(2,2):

H_out = H_in / 2
       = 28 / 2
       = 14
After second conv (padding=1, stride=1):
H_out = 14
After second pool (2×2):
H_out = 14 / 2 = 7
✅ So H_in is just the current spatial dimension (height or width) of the feature map that is fed into the layer.
Each layer transforms it into H_out based on kernel, stride, and padding.
'''


    

# -------------------
# 3. Training
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # -------------------
# 4. Evaluation
# -------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")

# -------------------
# 5. Prediction on Single Image
# -------------------
import matplotlib.pyplot as plt

# get one sample from test dataset
image, label = test_dataset[1]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True Label: {label}")
plt.show()

# prepare image for model
image = image.unsqueeze(0).to(device)   # add batch dim [1,1,28,28]
output = model(image)
_, predicted = torch.max(output, 1)
print(f"Predicted Label: {predicted.item()}")


'''CNN layers output a 4D tensor:
After convolution + pooling, the shape of x is:
[batch_size, channels, height, width]
Example:
For MNIST (28×28 grayscale image):
x.shape = [64, 64, 7, 7]  
# batch of 64 images, 64 channels, 7×7 feature maps
Fully connected (Linear) layers expect 2D input:
Shape:
[batch_size, num_features]
That means we must "flatten" each image into a 1D vector before feeding 
it into the dense layer.
.view(x.size(0), -1):
x.size(0) = batch size (e.g., 64)
-1 tells PyTorch: “figure out the rest automatically”.
So the 3D feature map [channels, height, width] is flattened into one dimension.
Example:
Input shape  = [64, 64, 7, 7]
After view   = [64, 3136]   # because 64*7*7 = 3136
Now, you can feed it into:
self.fc1 = nn.Linear(64 * 7 * 7, 128)

✅ Equivalent ways

Using .flatten:

x = torch.flatten(x, 1)  # flatten from 2nd dim onwards


Using .reshape (similar to view but safer with non-contiguous tensors):

x = x.reshape(x.size(0), -1)


So in short:
x = x.view(x.size(0), -1) reshapes the convolutional feature maps into a flat vector per image, ready for the dense layer.

CNN Flow Diagram (Text-based):

Input Image (1, 28, 28)
   │
   ▼
Conv2d(1 → 32, 3×3, padding=1)
   │
   └──> Output shape: (32, 28, 28)
        → 32 feature maps (edges, corners, textures)

   ▼
MaxPool2d(2×2)
   │
   └──> Output shape: (32, 14, 14)
        → each feature map shrinks, key info kept

   ▼
Conv2d(32 → 64, 3×3, padding=1)
   │
   └──> Output shape: (64, 14, 14)
        → higher-level features (shapes, strokes)

   ▼
MaxPool2d(2×2)
   │
   └──> Output shape: (64, 7, 7)
        → compact but rich representation

   ▼
Flatten
   │
   └──> (64 × 7 × 7 = 3136 features)

   ▼
Fully Connected Layer (3136 → 128)
   │
   ▼
Fully Connected Layer (128 → 10)
   │
   ▼
Softmax → Probabilities for digits 0–9


'''