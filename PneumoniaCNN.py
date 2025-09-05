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



''' Steps

Download dataset (from Kaggle → chest_xray folder: train/test/val)

Preprocess images (resize, tensor, normalize)

Create DataLoader

Train CNN model

Evaluate performance (accuracy, confusion matrix)'''

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # grayscale input
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)       
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        
        # Auto-calc flattened size using dummy input
        dummy_input = torch.zeros(1, 1, 28, 28)   # batch=1, 1 channel, 28x28
        out = self._forward_conv(dummy_input)
        n_features = out.view(1, -1).size(1)
        print(f"Flattened features: {n_features}")
        
        # Fully connected layers
        self.fc1   = nn.Linear(n_features, 256)   
        self.fc2   = nn.Linear(256, 2)   # Normal vs Pneumonia
    
    # Conv forward helper (no linear yet)
    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 → 7x7
        x = self.pool(F.relu(self.conv3(x)))   # 7x7   → 3x3
        return x
    
    # Full forward pass
    def forward(self, x):
        x = self._forward_conv(x)          # conv layers
        x = x.view(x.size(0), -1)          # flatten (batch_size, n_features)
        x = F.relu(self.fc1(x))            # FC1
        x = self.fc2(x)                    # FC2 (logits for 2 classes)
        return x


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


#Training setup

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epoch = 10
for epoch in range(1,num_epoch+1):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss   = criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
all_preds,all_labels = [],[]
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device),labels.to(device)
        output = model(images)
        preds = torch.argmax(output,dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
print('\n Classification Report:',classification_report(all_labels,all_preds,target_names=train_data.classes))

cm = confusion_matrix(all_labels,all_preds)
#sns.heatmap(cm,annot=True,fmt='d',cmap='blue',xticklabels=train_data.classes,yticklabels=train_data.classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()