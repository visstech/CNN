# Multi class classification with Confusion Matrix (Fixed)
import torch 
import pandas as pd 
from torch import nn,optim
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
data = sns.load_dataset('iris')
data['species'] = data['species'].map({'setosa':0,'versicolor':1, 'virginica':2})

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(['species'],axis=1))

X = torch.tensor(X_scaled, dtype=torch.float32)
y = torch.tensor(data['species'].values, dtype=torch.long)

# Model
class MultiNN(nn.Module):
    def __init__(self, num_neuron, output_neuron):
        super(MultiNN,self).__init__()
        self.fc1 = nn.Linear(num_neuron,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,output_neuron)
        
    def forward(self,X):
        X = torch.relu(self.fc1(X))   
        X = torch.relu(self.fc2(X))
        X = self.fc3(X)  # raw logits
        return X

input_neurons = X.shape[1]  
output_neurons = len(y.unique())

model = MultiNN(input_neurons,output_neurons)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
num_epoch = 500
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        _, predicted = torch.max(output, 1)
        acc = (predicted == y).sum().item() / len(y)
        print(f'Epoch {epoch}/{num_epoch}, Loss = {loss.item():.4f}, Accuracy = {acc:.2f}')

# Prediction
with torch.no_grad():
    logits = model(X)
    predictions = torch.argmax(logits, dim=1)

# Confusion Matrix
cm = confusion_matrix(y.numpy(), predictions.numpy())
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['setosa','versicolor','virginica'],
            yticklabels=['setosa','versicolor','virginica'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Iris Classification")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y.numpy(), predictions.numpy(),
                            target_names=['setosa','versicolor','virginica']))
