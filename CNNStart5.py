# Multi-class classification (Iris dataset) with 2 hidden layers
import torch
import pandas as pd
import seaborn as sns
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = sns.load_dataset('iris')
data['species'] = data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Features and labels
X = torch.tensor(data.drop(['species'], axis=1).values, dtype=torch.float32)
y = torch.tensor(data['species'].values, dtype=torch.long)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define simple neural network (2 hidden layers)
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)   # hidden layer 1
        self.fc2 = nn.Linear(16, 8)           # hidden layer 2
        self.fc3 = nn.Linear(8, output_dim)   # output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # CrossEntropyLoss expects raw logits
        return x

# Model, loss, optimizer
input_dim = X.shape[1]
output_dim = len(torch.unique(y))
model = SimpleNN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
for epoch in range(1, num_epochs+1):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss = {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    logits = model(X_test)
    preds = torch.argmax(logits, dim=1)

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['setosa','versicolor','virginica']))

# Confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=['setosa','versicolor','virginica'],
            yticklabels=['setosa','versicolor','virginica'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
