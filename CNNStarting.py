import torch
import torchaudio
import torchvision
from torch import nn,optim #neural network 

print('torch:',torch.__version__)
print('torchaudio:',torchaudio.__version__)
print('torchvision:',torchvision.__version__)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(3,2)   #fully connected neron 3 input layer and 2 is the output layer or hidden layers
        self.fc2 = nn.Linear(2,1)   #fully connected layer2 2 input neron from the previous layer and 1 output layer.
        # This is the architecture of my nn.
    def forward(self,X): #Forward function this name can not be changed it is coming from nn.Module.
        X = self.fc1(X)
        X = self.fc2(X)
        return X
  
model = SimpleNN()
criterion = nn.MSELoss() # Loss function to check how much my model is making error
optimizer = optim.SGD(model.parameters(),lr=0.001) # optimizer will help me do the wait update.

X = torch.rand(100,3)
y = torch.rand(100,1)
num_epoch = 100
for epoch in range(1,num_epoch+1):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output,y)
    loss.backward() # will find out the error in each neuron
    optimizer.step()
    print(f'{epoch}/{num_epoch}, loss----> {loss.item()}')
  
