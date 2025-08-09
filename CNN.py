#Convolutional Neural Networks (CNNs)
import torchvision
from torchvision import datasets,transforms
import torch 
import matplotlib.pyplot as plt 
from PIL import Image
from torch import nn,optim
from torch.utils.data import DataLoader
#Reading image data 
## Loading datasets from torchvision dataset to our local folder.
train_data = torchvision.datasets.CIFAR100(root='C:\\ML\\CNN',train=True,transform=torchvision.transforms.ToTensor(),download=True)
#To download the dataset to c:ML\CNN folder.
train_data =  DataLoader(train_data,batch_size=64) 
print(train_data)

##Loading the dataset from local environment 
#dataset which we have in our local 
dataset = datasets.ImageFolder(root='C:\\ML\\CNN\\train_cancer',transform=transforms.ToTensor())
print(dataset)
img,label = dataset[0]
img = img.permute(1,2,0)

''' # Original tensor shape (Batch, Height, Width, Channels)
tensor = torch.randn(32, 64, 64, 3)  # Channels Last
# Permute to (Batch, Channels, Height, Width)
tensor = tensor.permute(0, 3, 1, 2) 
the permute function is typically employed to rearrange the order of dimensions in a tensor. 
This is important when the data's shape does not align with the expected input format of a CNN layer or for tasks like reshaping data for specific operations.
'''
plt.imshow(img)
plt.title(label=label)
plt.show()

print(img) # it will be in the form of tensure data means numberical data. 

img = Image.open('c:\\ML\\Blue-Lotus.jpg')
transformation = transforms.ToTensor()
tensor_img = transformation(img) # transforming image to tensor data 
img = torch.tensor(tensor_img,dtype=torch.float32)
img = (img.permute(1,2,0)).numpy()
plt.imshow(img)
plt.show()


#Converting image to one channel meaning only one colored image
img = Image.open('c:\\ML\\Blue-Lotus.jpg')

transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

tensor_img = transformation(img) # transforming image to tensor data 
img = torch.tensor(tensor_img,dtype=torch.float32)
img = (img.permute(1,2,0)).numpy() # Here 1 is the number of channel, 2 = height, 0 =width 
plt.imshow(img,cmap='gray')
plt.show()

print(tensor_img)
print(tensor_img.shape) # Here the shape is ([1,835,1200]) 1 is the channel, 835 is height, 1200 is represent width 
#whenever we want to apply the filter we need data in the form of [batch,channel,height,width]
# to convert that we need to use 
tensor_img = tensor_img.unsqueeze(0) # it will convert the image data in to [batch,channel,height,width] this format which we can apply filter 
print(tensor_img.shape) # ([1, 1, 835, 1200]) now the shape 1 = batch size, 1 = channel, 835 height, 1200 
# batch meaning number of images processed at a time, we can not process all the images at a time, so we apply batch 10 batch meaning 10 images at a time.
sobal = torch.tensor([ # 3 * 3 Filter  sobal is a standard filter by some of the researchers these values are default
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
],dtype=torch.float32).unsqueeze(0).unsqueeze(0) # first the sobal shape will be [hight,width] so we apply unsqueeze to convert it in to [channe1,height,width]
# second unsqueeze for [batch,channel,height,width] then only we can apply the filter.

# applying the filter now
output_image = torch.nn.functional.conv2d(tensor_img,sobal,stride=1) # Here image is which we are going to apply filter sobal is the filter
print(output_image.shape) # now the shape is ([1, 1, 833, 1198]) batch,channel,height,width formate
# to show this image in to plt.imshow() we need in the form of [Height,width,channel] so we need to reduce the dimension squeeze() is used
output_image = output_image.squeeze(0)
print('After applying squeeze() the shape is changed :',output_image.shape) # now ([1, 833, 1198]) channel,height,width below permute convert it to height,width,channel format 

output_image = output_image.permute(1,2,0) # Changes to ([833, 1198, 1]) [height,width,channel] which can be shown using plt.imshow()

print('After applying permute the shape is changed :',output_image.shape) # now it can be shown using plt.imshow()
plt.imshow(output_image,cmap='gray')
plt.show()

# to convert all the images into same size 244 * 244 pixels
transformation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root='C:\\ML\\CNN\\train_cancer',transform = transformation)
print(dataset[0])

class CNN_Cancer(nn.Module):
    def __init__(self, num_classes=2):  # Adjust for your dataset
        super(CNN_Cancer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 224 * 224, 512)  # Adjust based on input size
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, num_classes)  # Set num_classes dynamically
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.relu(self.conv2(X))
        X = self.relu(self.conv3(X))
        X = X.view(X.size(0), -1)  # Flatten
        X = self.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X
    
# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the appropriate device
model = CNN_Cancer(num_classes=2)  # Set num_classes based on your dataset
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001)

num_epoch = 10

for epoch in range(1, num_epoch + 1):
    model.train()
    total_loss = 0
    for img, label in train_data:
        # Move data and labels to the appropriate device
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch}/{num_epoch}], Loss: {total_loss:.4f}')
