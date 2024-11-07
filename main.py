# %%
from PIL import Image
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed for reproducibility
set_seed(42)

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root="assignment_train"):
        super().__init__(root=root)
        self.root = root
        self.transform = transforms.Compose([transforms.Resize((150, 150)), #(224,224)
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], 
                                                                  [0.5, 0.5, 0.5])
                                            ])
        
        self.labels_dict = {"breed9": 0, "breed28": 1, "breed41": 2, "other": 3}
        
    def __getitem__(self, index):
        # Original image and label
        image, label = super().__getitem__(index)
        
        # Obtain the original class name
        class_name = self.classes[label] 
        
        # Custom labeling
        custom_label = self.labels_dict.get(class_name, self.labels_dict["other"])
        
        return image, custom_label
    

train_dataset = CustomImageFolder(root="assignment_train")
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataset = CustomImageFolder(root="assignment_test")
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# print(len(train_dataset))
# print(len(test_dataset))

class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        """
        Output size after convolution filter
        Output size= (w-f+2P)/s + 1
        w: width
        f: kernel size
        P: padding
        s: stride
        
        Input shape= (256, 3, 150, 150)
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, 
                               kernel_size=3, stride=1, padding=1)
        # New shape = (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # shape = (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        
        # shape = (256, 12, 150, 150)
        self.pool = nn.MaxPool2d(kernel_size=2) 
        # Reduce the image size by factor 2
        # Shape(256, 12, 75, 75)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, 
                               kernel_size=3, stride=1, padding=1)
        # New shape = (256, 20, 75, 75)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        self.relu2 = nn.ReLU()
        # shape = (256, 20, 75, 75)
            
        # New shape = (256, 32, 75, 75)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)# New shape = (256, 32, 75, 75)
        self.relu3 = nn.ReLU() # shape = (256, 32, 75, 75)
        
        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)
        
    def forward(self, input):
        output = self.relu1(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = self.relu2(self.bn2(self.conv2(output)))         
        output = self.relu3(self.bn3(self.conv3(output)))
        
        output = output.view(-1, 32*75*75)
        output = self.fc(output)
        
        return output        
# %%
model = CNNModel(num_classes=4).to(device)

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()
num_epochs = 10

best_accuracy = 0.0
best_state_dict = None
for epoch in range(1, num_epochs+1):
    
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(train_loader), desc="Mini-Batch"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.cpu().data*images.size(0)
        _, predictions = torch.max(outputs.data, 1)
        
        train_accuracy += int(torch.sum(predictions==labels.data))
    train_accuracy = train_accuracy/len(train_dataset)
    train_loss = train_loss/len(train_dataset)
    
    model.eval()
    test_accuracy=0.0
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs=model(images)
        _, predictions = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(predictions==labels.data))
    test_accuracy = test_accuracy/len(test_dataset)
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_state_dict = model.state_dict()
    
    print(f"{epoch=},{train_accuracy=},{train_loss=},{test_accuracy=},{best_accuracy=}")
os.mkdir("checkpoint")
torch.save(best_state_dict, "CNNmodel.pt")
    
    
    