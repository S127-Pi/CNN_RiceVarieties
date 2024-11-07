import torch
import torch.nn as nn
import torchvision.transforms as transforms

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
        
        self.fc1 = nn.Linear(in_features=32*75*75, out_features=num_classes)
        
    def forward(self, input):
        output = self.relu1(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = self.relu2(self.bn2(self.conv2(output)))         
        output = self.relu3(self.bn3(self.conv3(output)))
        
        output = output.view(-1, 32*75*75)
        output = self.fc1(output)
        
        return output 