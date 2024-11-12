import torch.nn as nn
from torchvision import models
from config import *
    
def load_pretrained_model():
    """Load pretrained ResNet18 model and adjust the final layer"""
    pretrained_model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    num_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_features, args.num_classes) # Adjust final layer
    return pretrained_model
