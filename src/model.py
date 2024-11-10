import torch.nn as nn
from torchvision import models
    
def _load_pretrained_model():
    """Load pretrained ResNet18 model"""
    pretrained_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')

    return pretrained_model

pretrained_model = _load_pretrained_model()