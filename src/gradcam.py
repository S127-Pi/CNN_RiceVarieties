import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import warnings

from dataset import CustomImageFolder
from model import CNNModel
from utils import *

set_seed(42)
model = CNNModel()

if os.path.exists("../checkpoint/CNNmodel.pt"):
    # Load the model if the file exists
    model.load_state_dict(torch.load("../checkpoint/CNNmodel.pt"))
    model.to(device)
    model.eval()
    print("Model loaded")
else:
    warnings.warn("Checkpoint file not found.", ResourceWarning)

conv_weights = [] 
conv_layers = [] 
for module in model.children():
    if isinstance(module, nn.Conv2d):
        conv_weights.append(module.weight)
        conv_layers.append(module)


test_dataset = CustomImageFolder(root="../data/assignment_test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
label_dict = {"breed9": 0, "breed28": 1, "breed41": 2, "other": 3}
label_set = set()
unique_breed = []

for i, (image, label, breed) in enumerate(test_loader):
    breed = breed[0]
    if len(label_set) == 4:
        break
    if label.item() in label_set:
        continue
    print(label.item())
    print(breed)
    label_set.add(label.item())
    unique_breed.append((image, label, breed))


# plt.imshow(unique_breed[0][0].squeeze(0).permute(1, 2, 0).cpu().numpy())
# plt.show()

feature_maps = []  # List to store feature maps
layer_names = []  # List to store layer names
# for i, (image, label, breed) in enumerate(unique_breed):
#     if label.item() in breed_set:
#         continue
#     print(label.item())
#     breed_set.add(label.item())

#     if i == 2:
#         break
#     image, label = image.to(device), label.to(device)
#     for layer in conv_layers:
#         image = layer(image)
#         feature_maps.append(image)
#         layer_names.append(str(layer))
#     processed_feature_maps = []  # List to store processed feature maps
#     for feature_map in feature_maps:
#         feature_map = feature_map.squeeze(0)  # Remove the batch dimension
#         mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
#         processed_feature_maps.append(mean_feature_map.data.cpu().numpy())
#     # Plot the feature maps
#     fig = plt.figure(figsize=(30, 50))
#     for i in range(len(processed_feature_maps)):
#         ax = fig.add_subplot(5, 4, i + 1)
#         ax.imshow(processed_feature_maps[i])
#         ax.set_title(layer_names[i].split('(')[0], fontsize=30)
    



