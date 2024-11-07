import torch
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import CNNModel
from dataset import CustomImageFolder

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
model = CNNModel()
test_dataset = CustomImageFolder(root="../assignment_test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

cam_extractor = GradCAM(model, target_layer="conv_layer")  # Specify a convolutional layer
# Forward pass and extract CAM
scores = model(image.unsqueeze(0).to(device))
cam = cam_extractor(class_idx=scores.argmax().item(), scores=scores)

# Plot CAM
plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Original image
plt.imshow(cam[0].cpu().numpy(), cmap="jet", alpha=0.5)  # Overlay heatmap
plt.show()