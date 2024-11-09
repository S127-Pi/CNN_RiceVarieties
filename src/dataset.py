
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from config import *

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root=args.train_dir):
        super().__init__(root=root)
        self.labels_dict = {"breed9": 0, "breed28": 1, "breed41": 2, "other": 3}
        
    def __getitem__(self, index):
        # Original image and label
        image, label = super().__getitem__(index)
        # Obtain the original class name
        class_name = self.classes[label] 
        # Custom labeling
        custom_label = self.labels_dict.get(class_name, self.labels_dict["other"])
        image_path = self.samples[index][0]
        
        return image, custom_label, image_path
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.Resize((150, 150)), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ])

    def __getitem__(self, index):
        image, label, image_path = self.dataset[index]
        # Apply transformation to the image if specified
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

    def __len__(self):
        return len(self.dataset)