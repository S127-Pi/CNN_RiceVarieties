
import torchvision.transforms as transforms
from torchvision import datasets

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root="../data/assignment_train"):
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