import random
import numpy as np
import torch
import os
import warnings
from config import *


device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(model):
    if os.path.exists(f"{args.checkpoint}/CNNmodel.pt"):
        # Load the model if the file exists
        model.load_state_dict(torch.load(f"{args.checkpoint}/CNNmodel.pt"))
        model.to(device)
        model.eval()
        print("Model loaded")
        return model
    else:
        warnings.warn("Checkpoint file not found.", ResourceWarning)
        return model
