import random
import numpy as np
import torch
import os
import warnings
from config import *
import matplotlib.pyplot as plt

def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else
                        "cpu")
    return device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_accuracy(epochs, train_accuracy, validation_accuracy):
    epochs = range(1, epochs + 1)
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f'{args.checkpoint}/accuracy_plot.png', format='png', dpi=300)

def load_model(model):
    device = get_device()
    if os.path.exists(f"{args.checkpoint}/model.pt"):
        # Load the model if the file exists
        model.load_state_dict(torch.load(f"{args.checkpoint}/model.pt"))
        model.to(device)
        model.eval()
        print("Model loaded")
        return model
    else:
        warnings.warn("Checkpoint file not found.", ResourceWarning)
        return model

def load_hyperparameter():
    if os.path.exists(f"{args.checkpoint}/checkpoint_hyperparams.txt"):
        with open(f"{args.checkpoint}/checkpoint_hyperparams.txt", 'r') as file:
                content = file.read()
                dict_hyper = eval(content)
                batch_size , lr, momentum, weight_decay = dict_hyper.values()
                args.batch_size = int(batch_size)
                args.lr = float(lr)
                args.weight_decay = float(weight_decay)
                args.momentum = float(momentum)
                
                print("Hyperparameters loaded")
                print(f"{batch_size=}, {lr=}, {momentum=}, {weight_decay=}")
    else:
        print("Hyperparameters not loaded")