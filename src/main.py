# %%
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import random_split, DataLoader
import numpy as np
import os
import json

from config import args
from utils import *
from model import CNNModel
from dataset import CustomImageFolder

def train(model):

    set_seed(42)
    train_dataset = CustomImageFolder(root=args.train_dir)
    train_size = int(0.8 * len(train_dataset)) 
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"{train_size=}, {val_size=}")
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    checkpoint = {}
    num_epochs = 10
    best_accuracy = 0.0
    best_state_dict = None
    for epoch in range(1, num_epochs+1):
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        for i, (images, labels, _) in tqdm(enumerate(train_loader), desc="Mini-Batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += outputs.shape[0] * loss.item()
            _, predictions = torch.max(outputs.data, 1)
            
            train_accuracy += int(torch.sum(predictions==labels.data))
        train_accuracy = train_accuracy/train_size
        train_loss = train_loss/len(train_dataset)
        
        model.eval()
        val_accuracy=0.0
        for i, (images, labels, breed) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            _, predictions = torch.max(outputs.data, 1)
            val_accuracy += int(torch.sum(predictions==labels.data))
        val_accuracy = val_accuracy/val_size
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
            checkpoint["Training accuracy"], checkpoint["Validation accuracy"] = train_accuracy, val_accuracy


        print(f"{epoch=},{train_accuracy=},{train_loss=},{val_accuracy=},{best_accuracy=}")


    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoing)
    torch.save(best_state_dict, f"{args.checkpoint}/CNNmodel.pt")
    try:
        with open(f'{args.checkpoint}/checkpoint_train.txt', 'w') as file:
            file.write(json.dumps(checkpoint)) 
    except:
        print("Error")

def test(model):
    set_seed(42)
    model = load_model(model).to(device)
    test_dataset = CustomImageFolder(root=args.test_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    checkpoint = {}
    test_accuracy=0.0

    model.eval()
    for i, (images, labels, _) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs=model(images)
        _, predictions = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(predictions==labels.data))
    test_accuracy = test_accuracy/len(test_dataset)

    print(f"{test_accuracy=}")
    checkpoint["Test accuracy"] = test_accuracy

    try:
        with open(f'{args.checkpoint}/checkpoint_test.txt', 'w') as file:
            file.write(json.dumps(checkpoint)) 
    except:
        print("Error")

if __name__ == '__main__':
    model = CNNModel(num_classes=4).to(device)
    print(device)
    print(args)
    if (args.train):
        train(model)
    if (args.test):
        test(model)