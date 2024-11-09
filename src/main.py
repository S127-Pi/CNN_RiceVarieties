# %%
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import numpy as np
import os
import json
import pandas as pd
from collections import Counter

from config import args
from earlystopping import EarlyStopping
from utils import *
from model import CNNModel
from dataset import *

def train(model, device):
    set_seed(42)
    train_set = CustomImageFolder(root=args.train_dir)
    train_size = int(0.8 * len(train_set)) 
    validation_size = len(train_set) - train_size
    train_set, validation_set = random_split(train_set, [train_size, validation_size])
    train_set, validation_set = TransformedDataset(train_set), TransformedDataset(validation_set)

    targets = np.array([label for _, label, _ in train_set])
    # Calculate the frequency of each class
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count  # Inverse of class frequencies

    # Assign weights to each sample based on its class
    samples_weight = np.array([weight[int(t)] for t in targets])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_loader = DataLoader(train_set, sampler=sampler,
                              batch_size=args.batch_size, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    checkpoint = {}
    num_epochs = args.epochs
    best_accuracy = 0.0
    best_state_dict = None
    train_epoch_accuracy = []
    validation_epoch_accuracy = []
    for epoch in range(1, num_epochs+1):
        train_accuracy = 0.0
        train_loss = 0.0
        validation_accuracy= 0.0
        validation_loss = 0.0
        total_size = 0
        class_counter = Counter()
        
        model.train()
        for i, (images, labels, _) in tqdm(enumerate(train_loader), desc="Mini-Batch"):
            
            class_counter.update(labels.cpu().numpy())
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_size += labels.size(0)
            
            train_loss += outputs.shape[0] * loss.item()
            _, predictions = torch.max(outputs.data, 1)
            train_accuracy += int(torch.sum(predictions==labels.data))

        train_accuracy = train_accuracy/total_size
        train_loss = train_loss/total_size
        train_epoch_accuracy.append(train_accuracy)
        print(class_counter)
        
        model.eval()
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(validation_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                validation_loss += outputs.shape[0] * loss.item()
                _, predictions = torch.max(outputs.data, 1)
                validation_accuracy += int(torch.sum(predictions==labels.data))
            validation_accuracy = validation_accuracy/validation_size
            validation_loss = validation_loss/validation_size
            validation_epoch_accuracy.append(validation_accuracy)
            
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
                checkpoint["Training accuracy"], checkpoint["Validation accuracy"] = train_accuracy, validation_accuracy

        print(f"{epoch=},{train_accuracy=},{train_loss=},{validation_accuracy=},{validation_loss=}, {best_accuracy=}")

        # early stopping
        early_stopping(train_loss, validation_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch:{epoch}\n {checkpoint=}")
            break
    try:
        plot_accuracy(args.epochs, train_epoch_accuracy, validation_epoch_accuracy )
    except Exception as e:
        print(e)

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoing)
    torch.save(best_state_dict, f"{args.checkpoint}/CNNmodel.pt")

    try:
        with open(f'{args.checkpoint}/checkpoint_train.txt', 'w') as file:
            file.write(json.dumps(checkpoint)) 
    except Exception as e:
        print(e)

def test(model, device):
    set_seed(42)
    model = load_model(model).to(device)
    test_dataset = CustomImageFolder(root=args.test_dir)
    test_dataset = TransformedDataset(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    checkpoint = {}
    results = []
    test_accuracy=0.0

    model.eval()
    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            _, predictions = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(predictions==labels.data))
            
            for path, label, prediction in zip(paths, labels.cpu(), predictions.cpu()):
                results.append({
                    'path': path,
                    'label': label.item(),
                    'prediction': prediction.item()
                })
        test_accuracy = test_accuracy/len(test_dataset)

        print(f"{test_accuracy=}")
        checkpoint["Test accuracy"] = test_accuracy
        df_results = pd.DataFrame(results)

    try:
        with open(f'{args.checkpoint}/checkpoint_test.txt', 'w') as file:
            file.write(json.dumps(checkpoint)) 
        df_results.to_csv(f'{args.checkpoint}/predictions.csv', index=False)
        print("Checkpoint and predictions saved successfully.")
    except Exception as e:
        print(e)

if __name__ == '__main__':
    device = get_device()
    model = CNNModel(num_classes=4).to(device)
    print(device)
    print(args)
    if (args.train):
        train(model, device)
    if (args.test):
        test(model, device)