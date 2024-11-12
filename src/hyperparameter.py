import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from torcheval.metrics.functional import multiclass_f1_score

from config import *
from dataset import *
from utils import *
from model import *


def objective(trial):
    set_seed(42)
    # Hyperparameters
    batch_size = trial.suggest_int('batch_size', 64, 128 )
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    # DataLoader
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
                              batch_size=batch_size, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Model
    model = load_pretrained_model()
    device = get_device()
    model.to(device)

    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Training
    model.train()
    for epoch in tqdm(range(3), desc="epoch"):
        running_loss = 0.0
        for (images, labels, _) in tqdm(train_loader, desc="Mini-batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels, _) in tqdm(enumerate(validation_loader), desc="Mini-batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    validation_f1_score = multiclass_f1_score(torch.tensor(all_predictions),
                                              torch.tensor(all_labels),
                                              num_classes=4,
                                              average="macro").item()
    return validation_f1_score


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', study_name="Hyperparameter Optimization")
    study.optimize(objective, n_trials=3)

    print("Best hyperparameters:", study.best_params)
    print("Best F1-score:", study.best_value)

    try:
        with open(f'{args.checkpoint}/checkpoint_hyperparams.txt', 'w') as file:
            file.write(json.dumps(study.best_params)) 
    except Exception as e:
        print(e)