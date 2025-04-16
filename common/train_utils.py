import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from common.utils import get_device
from typing import Tuple
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


def get_gradients(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.norm().item()
    return grads

def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    grad_history = [] 
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        #loss = F.nll_loss(output, target)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  
        loss = criterion(output, target) 

        optimizer.zero_grad()
        loss.backward()
        grad_history.append(get_gradients(model))
        optimizer.step()


        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(1) == target).sum().item()
        total_samples += data.size(0)

    return total_loss / total_samples, total_correct / total_samples , grad_history


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #loss = F.nll_loss(output, target)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  
            loss = criterion(output, target) 

            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        epochs: int = 10,
        log_file: str = "train_log.json",
        model_file: str = "model.pth",
) -> None:
    print("Training...")
    model.to(get_device())

    train_log = {
        "loss": [],
        "accuracy": [],
        "gradients": []
    }

    for epoch in range(epochs):
        train_loss, train_acc, gradients = train_epoch(model, data_loader, optimizer)
        #train_loss, train_acc = train_epoch(model, data_loader, optimizer)
        train_log["loss"].append(train_loss)
        train_log["accuracy"].append(train_acc)
        train_log["gradients"].append(gradients)
        print(f"Epoch {epoch + 1} / {epochs} | " +
              f"Train Loss: {train_loss:.4f} | " +
              f"Train Acc: {train_acc:.4f}")
        
    # Save the training log to a JSON file
    with open(f"/home/aditis/ML/Assignment2/training_logs/{log_file}", "w") as f:
        json.dump(train_log, f)

    # Save the model
    torch.save(model.state_dict(), f"/home/aditis/ML/Assignment2/models/{model_file}")
    print("Training complete!")

    plot_gradients(train_log["gradients"], "gradients_110.png")


def plot_gradients(grad_history, save_path="gradients.png"):

    all_grads = [g for epoch in grad_history for g in epoch]

  
    layer_names = list(all_grads[0].keys())

    grad_matrix = np.array([[g[name] for name in layer_names] for g in all_grads])

    plt.figure(figsize=(14, 6))
    for i, name in enumerate(layer_names):
        plt.plot(grad_matrix[:, i], label=name)

    plt.xlabel("Batch Iteration")
    plt.ylabel("Gradient Norm")
    plt.yscale("log")
    plt.title("Layer-wise Gradient Norms During Training")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    os.makedirs("/home/aditis/ML/Assignment2/plots", exist_ok=True)
    plt.savefig(f"/home/aditis/ML/Assignment2/plots/{save_path}")
    plt.close()

