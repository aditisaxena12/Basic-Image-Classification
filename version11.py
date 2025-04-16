import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *
from plot_results import plot_loss_accuracy
from ResNet import ResNet


def main() -> None:
    # Load the data
    train_loader, test_loader = get_data('cifar10', batch_size=100)

    # Create a model
    model = ResNet(3)
    print("Model Parameter Count:", sum(p.numel() for p in model.parameters()))

    # Create an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001 , momentum=0.9, weight_decay=0.0001)

    # Train the model
    train(model, train_loader, optimizer, epochs=100, log_file="train_log_resnet20_100.json", model_file="model_resnet20_100.pth")

    # Evaluate the model
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    plot_loss_accuracy("/home/aditis/ML/Assignment2/training_logs/train_log_resnet20_100.json", "/home/aditis/ML/Assignment2/plots/train_loss_accuracy_resnet20_100.png", test_acc)
