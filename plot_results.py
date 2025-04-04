import json
import matplotlib.pyplot as plt

def plot_loss_accuracy(json_file: str, plot_file: str, test_accuracy: float) -> None:
    
    with open(json_file, "r") as f:
        log = json.load(f)

    train_losses = log["loss"]
    train_accuracies = log["accuracy"]


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color="blue")
    plt.axhline(y=test_accuracy, color='green', linestyle='--', label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.legend()

    plt.savefig(plot_file)

