import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    # log_files = {
    #     "Baseline": "train_log.json",
    #     "LR Decrease": "train_log_LRdec.json",
    #     "LR Increase": "train_log_LRinc.json",
    # }

    log_files = {
        "Baseline": "train_log.json",
        "BS = 32": "train_log_BSdec.json",
        "BS = 128": "train_log_BSinc.json",
    }

    # Create subplots for Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")

    for label, log_file in log_files.items():
        with open(f"/home/aditis/ML/Assignment2/training_logs/{log_file}", "r") as f:
            train_log = json.load(f)
        train_losses = train_log["loss"]
        train_accuracies = train_log["accuracy"]
        epochs = range(1, len(train_losses) + 1)

        ax1.plot(epochs, train_losses, label=label)
        ax2.plot(epochs, train_accuracies, label=label)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig("/home/aditis/ML/Assignment2/plots/train_loss_accuracy_BScomparison.png")
    plt.show()
if __name__ == "__main__":
    main()