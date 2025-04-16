import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

plainnet110 = '/home/aditis/ML/Assignment2/training_logs/train_log_plainnet110_25.json'


def main():

    with open(plainnet110, "r") as f:
        gradients = json.load(f)["gradients"]
    gradients110 = np.array(gradients)

    print(gradients110.shape)

    plot_layerwise_gradients(gradients110, num_groups=10, save_path= "gradient_plainnet110_25.png")
    plot_gradient_group_heatmap(gradients110, num_groups=10, save_path="gradient_groups_plainnet110_25.png")

def plot_layerwise_gradients(grad_history, num_groups=10, save_path="layerwise_gradients.png"):
    all_layers = list(grad_history[0][0].keys())
    all_layers.sort() 

    total_layers = len(all_layers)
    group_size = total_layers // num_groups
    layer_groups = {
        f"Group {i+1}": all_layers[i*group_size : (i+1)*group_size] if i < num_groups - 1 else all_layers[i*group_size:]
        for i in range(num_groups)
    }

    #  Accumulate and average gradients per group
    grouped_data = defaultdict(list)

    for epoch in grad_history:
        epoch_group_grads = defaultdict(list)

        for batch in epoch:
            for group_name, layer_names in layer_groups.items():
                for lname in layer_names:
                    if lname in batch:
                        epoch_group_grads[group_name].append(batch[lname])

        for group_name, values in epoch_group_grads.items():
            grouped_data[group_name].append(np.mean(values) if values else 0.0)

    plt.figure(figsize=(10, 5))
    for group_name, values in grouped_data.items():
        plt.plot(values, label=group_name)

    plt.xlabel("Epoch")
    plt.ylabel("Average Gradient Norm")
    plt.yscale("log")
    plt.title(f"Gradient Norms Across {num_groups} Equal Layer Groups")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/home/aditis/ML/Assignment2/plots/{save_path}")
    plt.close()

def plot_gradient_group_heatmap(grad_history, num_groups=4, save_path="gradient_groups_heatmap.png"):
    epoch_idx = 0  # Change this to select a different epoch
    epoch = grad_history[epoch_idx]

    all_layers = list(epoch[0].keys())
    all_layers.sort()
    total_layers = len(all_layers)
    group_size = total_layers // num_groups

    layer_groups = {
        f"Group {i+1}": all_layers[i * group_size: (i + 1) * group_size] if i < num_groups - 1 else all_layers[i * group_size:]
        for i in range(num_groups)
    }

    # Compute group-wise average gradients per batch
    heatmap_data = []
    for batch in epoch:
        group_values = []
        for group_layers in layer_groups.values():
            grads = [batch[layer] for layer in group_layers if layer in batch]
            mean_grad = np.mean(grads) if grads else 0.0
            group_values.append(mean_grad)
        heatmap_data.append(group_values)

    heatmap_array = np.array(heatmap_data).T  # [groups x batches]
    heatmap_array = np.clip(heatmap_array, 1e-8, None)  # Avoid log(0)

    log_heatmap = np.log10(heatmap_array)

    plt.figure(figsize=(12, 5))
    ax = sns.heatmap(
        log_heatmap,
        annot=False,
        fmt=".1f",
        cmap="viridis",
        xticklabels=[f"B{i+1}" for i in range(log_heatmap.shape[1])],
        yticklabels=[f"Group {i+1}" for i in range(num_groups)],
        cbar_kws={"label": "log₁₀(Avg Gradient Norm)"}
    )
    # Calculate 5 evenly spaced positions on the X-axis
    num_labels = 5
    step = log_heatmap.shape[1] // (num_labels - 1)  # Number of steps between labels
    positions = [i * step for i in range(num_labels)]

    # Display only the 5 selected labels
    plt.xticks(
        ticks=positions, 
        labels=[f"B{i+1}" for i in positions],
        rotation=45, ha="right", fontsize=10
    )

    plt.title("Gradient Norms per Layer Group (Epoch 1)")
    plt.xlabel("Batch")
    plt.ylabel("Layer Groups")
    plt.tight_layout()
    plt.savefig(f"/home/aditis/ML/Assignment2/plots/{save_path}")
    plt.close()

if __name__ == "__main__":
    main()