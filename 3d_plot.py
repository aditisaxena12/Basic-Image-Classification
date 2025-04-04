import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from common.utils import *
from common.train_utils import *
from ResNet import PlainNet, ResNet
from version9 import Net
import copy
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Compute loss
def compute_loss(model, data_loader, criterion, device):
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples

# Generate loss surface
def generate_loss_surface(model, data_loader, criterion, device, surf_file):
    # choose random directions
    param_list = list(model.parameters())
    param_shapes = [p.shape for p in param_list]

    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)

    alpha_range = np.linspace(-0.01, 0.01, 50)
    beta_range = np.linspace(-0.01, 0.01, 50)

    loss_surface = np.zeros((len(alpha_range), len(beta_range)))

    for alpha_idx, alpha in tqdm(enumerate(alpha_range), total=len(alpha_range)):
        for beta_idx, beta in enumerate(beta_range):
            # Create a new model with perturbed weights
            perturbed_model = copy.deepcopy(model)  # Creates an exact copy of the model
            perturbed_model.load_state_dict(model.state_dict())
            perturbed_model.to(device)

            # Perturb the weights
            for param, shape in zip(perturbed_model.parameters(), param_shapes):
                # choose random directions
                direction1 = torch.randn(shape).to(device)
                direction2 = torch.randn(shape).to(device)
                # Normalize the directions
                direction1 /= torch.norm(direction1)
                direction2 /= torch.norm(direction2)

                # Filter Normalize
                direction1 = direction1 * torch.norm(param.data)
                direction2 = direction2 * torch.norm(param.data)

                # Perturb the weights
                param.data = param.data + alpha * direction1 + beta * direction2


            # Compute loss
            loss = compute_loss(perturbed_model, data_loader, criterion, device)
            loss_surface[alpha_idx, beta_idx] = loss

    np.savez(surf_file, alphas=alpha_range, betas=beta_range, loss_surface=loss_surface)
    return alpha_range, beta_range, loss_surface    
    

# Plot loss landscape
def plot_loss_surface(alphas, betas, loss_surface, surf_file = "loss_surface.png", cont_file = "loss_contour.png"):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    X, Y = np.meshgrid(alphas, betas)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, loss_surface, cmap=cm.viridis, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(loss_surface.min(), loss_surface.max())  # Dynamically set limits
    ax.zaxis.set_major_locator(LinearLocator(10))  # 10 ticks on z-axis
    ax.zaxis.set_major_formatter('{x:.02f}')  # Format labels

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Labels and title
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Surface (3D)")

    # Save and close
    plt.savefig(surf_file, dpi = 300)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    contour = plt.contour(X,Y, loss_surface, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label="Loss Value")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title("Loss Contour (2D)")
    plt.savefig(cont_file)
    plt.close(fig)

# Main function
def main():

    resnet_models = ['resnet20_500', 'resnet56_500', 'resnet110_500']
    plainnet_models = ['plainnet200_500', 'plainnet56_500', 'plainnet110_500']
    other_models = ['plainnet110_25', 'WD_LS_DA_50']
    layers = [3,9,18]

    # Using only test data with Normalization transform to calculate loss
    # Using same batch size,  can be altered
    _, test_loader = get_data("cifar10", batch_size=64)

    # Loss function - no lable smoothing - compare how the model behaves with hard labels
    criterion = nn.CrossEntropyLoss()

    for idx, resnet_model in enumerate(resnet_models):
        print(f"Generating loss surface for {resnet_model}...")
        layer = layers[idx]
        model_path = f"/home/aditis/ML/Assignment2/models/model_{resnet_model}.pth"
        model = ResNet(layer)
        plot_file = f"/home/aditis/ML/Assignment2/plots/loss_surface_{resnet_model}.png"
        contour_file = f"/home/aditis/ML/Assignment2/plots/loss_contour_{resnet_model}.png"
        surf_file = f"/home/aditis/ML/Assignment2/models/surface_{resnet_model}.npz"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Compute loss landscape
        alphas, betas, loss_surface = generate_loss_surface(model, test_loader, criterion, device, surf_file)

        # Plot the loss landscape
        plot_loss_surface(alphas, betas, loss_surface, plot_file, contour_file)

    for idx, plainnet_model in enumerate(plainnet_models):
        print(f"Generating loss surface for {plainnet_model}...")
        layer = layers[idx]
        model_path = f"/home/aditis/ML/Assignment2/models/model_{plainnet_model}.pth"
        model = PlainNet(layer)
        plot_file = f"/home/aditis/ML/Assignment2/plots/loss_surface_{plainnet_model}.png"
        contour_file = f"/home/aditis/ML/Assignment2/plots/loss_contour_{plainnet_model}.png"
        surf_file = f"/home/aditis/ML/Assignment2/models/surface_{plainnet_model}.npz"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Compute loss landscape
        alphas, betas, loss_surface = generate_loss_surface(model, test_loader, criterion, device, surf_file)

        # Plot the loss landscape
        plot_loss_surface(alphas, betas, loss_surface, plot_file, contour_file)

    for idx, other_model in enumerate(other_models):
        print(f"Generating loss surface for {other_model}...")
        model_path = f"/home/aditis/ML/Assignment2/models/model_{other_model}.pth"
        model = Net()
        plot_file = f"/home/aditis/ML/Assignment2/plots/loss_surface_{other_model}.png"
        contour_file = f"/home/aditis/ML/Assignment2/plots/loss_contour_{other_model}.png"
        surf_file = f"/home/aditis/ML/Assignment2/models/surface_{other_model}.npz"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Compute loss landscape
        alphas, betas, loss_surface = generate_loss_surface(model, test_loader, criterion, device, surf_file)

        # Plot the loss landscape
        plot_loss_surface(alphas, betas, loss_surface, plot_file, contour_file)


    print("Loss surface plots generated for all models.")




    

if __name__ == "__main__":

    main()