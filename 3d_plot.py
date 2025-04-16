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

def compute_loss(model, data_loader, criterion, device):
    total_loss = 0.0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    return total_loss / total_samples

def get_random_directions(model):
    direction = copy.deepcopy(model)
    for param in direction.parameters():
        param.data = torch.randn_like(param)
    return direction

def normalize_filterwise(direction, model):
    for (name, param), (_, dir_param) in zip(model.named_parameters(), direction.named_parameters()):
        if "weight" in name and len(param.size()) >= 2:  # for conv net filters
            #print("param size",param.size())   # 64,64,3,3 
            #print("dir size",dir_param.size()) # 64,64,3,3
            filter_norms = param.view(param.size(0), -1).norm(p=2, dim=1) # norm of theta
            norms = dir_param.view(param.size(0), -1).norm(p=2, dim=1, keepdim=True) # norm of d
            #print("size of filter norms",filter_norms.size()) # 64
            #print("size of norms",norms.size()) # 64,1
            norms = norms.clamp(min=1e-10)
            dir_param.data = dir_param.view(param.size(0), -1) / norms # divide by norm of d
            dir_param.data *= filter_norms.view(-1,1) # multiply by norm of theta
            dir_param.data = dir_param.data.view_as(param)

        else:
            # Normalize dense layers or bias terms as a whole
            norm = dir_param.norm(p=2).clamp(min=1e-10)
            dir_param.data = dir_param / norm
            dir_param.data *= param.norm(p=2)
            dir_param.data = dir_param.data.view_as(param)
    return direction

# Generate loss surface
def generate_loss_surface(model, data_loader, criterion, device, surf_file):

    # Get random directions
    direction1 = get_random_directions(model)
    direction2 = get_random_directions(model)
    direction1 = normalize_filterwise(direction1, model)
    direction2 = normalize_filterwise(direction2, model)


    base_weights = [p.clone().detach() for p in model.parameters()]

    alpha_range = np.linspace(-0.5, 0.5, 21)
    beta_range = np.linspace(-0.5, 0.5, 21)

    loss_surface = np.zeros((len(alpha_range), len(beta_range)))

    for alpha_idx, alpha in tqdm(enumerate(alpha_range), total=len(alpha_range)):
        for beta_idx, beta in enumerate(beta_range):
            with torch.no_grad():
                for p, w, d1, d2 in zip(model.parameters(), base_weights, direction1.parameters(), direction2.parameters()):
                    p.copy_(w + alpha * d1 + beta * d2)

                        
            loss = compute_loss(model, data_loader, criterion, device)
            loss_surface[alpha_idx, beta_idx] = min(loss, 20)  

    np.savez(surf_file, alphas=alpha_range, betas=beta_range, loss_surface=loss_surface)
    return alpha_range, beta_range, loss_surface  
    

# Plot loss landscape
def plot_loss_surface(alphas, betas, loss_surface, surf_file = "loss_surface.png", cont_file = "loss_contour.png"):
    
    X, Y = np.meshgrid(alphas, betas)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, loss_surface shape: {loss_surface.shape}")
    print("minimum loss:", loss_surface.min())
    print("maximum loss:", loss_surface.max())
    surf = ax.plot_surface(X, Y, loss_surface, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_zlim(loss_surface.min(), loss_surface.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_zlabel("Loss")
    ax.set_zscale("log")
    ax.set_title("Loss Surface (3D)")
    plt.savefig(surf_file, dpi=300)
    plt.close(fig)

    # 2D Contour Plot
    fig = plt.figure(figsize=(16, 9))
    
    contour = plt.contour(X, Y, loss_surface, levels=40, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
    plt.colorbar(contour, label="Loss Value")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title("Loss Contour (2D)")
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(cont_file, dpi = 300)
    plt.close(fig)

    
def main():

    resnet_models = ['resnet20_500', 'resnet56_500', 'resnet110_500']
    plainnet_models = ['plainnet20_500', 'plainnet56_500', 'plainnet110_500']
    other_models = ['WD_LS_DA_50']
    layers = [3,9,18]

    # Using only test data with Normalization transform to calculate loss
    # Using same batch size,  can be altered
    _, test_loader = get_data("cifar10", batch_size=128)

    # Loss function - no label smoothing - compare how the model behaves with hard labels
    criterion = nn.CrossEntropyLoss()

    for idx, resnet_model in enumerate(resnet_models):
        print(f"Generating loss surface for {resnet_model}...")
        layer = layers[idx]
        model_path = f"/home/aditis/ML/Assignment2/models/model_{resnet_model}.pth"
        model = ResNet(layer)
        plot_file = f"/home/aditis/ML/Assignment2/plots/loss_surface_{resnet_model}_new.png"
        contour_file = f"/home/aditis/ML/Assignment2/plots/loss_contour_{resnet_model}_new.png"
        surf_file = f"/home/aditis/ML/Assignment2/models/surface_{resnet_model}_new.npz"
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
        plot_file = f"/home/aditis/ML/Assignment2/plots/loss_surface_{plainnet_model}_new.png"
        contour_file = f"/home/aditis/ML/Assignment2/plots/loss_contour_{plainnet_model}_new.png"
        surf_file = f"/home/aditis/ML/Assignment2/models/surface_{plainnet_model}_new.npz"
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
        plot_file = f"/home/aditis/ML/Assignment2/plots/loss_surface_{other_model}_new.png"
        contour_file = f"/home/aditis/ML/Assignment2/plots/loss_contour_{other_model}_new.png"
        surf_file = f"/home/aditis/ML/Assignment2/models/surface_{other_model}_new.npz"
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