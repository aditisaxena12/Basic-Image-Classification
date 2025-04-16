import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def main():

    surf_file = "/home/aditis/ML/Assignment2/models/surface_resnet20_500.npz"
    data = np.load(surf_file)
    alphas = data["alphas"]
    betas = data["betas"]
    loss_surface = data["loss_surface"] 

    print("Maximum loss:", np.max(loss_surface))
    print("Minimum loss:", np.min(loss_surface))
    print(alphas.shape)
    print(betas.shape)
    print(loss_surface.shape)


    plot_file = "test_plot.png"
    cont_file = "test_contour.png"
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 5))
    X, Y = np.meshgrid(alphas, betas)

    surf = ax.plot_surface(X, Y, loss_surface, cmap=cm.viridis, linewidth=0, antialiased=False)


    ax.set_zlim(loss_surface.min(), loss_surface.max())  # Dynamically set limits
    ax.zaxis.set_major_locator(LinearLocator(10))  # 10 ticks on z-axis
    ax.zaxis.set_major_formatter('{x:.02f}')  # Format labels


    fig.colorbar(surf, shrink=0.5, aspect=5)


    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    #ax.set_zscale("log")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Surface (3D)")


    plt.savefig(plot_file, dpi = 300)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 5))
    contour = plt.contour(X,Y, loss_surface, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label="Loss Value")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title("Loss Contour (2D)")
    plt.savefig(cont_file)
    plt.close(fig)



if __name__ == "__main__":
    main()