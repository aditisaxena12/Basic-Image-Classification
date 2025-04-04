import numpy as np
import seaborn as sns
from torchvision import datasets
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
train_data = datasets.CIFAR10(root='data', train=True, download=True)
test_data = datasets.CIFAR10(root='data', train=False, download=True)
x_train, y_train = train_data.data, np.array(train_data.targets)
x_test, y_test = test_data.data, np.array(test_data.targets)
class_names = train_data.classes
print(f"Number of training samples: {len(x_train)}")
print(f"Number of testing samples: {len(x_test)}")
print(f"Class names: {class_names}")

# Function to plot sample images
def plot_sample_images(x, y, class_names, num_samples=10):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("/home/aditis/ML/Assignment2/plots/sample_images.png")

# Function to plot class distribution
def plot_class_distribution(y, class_names):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y.flatten(), palette="viridis")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.savefig("/home/aditis/ML/Assignment2/plots/class_distribution.png")


print("Shape of one image: ", x_train[0].shape)
print(y_train)
# Plot sample images
print("Sample Images:")
plot_sample_images(x_train, y_train, class_names)

# Plot class distribution
print("Class Distribution:")
plot_class_distribution(y_train, class_names)
