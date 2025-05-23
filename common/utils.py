import torch
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_data(
        dataset_name: str,
        batch_size: int = 32,
        transform = transforms.ToTensor(),
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crops & pads images
        transforms.RandomHorizontalFlip(),  # 50% chance to flip image
        transforms.AutoAugment(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ]),

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'mnist':
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_data = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError('Dataset not supported')

    # for some reason num_workers>0 pauses training after every epoch, so not
    # using it and keeping it at 0 (default value). For more discussion visit:
    # https://discuss.pytorch.org/t/dataloader-with-num-workers-1-hangs-every-epoch/20323/18
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader
