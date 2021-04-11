import torch
t = torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist', 
    train=True, 
    download=True, 
    transform=transform)
dataset_valid = datasets.MNIST(
    '~/mnist', 
    train=False, 
    download=True, 
    transform=transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
