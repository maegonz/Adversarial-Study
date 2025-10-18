import numpy as np
import pandas as pd
import torch
import torchvision as tv
import torchvision.transforms as transforms


def dataloader(name: str,
               batch_size: int,
               shuffle: bool=True):
  """
  Dataloader for the specified dataset.

  Params
  -------
  name: str
    Name of the dataset.
  batch_size: int
    Batch size for the dataloader.
  shuffle: bool
    Whether to shuffle the dataset, defaults to True.

  Returns
  -------
  train_loader: torch.utils.data.DataLoader
    Dataloader for the training set.
  test_loader: torch.utils.data.DataLoader
    Dataloader for the test set.
  """

  # Image transformation
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  # Load dataset
  if name == 'CIFAR10':
    train_set = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  elif name == 'MNIST':
    train_set = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  elif name == 'CIFAR100':
    train_set = tv.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = tv.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
  else:
    raise ValueError("Invalid dataset name. Choose between 'MNIST', 'CIFAR10' or 'CIFAR100'.")

  # Create DataLoaders
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader


def get_tensors(dataloader: torch.utils.data.DataLoader,
                array: bool=False):
  """
  Get tensors from dataloader.

  Params
  -------
  dataloader: torch.utils.data.DataLoader
    Dataloader for the dataset.
  array: bool
    Whether to return tensors as numpy arrays or torch tensor, defaults to False.

  Returns
  -------
  x: torch.Tensor
    Images tensor from the dataloader.
  y: torch.Tensor
    Label tensor of the images from the dataloader.
  """

  x, y = [], []

  for images, labels in dataloader:
    x.append(images)
    y.append(labels)

  # Concatenate all batches into single tensors
  if array:
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
  else:
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

  return x, y