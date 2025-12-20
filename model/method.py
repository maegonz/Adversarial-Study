import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def training(model: nn.Module,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            device: torch.device,
            epochs: int,
            val_loader: DataLoader=None):
    """
    Train a PyTorch model

    Params
    -------
    model: nn.Module
        The model to be trained.
    train_set: DataLoader
        Data to train the model on.
    criterion: nn.Module
        Loss function used for optimization.
    optimizer: torch.optim.Optimizer
        Optimizer used to update model parameters.
    device: torch.device or str
        Device on which to train the model (e.g., 'cpu' or 'cuda').
    epochs: int
        Number of epochs to train for.
    val_set: DataLoader, optional
        Data to validate the model on. Defaults to None.

    Returns
    -------
    None
        Prints training and validation loss per epoch.
    """

    model = model.to(device)
    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            val_loss, val_accuracy  = evaluating(model, val_loader, criterion, device)
            print(f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_accuracy:.4f}")


def evaluating(model: nn.Module, 
               data_loader: DataLoader,
               criterion: nn.Module,
               device: torch.device):
    """
    Evaluate the model on a dataset.

    Params
    -------
    model: nn.Module 
        Model to be evaluated.
    data_set: DataLoader
        Dataset to evaluate the model on.
    criterion: nn.Module
        Loss function used for evaluation.
    device: torch.device or str
        Device to perform evaluation on. ('cpu' or 'cuda')

    Returns
    -------
    avg_loss: float
        Average loss over the dataset.
    avg_accuracy: float
        Average accuracy (in percentage) over the dataset.
    """

    model.eval()

    total_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, pred = torch.max(outputs, 1)
            accuracy += (pred == labels).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_accuracy = 100 * (accuracy / len(data_loader.dataset))

    return avg_loss, avg_accuracy