import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model,
         train_set: DataLoader,
         criterion,
         optimizer,
         device,
         epochs,
         val_set: DataLoader=None):
    """
    Train a PyTorch model
    """

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        # running_dc = 0

        loop = tqdm(train_set, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_set.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}")

        if val_set is not None:
            val_loss = evaluate(model, val_set, criterion, device)
            print(f"Val Loss:   {val_loss:.4f}")

def evaluate(model, 
             data_set: DataLoader,
             criterion,
             device):
    """
    Evaluate the model on a dataset.
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in data_set:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    
    return total_loss/len(data_set.dataset)