import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

def separation(dataset: DataLoader,
               id_class: int,
               array: bool=False):
    """
    Extract all image tensors belonging to a specific class from a DataLoader.

    Params
    -------
    dataset : DataLoader
        The DataLoader providing batches of (image, label) pairs.
    id_class : int
        The class label used to filter the dataset. Only images with this label 
        will be included in the output.
    array: bool
        Whether or not the givent function return an np.array(set to True) or a torch.tensor(set to False).
        Defaults to False.

    Returns
    -------
    x:  np.ndarray or torch.tensor
        A concatenated array or tensor containing all images from the specified class.
    """

    x = []

    for batch in dataset:
        images, labels = batch[0], batch[1]
        mask = labels == id_class
        selected_images = images[mask]
        x.append(selected_images)

    # Concatenate all selected images into one tensor
    x = torch.cat(x, dim=0)
    # Turn the tensor into a np.array if array is True
    if array:
        try:
            x.detach().cpu().numpy()
        except:
            x = np.array(x)

    return x