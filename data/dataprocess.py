import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

def separation(dataset: DataLoader,
               id_class: int):
    """
    Extract all image tensors belonging to a specific class from a DataLoader.

    Params
    -------
    dataset : DataLoader
        The DataLoader providing batches of (image, label) pairs.
    id_class : int
        The class label used to filter the dataset. Only images with this label 
        will be included in the output.
    Returns
    -------
    x:  np.ndarray
        A concatenated array containing all images from the specified class.
    """

    x = []

    for image, label in dataset:
        if label == id_class:
            x.append(image)

    x = np.concatenate(x, axis=0)

    return x