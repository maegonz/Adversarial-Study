import torch
import torch.nn as nn
from typing import Optional

def split(model: nn.Module,
          id: Optional[int] = None):
    """
    Split a model in two, defaults to split at the penultimate layer.

    Params
    -------
    model: nn.Module
        Model to split.
    id: int, Optional
        Index of the layer split point. Defaults to the penultimate layer.

    Returns
    -------
    g: nn.Sequential
        Layers before the split point.
    h: nn.Sequential
        Layers after the split point.
    """

    layers = list(model.children())

    if id == None:
        id = len(layers)-1
    
    g = nn.Sequential(*layers[:id])
    h = nn.Sequential(*layers[id:])

    return g, h

