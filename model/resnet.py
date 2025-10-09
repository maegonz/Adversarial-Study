import torch
import torch.nn as nn
from model.resblock import ResBlock

class ResNet(nn.Module):
    def __init__(self, n_class : int):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        