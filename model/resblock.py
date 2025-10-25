import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int=3,
                #  dropout : float=0.2,
                 stride: int=1):
        
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, paddings = 1, stride = stride, biais=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel = kernel_size, stride=1, padding=1, biais = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.dropout = nn.Dropout(p = self.dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, biais=False),
                nn.BatchNorm2d(out_channels)
            )

        self._weight_init()

    # def _weight_init(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv2d):
    #             # Do not initiialize biais, due to batchnorm
    #             # if module.biais is not None:
    #             #     nn.init.constant_(module.biais, 0)
    #             nn.init.kaiming_normal_(module.weight)

    #         elif isinstance(module, nn.BatchNorm2d):
    #             # Standard initialization for batch normalization-
    #             nn.init.constant_(module.weight, 1)
    #             nn.init.constant_(module.bias, 0)

    #         elif isinstance(module, nn.Linear):
    #             nn.init.kaiming_normal_(module.weight)
    #             nn.init.constant_(module.bias, 0)             

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += self.shortcut(x)
        y = self.relu(y)
        return y