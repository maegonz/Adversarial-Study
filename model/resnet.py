import torch
import torch.nn as nn
from model.resblock import ResBlock

class ResNet(nn.Module):
    def __init__(self, n_class : int):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = self._make_layer(ResBlock, 64, 2, stride=1)
        self.block2 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.block3 = self._make_layer(ResBlock, 256, 2, stride=2)
        self.block4 = self._make_layer(ResBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512, n_class)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)  # Flatten layer
        y = self.dense(y)
        return y
    

# class ResNet18(nn.Module):
#     def __init__(self):
#         super(ResNet18, self).__init__()
        
#         self.conv1 = nn.Conv2d(
#             in_channels = 3, out_channels = 64,
#             kernel_size = 3, padding = 1,
#             stride = 1, bias = False
#         )
#         self.bn1 = nn.BatchNorm2d(num_features = 64)
        
#         self.resblock1 = ResNet_Block(
#             num_inp_channels = 64, num_channels = 64,
#             stride = 1, dropout = 0.2,
#             use_1x1_conv = False
#         )
        
#         self.resblock2 = ResNet_Block(
#             num_inp_channels = 64, num_channels = 64,
#             stride = 1, dropout = 0.2,
#             use_1x1_conv = False
#         )
        
#         # Downsample-
#         self.resblock3 = ResNet_Block(
#             num_inp_channels = 64, num_channels = 128,
#             stride = 2, dropout = 0.2,
#             use_1x1_conv = True
#         )
        
#         self.resblock4 = ResNet_Block(
#             num_inp_channels = 128, num_channels = 128,
#             stride = 1, dropout = 0.2,
#             use_1x1_conv = False
#         )

#         # Downsample-
#         self.resblock5 = ResNet_Block(
#             num_inp_channels = 128, num_channels = 256,
#             stride = 2, dropout = 0.2,
#             use_1x1_conv = True
#         )

#         self.resblock6 = ResNet_Block(
#             num_inp_channels = 256, num_channels = 256,
#             stride = 1, dropout = 0.2,
#             use_1x1_conv = False
#         )

#         # Downsample-
#         self.resblock7 = ResNet_Block(
#             num_inp_channels = 256, num_channels = 512,
#             stride = 2, dropout = 0.2,
#             use_1x1_conv = True
#         )

#         self.resblock8 = ResNet_Block(
#             num_inp_channels = 512, num_channels = 512,
#             stride = 1, dropout = 0.2,
#             use_1x1_conv = False
#         )
        
#         self.avg_pool = nn.AvgPool2d(kernel_size = 3, stride = 2)
        
    
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.resblock3(x)
#         x = self.resblock4(x)
#         x = self.resblock5(x)
#         x = self.resblock6(x)
#         x = self.resblock7(x)
#         x = self.resblock8(x)
#         x = self.avg_pool(x).squeeze()
#         return x