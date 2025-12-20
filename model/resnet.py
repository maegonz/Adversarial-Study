import torch
import torch.nn as nn
from model.resblock import ResBlock


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten =  nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out