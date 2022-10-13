'''
Reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_chans=3, num_classes=10, base_width=64):
        super().__init__()
        self.k = base_width
        self.in_planes = base_width
        self._block = block

        self.conv1 = nn.Conv2d(in_chans, self.k, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.k)
        self.layer1 = self._make_layer(block, self.k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * self.k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.k, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * self.k, num_blocks[3], stride=2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * self.k * self._block.expansion, num_classes)
        )

        # self.pool = nn.AvgPool2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.fc(out)
        return out

    def reset_classifier(self, num_classes):
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * self.k * self._block.expansion, num_classes)
        )


@register_model
def resnet18k(base_width=64, num_classes=10, **_):
    '''
    For standard ResNet18, base_width = 64
    '''
    return _ResNet(_BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=base_width)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBiggerBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBiggerBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class BiggerResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, base_width=16, in_chans=3):
        super(BiggerResNet, self).__init__()
        self._block = block
        self.num_layers = sum(layers)
        self.k = base_width
        self.inplanes = self.k
        self.conv1 = conv3x3(in_chans, self.k)
        self.bn1 = nn.BatchNorm2d(self.k)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.k, layers[0])
        self.layer2 = self._make_layer(block, 2 * self.k, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * self.k, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * self.k * self._block.expansion, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBiggerBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ## NOTE: Hotfix when input channels are 1 and network channels are not.
        if self.conv1.in_channels != x.size(1) and x.size(1) == 1:
            x = x.expand(-1, self.conv1.in_channels, -1, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def reset_classifier(self, num_classes):
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * self.k * self._block.expansion, num_classes)
        )

@register_model
def resnet20(num_classes=10, base_width=16, in_chans=3, **_):
    """Constructs a ResNet-20 model.
    """
    model = BiggerResNet(BasicBiggerBlock, [3, 3, 3], num_classes=num_classes,
                         base_width=base_width, in_chans=in_chans)
    return model
