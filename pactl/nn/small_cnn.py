import torch.nn as nn
from timm.models import register_model


class Named(type):
    def __str__(self):
        return self.__name__
    def __repr__(self):
        return self.__name__


class ConvBNrelu(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))


class layer13s(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10, in_chans=3, base_width=8):
        super().__init__()
        k = 2*base_width
        self.num_classes = num_classes
        self.net = nn.Sequential(
            ConvBNrelu(in_chans,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
        )
        self.fc = nn.Linear(2*k,num_classes)
    def forward(self,x):
        return self.fc(self.net(x))

class Expression(nn.Module):
    def __init__(self,func):
        super().__init__()
        self.func = func
    def forward(self,x):
        return self.func(x)

@register_model
def Layer13s(num_classes=10, in_chans=3, base_width=8, **_):
    return layer13s(num_classes=num_classes, in_chans=in_chans, base_width=base_width)
