import math
import torch.nn as nn
from timm.models import register_model

class _LeNet(nn.Module):
    def __init__(self, in_chans=1, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 6, 5),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        # self.conv1 = nn.Conv2d(in_channels, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(16*5*5, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        out = self.net(x)
        return out


class _LeNet5(nn.Module):
    def __init__(self, in_chans=1, num_classes=10):
        super().__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_chans, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(True),
            nn.Linear(500, num_classes)
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        ## NOTE: Hotfix when input channels are 1 and network channels are not.
        if self.conv_part[0].in_channels != x.size(1) and x.size(1) == 1:
            x = x.expand(-1, self.conv_part[0].in_channels, -1, -1)

        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x

    def reset_classifier(self, num_classes):
        self.fc_part = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(True),
            nn.Linear(500, num_classes)
        )


@register_model
def LeNet(in_chans=1, num_classes=10, **_):
    return _LeNet(in_chans=in_chans, num_classes=num_classes)


@register_model
def LeNet5(in_chans=1, num_classes=10, **_):
    return _LeNet5(in_chans=in_chans, num_classes=num_classes)
