import torch.nn as nn
from timm.models import register_model


class FC(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, out_size),
        )
    
    def forward(self, x):
        return self.net(x)

    def reset_classifier(self, num_classes):
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200, num_classes)
        )


@register_model
def fc_784_10(**_):
    return FC(784, 10)
