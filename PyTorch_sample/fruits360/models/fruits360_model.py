import torch
import torch.nn as nn
import torch.nn.functional as F

class Fruits360(nn.Module):
    def __init__(self):
        super(Fruits360, self).__init__()

        # Need to Modify : -->
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = F.relu
        self.maxp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Need to Modify : <--

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxp1(out)

        return out


def fruits360():
    return Fruits360()
