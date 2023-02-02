import torch
import torch.nn as nn
import torch.nn.functional as F

class fruits360_model(nn.Module):
    def __init__(self, output_size):
        super(fruits360_model, self).__init__()

        # image size : 100 -> 50 -> 25 -> 13 -> 7 -> flatten -> Linear 1024 -> Linear 256 -> class N
        
        # input size : 100 x 100
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding="same", bias=False)
        self.relu1 = F.relu
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2) # size : 50

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding="same", bias=False)
        self.relu2 = F.relu
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2) # size : 25

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding="same", bias=False)
        self.relu3 = F.relu
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # size : 13

        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding="same", bias=False)
        self.relu4 = F.relu
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # size : 7

        self.flatten1 = nn.Flatten() # 7 * 7 * 128 = 6272
        self.linear1 = nn.Linear(6272, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, output_size)

        # Need to Modify : <--

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxp1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxp2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxp3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxp4(out)

        out = self.flatten1(out)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)

        return out

def FruitsNet(output_size):
    return fruits360_model(output_size)
