import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from torchsummary import summary

from common import *

# Setting : -->
dataset_name = "cifar10"
from cifar10.models import *
from cifar10.parser import *
# Setting : <--

# Main
if __name__ == '__main__':
    try:
        # Model
        if (True):
            net = ResNet50()
            image_size = (224, 224)
        else:
            net = ResNet50()
            image_size = (224, 224)

        # Device Check
        if torch.cuda.is_available():
            device = 'cuda'
            net = net.to(device)
            net = torch.nn.DataParallel(net)
        else:
            device = 'cpu'
            net = net.to(device)

        # Model Show
        # summary(net, (3, @, @))
        # print(net)

        # Parsing : NumPy
        train_dataset, test_dataset, classes = cifar10_parsing()

        # To Tensor : Resize
        tool_transform = transforms.Resize(size = (224, 224))

        # Train Dataset & Dataloader
        tensor_x = torch.Tensor(train_dataset[0]).type(dtype=torch.float)
        tensor_y = torch.Tensor(train_dataset[1]).type(dtype=torch.uint8)
        train_dataset = CustomTensorDataset( tensors=(tensor_x, tensor_y), transform=tool_transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

        # Test Dataset & Dataloader
        tensor_x = torch.Tensor(test_dataset[0]).type(dtype=torch.float)
        tensor_y = torch.Tensor(test_dataset[1]).type(dtype=torch.uint8)
        test_dataset = CustomTensorDataset( tensors=(tensor_x, tensor_y), transform=tool_transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Load Parameter
        start_epoch = 0
        net, start_epoch = parameter_load(net, start_epoch, dataset_name, False)

        # Loss
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        for epoch in range(start_epoch, 200):
            print("Start Epoch : {:d}".format(epoch))
            train(net, epoch, device, loss_fn, trainloader, optimizer)
            test(net, epoch, device, loss_fn, testloader)
            parameter_save(net, epoch, dataset_name)

    except Exception as e : print("Exception :", e)

