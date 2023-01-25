'''Train CIFAR10 with PyTorch.'''
import os
import sys
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

import torchvision
import torchvision.transforms as transforms

from models import *
from utils import *

# Common Path : -->
data_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_data/"
cache_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_cache/"
# Common Path : <--

def cifar10_parsing():
    try:
        print("Parsing Start")

        # Only for cifar10 Hard-Coding : -->
        train_count = 5
        test_count = 1

        lable_size = 1
        image_size = (3 * 32 * 32)
        image_count = 1000
        class_count = 10

        unit_size = lable_size + image_size
        # Only for cifar10 Hard-Coding : <--
        
        train_lable = []
        train_image = []
        test_lable = []
        test_image = []

        # Dataset : train
        for idx in range(train_count):
            filepath_cifar10 = data_path + "cifar-10-batches-bin/data_batch_" + str(idx+1) +".bin"
            binary_cifar10 = open(filepath_cifar10,'rb')
            numpy_cifar10 = np.fromfile(binary_cifar10, dtype=np.uint8)
            for sub_idx in range(image_count * class_count):

                # [ index (0000) ~ index (lable_size + image_size) ]
                # [ index (0000) ~ index (3072) ] = [ 3073 byte ]
                # [ index (3073) ~ index (6145) ] = [ 3073 byte ]
                # ...
                unit = numpy_cifar10[ sub_idx*unit_size : (sub_idx+1)*unit_size ]

                # lable
                train_lable.append( unit[0] )

                # image
                train_image.append( unit[1:].reshape(3,32,32) ) # For PyTorch : channel row col 

                # show
                if (False):
                    transpose_data = np.transpose(train_image[sub_idx], (1, 2, 0)) # For PLT : row col channel
                    plt.imshow(transpose_data)
                    plt.show()

            binary_cifar10.close()

        # Dataset : test
        for idx in range(test_count):
            filepath_cifar10 = data_path + "cifar-10-batches-bin/test_batch.bin"
            binary_cifar10 = open(filepath_cifar10,'rb')
            numpy_cifar10 = np.fromfile(binary_cifar10, dtype=np.uint8)
            for sub_idx in range(image_count * class_count):

                unit = numpy_cifar10[ sub_idx*unit_size : (sub_idx+1)*unit_size ]

                test_lable.append(unit[0])
                
                test_image.append(unit[1:].reshape(3,32,32))

            binary_cifar10.close()

        # To NumPy
        train_image = np.array(train_image, dtype=np.uint8)
        train_lable = np.array(train_lable, dtype=np.uint8)
        test_image = np.array(test_image, dtype=np.uint8)
        test_lable = np.array(test_lable, dtype=np.uint8)

        # Image Normalize
        train_image = np.array(train_image/255.0, dtype=np.float32)
        test_image = np.array(test_image/255.0, dtype=np.float32)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        print("Parsing Done\n")
        return [train_image, train_lable], [test_image, test_lable], classes

    except Exception as e : print("Exception :", e)

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform: x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

# Train
def train(net, epoch, device, loss_fn):
    try:
        print("Start Epoch : %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(   batch_idx,
                            len(trainloader),
                            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1),
                            100.*correct/total,
                            correct,
                            total)
                        )
    except Exception as e : print("Exception :", e)

# Test
def test(net, epoch, device, loss_fn):
    try:
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(   batch_idx,
                                len(testloader),
                                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1),
                                100.*correct/total,
                                correct,
                                total)
                            )
    except Exception as e : print("Exception :", e)

# Save
def parameter_save(net, epoch):
    try:
        print("\nPyTorch Save Start")

        state = {
                    'net': net.state_dict(),
                    'epoch': epoch
                }

        if not os.path.isdir('checkpoint'): os.mkdir('checkpoint')

        torch.save(state, cache_path + "PyTorch_epoch_"+ str(epoch) + ".pth")

        print("PyTorch_epoch_"+ str(epoch) + ".pth : Saved.")

        print("PyTorch Save Done\n")

    except Exception as e : print("Exception :", e)

# Main
if __name__ == '__main__':
    try:
        # Model
        net = ResNet50()

        # Device Check
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available():
            device = 'cuda'
            net = net.to(device)
            net = torch.nn.DataParallel(net)
        else:
            device = 'cpu'
            net = net.to(device)

        train_dataset, test_dataset, classes = cifar10_parsing()

        # To Tensor : Resize
        tool_transform = transforms.Resize(size = (224, 224))

        # Tensor Dataset : -->
        # Train
        tensor_x = torch.Tensor(train_dataset[0]).type(dtype=torch.float)
        tensor_y = torch.Tensor(train_dataset[1]).type(dtype=torch.uint8)
        train_dataset = CustomTensorDataset( tensors=(tensor_x, tensor_y), transform=tool_transform)
        # Test
        tensor_x = torch.Tensor(test_dataset[0]).type(dtype=torch.float)
        tensor_y = torch.Tensor(test_dataset[1]).type(dtype=torch.uint8)
        test_dataset = CustomTensorDataset( tensors=(tensor_x, tensor_y), transform=tool_transform)
        # Tensor Dataset : <--

        # Tensor Dataloader
        trainloader = torch.utils.data.DataLoader( train_dataset, batch_size=128, shuffle=False )
        testloader = torch.utils.data.DataLoader( test_dataset, batch_size=128, shuffle=False )

        # Load Parameter
        if(False):
            file_name = "PyTorch_epoch_0.pth"
            checkpoint = torch.load(cache_path + file_name)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch'] + 1
            print("Load Path :", cache_path + file_name)
            print("Load Epoch :", checkpoint['epoch'],"\n")
        else:
            start_epoch = 0

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        for epoch in range(start_epoch, 200):
            train(net, epoch, device, loss_fn)
            test(net, epoch, device, loss_fn)
            parameter_save(net, epoch)

    except Exception as e : print("Exception :", e)

