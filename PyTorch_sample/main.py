import os
import sys
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset

import torchvision.transforms as transforms

cache_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_cache/"

# Setting : -->
dataset_name = "cifar10"
from cifar10.models import *
from cifar10.parser import *
# Setting : <--`

# Custom Tensor Dataset
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
def train(net, epoch, device, loss_fn, trainloader):
    try:
        print("\n#########################################################")
        print("Train Batch Size : {:d}".format(trainloader.batch_size))
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

            if (batch_idx%10 == 9) or (batch_idx == len(trainloader)-1):
                print( "\nBatch : {:d}/{:d}".format(batch_idx+1, len(trainloader)) )
                print( "Train Loss : {:.3f} ".format((train_loss/(batch_idx+1))) )
                print( "Train ACC : {:.2f}% [{:d}/{:d}] ".format((100.0 * correct / total), correct, total) )

    except Exception as e : print("Exception :", e)

# Test
def test(net, epoch, device, loss_fn, testloader):
    try:
        print("\n#########################################################")
        print("Test Batch Size : {:d}".format(testloader.batch_size))
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

                if (batch_idx%10 == 9) or (batch_idx == len(testloader)-1):
                    print( "\nBatch : {:d}/{:d}".format(batch_idx+1, len(testloader)) )
                    print( "Test Loss : {:.3f} ".format((test_loss/(batch_idx+1))) )
                    print( "Test ACC : {:.2f}% [{:d}/{:d}] ".format((100.0 * correct / total), correct, total) )

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

        save_name = cache_path + "PyTorch_" + dataset_name + "_epoch_"+ str(epoch) + ".pth"
        torch.save(state, save_name)

        print(save_name, " : Saved.")

        print("PyTorch Save Done\n")

    except Exception as e : print("Exception :", e)

# Load
def parameter_load(net, start_epoch):
    try:
        if (False):
            # Load File
            file_name = "PyTorch_cifar10_epoch_0.pth"
            checkpoint = torch.load(cache_path + file_name)            

            # Get Parameter : weight
            net.load_state_dict(checkpoint['net'])

            # Get Parameter : state
            start_epoch = checkpoint['epoch'] + 1

            print("Load Path :", cache_path + file_name)
            print("Load Epoch :", checkpoint['epoch'],"\n")

        return net, start_epoch

    except Exception as e : print("Exception :", e)

# Main
if __name__ == '__main__':
    try:
        # Model
        net = ResNet50()

        # Device Check
        if torch.cuda.is_available():
            device = 'cuda'
            net = net.to(device)
            net = torch.nn.DataParallel(net)
        else:
            device = 'cpu'
            net = net.to(device)

        # Parsing : NumPy
        train_dataset, test_dataset, classes = cifar10_parsing()

        # To Tensor : Resize
        tool_transform = transforms.Resize(size = (224, 224))

        # Train Dataset & Dataloader
        tensor_x = torch.Tensor(train_dataset[0]).type(dtype=torch.float)
        tensor_y = torch.Tensor(train_dataset[1]).type(dtype=torch.uint8)
        train_dataset = CustomTensorDataset( tensors=(tensor_x, tensor_y), transform=tool_transform)
        trainloader = torch.utils.data.DataLoader( train_dataset, batch_size=128, shuffle=False )

        # Test Dataset & Dataloader
        tensor_x = torch.Tensor(test_dataset[0]).type(dtype=torch.float)
        tensor_y = torch.Tensor(test_dataset[1]).type(dtype=torch.uint8)
        test_dataset = CustomTensorDataset( tensors=(tensor_x, tensor_y), transform=tool_transform)
        testloader = torch.utils.data.DataLoader( test_dataset, batch_size=128, shuffle=False )

        # Load Parameter
        start_epoch = 0
        net, start_epoch = parameter_load(net, start_epoch)

        # Loss
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        for epoch in range(start_epoch, 200):
            print("Start Epoch : {:d}".format(epoch))
            train(net, epoch, device, loss_fn, trainloader)
            test(net, epoch, device, loss_fn, testloader)
            parameter_save(net, epoch)

    except Exception as e : print("Exception :", e)

