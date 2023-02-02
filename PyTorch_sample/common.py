import os
import torch

from torch.utils.data import Dataset

cache_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/PyTorch_cache/"

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

# For Inference Tensor Dataset
class InferenceTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform: x = self.transform(x)
        return x

    def __len__(self):
        return self.tensors.size(0)

# Train
def train(net, epoch, device, loss_fn, trainloader, optimizer):
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

# Inference
def inference(net, device, inferenceloader, classes):
    try:
        print("\n#########################################################")
        print("Inference Start")
        net.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(inferenceloader):
                inputs = inputs.to(device)
                outputs = net(inputs)
                #print(outputs)
                print("index : ", batch_idx , " : ", classes[torch.argmax(outputs)])
        print("Inference Done")

    except Exception as e : print("Exception :", e)

# Save
def parameter_save(net, epoch, dataset_name):
    try:
        print("\nPyTorch Save Start")

        state = {
                    'net': net.state_dict(),
                    'epoch': epoch
                }

        save_info = cache_path + "PyTorch_" + dataset_name + "_epoch_"+ str(epoch) + ".pth"
        torch.save(state, save_info)

        print(save_info, " : Saved.")

        print("PyTorch Save Done\n")

    except Exception as e : print("Exception :", e)

# Load
def parameter_load(net, start_epoch, dataset_name, load_flag):
    try:
        if (load_flag):
            # Load File

            load_info = cache_path + "PyTorch_" + dataset_name + "_epoch_"+ str(start_epoch) + ".pth"
            checkpoint = torch.load(load_info)            

            # Get Parameter : weight
            net.load_state_dict(checkpoint['net'])

            print("Load Path :", load_info)
            print("Load Epoch :", checkpoint['epoch'],"\n")

            start_epoch = checkpoint['epoch'] + 1

        return net, start_epoch

    except Exception as e : print("Exception :", e)

