'''Train CIFAR10 with PyTorch.'''
import os
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

from ..models import *
from ..utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 
start_epoch = 0 

# 이부분 내재화 해야함 : -->
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
# 이부분 내재화 해야함 : <--


# 2023.01.20 : -->
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
# 2023.01.20 : <--


if (False):

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10( root='./data', train=False, download=True, transform=transform_test )
    testloader = torch.utils.data.DataLoader( testset, batch_size=100, shuffle=False )

    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10( root='./data', train=True, download=True, transform=transform_train )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=128, shuffle=False )

    ############################## Dataset 출력 : -->
    get_data = trainset[0]
    show_index = 5

    print( get_data[0] ) # image
    print( get_data[1] ) # lable

    # row 출력
    print(get_data[0][0][show_index])
    # col 출력
    for col in range(32):
        print( get_data[0][0][show_index][col] )

    # image 출력
    # for PyTorch : channel row col
    # for PLT : row col channel
    # get_data : 현재 PyTorch를 위해 정렬되어있음
    transpose_data = np.transpose(get_data[0], (1, 2, 0))
    plt.imshow(transpose_data)
    plt.show()
    ############################## Dataset 출력 : <--

    ############################## Dataloader 출력 : -->
    # 방법 1
    if (False):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print( batch_idx )
            print( inputs[0][0] )

    # 방법 2
    if (False):
        batch_idx = 0
        get_iter = iter(trainloader)
        for idx in range(10000):
            # batch_idx 수동 체크 : -->
            print( batch_idx )
            inputs, targets = next(get_iter)
            batch_idx += 1
            print( inputs[0][0] )
            # batch_idx 수동 체크 : <--
    ############################## Dataloader 출력 : <--


def cifar10_parsing():
    try:
        print("Parsing Start")

        download_path = "C:/Users/" + os.environ.get("USERNAME") + "/Desktop/resnet/"

        train_count = 5
        test_count = 1

        lable_size = 1
        image_size = (3 * 32 * 32)
        image_count = 1000
        class_count = 10

        unit_size = lable_size + image_size
        
        train_lable = []
        train_image = []
        test_lable = []
        test_image = []

        # Train
        for idx in range(train_count):
            filepath_cifar10 = download_path + "cifar-10-batches-bin/data_batch_" + str(idx+1) +".bin"
            binary_cifar10 = open(filepath_cifar10,'rb')
            numpy_cifar10 = np.fromfile(binary_cifar10, dtype=np.uint8)
            for sub_idx in range(image_count * class_count):
                # [ index (0000) ~ index (lable_size + image_size) ]
                # [ index (0000) ~ index (3072) ]
                # [ 3073 byte ]
                # [ index (3073) ~ index (6145) ]
                # [ 3073 byte ]
                unit = numpy_cifar10[ sub_idx*unit_size : (sub_idx+1)*unit_size ]
                # lable
                train_lable.append(unit[0])
                # image
                train_image.append(unit[1:].reshape(3,32,32))  # for PyTorch : channel row col 
                # for PLT : row col channel
                #transpose_data = np.transpose(train_image[sub_idx], (1, 2, 0))
                #plt.imshow(transpose_data)
                #plt.show()
            binary_cifar10.close()

        # Test
        filepath_cifar10 = download_path + "cifar-10-batches-bin/test_batch.bin"
        binary_cifar10 = open(filepath_cifar10,'rb')
        numpy_cifar10 = np.fromfile(binary_cifar10, dtype=np.uint8)
        for sub_idx in range(image_count * class_count):
            unit = numpy_cifar10[ sub_idx*unit_size : (sub_idx+1)*unit_size ]
            test_lable.append(unit[0])
            test_image.append(unit[1:].reshape(3,32,32))  # for PyTorch : channel row col 
        binary_cifar10.close()

        # get raw
        train_image = np.array(train_image, dtype=np.uint8)
        train_lable = np.array(train_lable, dtype=np.uint8)
        test_image = np.array(test_image, dtype=np.uint8)
        test_lable = np.array(test_lable, dtype=np.uint8)

        # normalize
        train_image = np.array(train_image/255.0, dtype=np.float32)
        train_lable = np.array(train_lable, dtype=np.uint8)
        test_image = np.array(test_image/255.0, dtype=np.float32)
        test_lable = np.array(test_lable, dtype=np.uint8)

        # tensor
        tensor_x = torch.Tensor(train_image).type(dtype=torch.float)
        tensor_y = torch.Tensor(train_lable).type(dtype=torch.uint8)


        tool_transform = transforms.Resize(size = (224, 224))
        tensor_x = tool_transform(tensor_x)

        transform_train = transforms.Compose( [transforms.Resize(224)] )
        train_dataset_normal = CustomTensorDataset(tensors=(X_train, y_train), transform=None)
        train_dataset = TensorDataset(tensor_x, tensor_y, transform=transform_train)

        tensor_x = torch.Tensor(test_image).type(dtype=torch.float)
        tensor_y = torch.Tensor(test_lable).type(dtype=torch.uint8)
        test_dataset = TensorDataset(tensor_x, tensor_y)

        # Resize
        # Test 01 : -->
        if (False):
            get_data = train_dataset[0]
            get_image = get_data[0]

            print(get_image.shape)
            get_image = get_image.unsqueeze(0)
            get_image = torch.nn.functional.interpolate(get_image, size=(224, 224), mode='bilinear')
            get_image = get_image.squeeze(0)

            transpose_data = np.transpose(get_image, (1, 2, 0))
            plt.imshow(transpose_data)
            plt.show()
        # Test 01 : <--
        
        # Test 02 : -->
        if (True):
            tensor_x = torch.Tensor(train_image).type(dtype=torch.float)
            print(tensor_x.shape)
            tensor_x = tensor_x.unsqueeze(0)
            tensor_x = torch.nn.functional.interpolate(tensor_x, size=(3, 224, 224), mode='bilinear')
            tensor_x = tensor_x.squeeze(0)

            transpose_data = np.transpose(get_image, (1, 2, 0))
            plt.imshow(transpose_data)
            plt.show()
        # Test 01 : <--

        print("Parsing Done\n")
        return train_dataset, test_dataset

    except Exception as e : print("Exception :", e)

train_dataset, test_dataset = cifar10_parsing()
trainloader = torch.utils.data.DataLoader( train_dataset, batch_size=128, shuffle=False )
testloader = torch.utils.data.DataLoader( test_dataset, batch_size=100, shuffle=False )

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
#net = ResNet18()
net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == '__main__':

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
