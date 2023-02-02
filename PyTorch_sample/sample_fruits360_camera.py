import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from torchsummary import summary

from common import *

# Setting : -->
dataset_name = "fruits360"
from model import *
from dataset_parser import *
# Setting : <--

# Main
if __name__ == '__main__':

    try:

        # Parsing : NumPy
        inference_dataset = webcam_parser()
        classes = ['apple', 'banana', 'orange', 'strawberry']

        # Model
        net = ResNet50(len(classes))
        image_resize = (224, 224)

        # Device Check
        if torch.cuda.is_available():
            device = 'cuda'
            net = net.to(device)
            net = torch.nn.DataParallel(net)
        else:
            device = 'cpu'
            net = net.to(device)

        # Model Show
        summary( net, (3,) + image_resize )
        print(net)

        # To Tensor : Resize
        tool_transform = transforms.Resize(size = image_resize)

        # Inference Dataset & Dataloader
        tensor_x = torch.Tensor(inference_dataset).type(dtype=torch.float)
        inference_dataset = InferenceTensorDataset( tensor_x, tool_transform )
        inferenceloader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False)

        # Load Parameter
        load_epoch = 0
        net, start_epoch = parameter_load(net, load_epoch, dataset_name, load_flag=False)

        # Loss
        loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-5)

        # Camera Inference
        inference(net, device, inferenceloader, classes)

    except Exception as e : print("Exception :", e)

