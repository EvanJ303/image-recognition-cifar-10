#Imports 

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

#Transform Function Definition

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Import Sets

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                        download = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                          shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                       download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle = False, num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Neural Network

class mainnetwork(nn.Module):
    def __init__(self):
        super(mainnetwork, self).__init__()
        self.convlayer1 = nn.Conv2d(3, 7, 5)
        self.poollayer1 = nn.MaxPool2d(2, 2)
        self.convlayer2 = nn.Conv2d(7, 14, 5)
        self.poollayer2 = nn.MaxPool2d(2, 2)
        self.linlayer1 = nn.Linear(14 * 5 * 5, 130)
        self.linlayer2 = nn.Linear(130, 10)
    def forward(self, network_in):
        forward1 = self.convlayer1(network_in)
        forward1 = F.relu(forward1)
        forward2 = self.poollayer1(forward1)
        forward3 = self.convlayer2(forward2)
        forward3 = F.relu(forward3)
        forward4 = self.poollayer2(forward3)
        forward5 = torch.flatten(forward4, start_dim=1)
        forward6 = self.linlayer1(forward5)
        forward6 = F.relu(forward6)
        forwardfinal = self.linlayer2(forward6)
        return forwardfinal





#Hyperparams
learn_rate=0.1

model = mainnetwork().to('cuda')
criterion = nn.CrossEntropyLoss
SGDoptimize = torch.optim.SGD(model.parameters(), lr = learn_rate)

torch.manual_seed(30)

#Training Loop

total_steps