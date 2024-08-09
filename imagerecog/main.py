import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class mainnetwork(nn.Module):
    def __init__(self):
        super(mainnetwork, self).__init__()
        self.fc1 = nn.Conv2d