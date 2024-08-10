import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

training = True
Saving_State = True
Loading_State = True
testing = True


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():

    batch_size = 14

    trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                            download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                            shuffle = True, num_workers = 2)

    testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                        download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle = False, num_workers = 2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    class mainnetwork(nn.Module):
        def __init__(self):
            super(mainnetwork, self).__init__()
            self.convlayer1 = nn.Conv2d(3, 7, 3)
            self.poollayer1 = nn.MaxPool2d(2, 2)
            self.convlayer2 = nn.Conv2d(7, 14, 3)
            self.poollayer2 = nn.MaxPool2d(2, 2)
            self.linlayer1 = nn.Linear(14 * 6 * 6, 250)
            self.linlayer2 = nn.Linear(250, 130)
            self.linlayer3 = nn.Linear(130, 10)
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
            forward7 = self.linlayer2(forward6)
            forward7 = F.relu(forward7)
            network_out = self.linlayer3(forward7)
            return network_out

    learn_rate= 0.003
    epoch_count = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = mainnetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    SGDoptimize = torch.optim.SGD(model.parameters(), lr = learn_rate)

    torch.manual_seed(50)


    for epoch in range (epoch_count):
        for b_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            SGDoptimize.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            SGDoptimize.step()

            if b_idx % 5 ==0:
                print(f'Loss: {loss.item()}')

if __name__ == '__main__':
    main()