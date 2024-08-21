#Imports

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

#Hyperparameters

training = True
Saving_State = True
Loading_State = False
testing = True

#Img transform function

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Main neural network

class mainnetwork(nn.Module):
    def __init__(self):
        super(mainnetwork, self).__init__()
        self.convlayer1 = nn.Conv2d(3, 32, 3)
        self.convlayer2 = nn.Conv2d(32, 64, 3)
        self.poollayer1 = nn.MaxPool2d(2, 2)
        self.convlayer3 = nn.Conv2d(64, 128, 3)
        self.poollayer2 = nn.MaxPool2d(2, 2)
        self.linlayer1 = nn.Linear(4608, 4608)
        self.linlayer2 = nn.Linear(4608, 1000)
        self.linlayer3 = nn.Linear(1000, 500)
        self.linlayer4 = nn.Linear(500, 10)
    def forward(self, network_in):
        forward = self.convlayer1(network_in)
        forward = F.relu(forward)
        forward = self.convlayer2(forward)
        forward = F.relu(forward)
        forward = self.poollayer1(forward)
        forward = self.convlayer3(forward)
        forward = F.relu(forward)
        forward = self.poollayer2(forward)
        forward = torch.flatten(forward, start_dim = 1)
        forward = self.linlayer1(forward)
        forward = F.relu(forward)
        forward = self.linlayer2(forward)
        forward = F.relu(forward)
        forward = self.linlayer3(forward)
        forward = F.relu(forward)
        network_out = self.linlayer4(forward)
        return network_out

#Main function

def main():
    #Define batch size
    batch_size = 50
    #Detect if cuda is available (if not then use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Training and testing datasets and dataloaders

    if training:
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                                download = True, transform = transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                shuffle = True, num_workers = 2)

    if testing:
        testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                            download = True, transform = transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                                shuffle = False, num_workers = 2)
        #classes = ('plane', 'car', 'bird', 'cat',
                #'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = mainnetwork().to(device)

    if Loading_State:
        model.load_state_dict(torch.load('model_save.pth', map_location = device))
    
    #Hyperparameters

    learn_rate = 0.01
    epoch_count = 60

    criterion = nn.CrossEntropyLoss()
    SGDoptimize = optim.SGD(model.parameters(), lr = learn_rate)

    torch.manual_seed(25)
    
    #Training loop

    if training:
        for epoch in range (epoch_count):
            for b_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                SGDoptimize.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                SGDoptimize.step()

                if b_idx % 10 == 0:
                    print(f'loss: {loss.item()} epoch: {epoch+1}')
    
    #Testing loop

    if testing:
        print("time for eval")
        model.eval()
        score = 0
        count = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                unused, net_prediction = torch.max(outputs.data, 1)
                count += targets.size(0)
                score += (net_prediction == targets).sum().item()
        acc = score/count
        print(f"accuracy is: {acc}")

    if Saving_State:
        torch.save(model.state_dict(), 'model_save.pth')
        print("successfully saved model state")
                
#Execute main function (not as a child process)
if __name__ == '__main__':
    main()