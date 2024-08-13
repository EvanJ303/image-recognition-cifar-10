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
        self.convlayer1 = nn.Conv2d(3, 7, 3)
        self.convlayer2 = nn.Conv2d(7, 10, 3)
        self.poollayer1 = nn.MaxPool2d(2, 2)
        self.convlayer3 = nn.Conv2d(7, 14, 3)
        self.poollayer2 = nn.MaxPool2d(2, 2)
        self.linlayer1 = nn.Linear(14 * 6 * 6, 250)
        self.linlayer2 = nn.Linear(250, 130)
        self.linlayer3 = nn.Linear(130, 10)
    def forward(self, network_in):
        forward1 = self.convlayer1(network_in)
        forward2 = F.relu(forward1)
        forward3 = self.convlayer2(forward2)
        forward4 = F.relu(forward3)
        forward4 = self.poollayer1(forward3)
        forward5 = self.convlayer2(forward4)
        forward6 = F.relu(forward5)
        forward7 = self.poollayer2(forward6)
        forward8 = torch.flatten(forward7, start_dim = 1)
        forward9 = self.linlayer1(forward8)
        forward10 = F.relu(forward9)
        forward11 = self.linlayer2(forward10)
        forward12 = F.relu(forward11)
        network_out = self.linlayer3(forward12)
        return network_out

#Main function

def main():
    #Define batch size
    batch_size = 80
    #Detect if cuda is available (if not then use CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Training and testing datasets and dataloaders

    if training == True:
        trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                                download = True, transform = transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                shuffle = True, num_workers = 2)

    if testing == True:
        testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                            download = True, transform = transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                                shuffle = False, num_workers = 2)
        #classes = ('plane', 'car', 'bird', 'cat',
                #'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = mainnetwork().to(device)

    if Loading_State == True:
        model.load_state_dict(torch.load('model_save.pth', map_location = device))
    
    #Hyperparameters

    learn_rate = 0.03
    epoch_count = 70

    criterion = nn.CrossEntropyLoss()
    SGDoptimize = optim.SGD(model.parameters(), lr = learn_rate)

    torch.manual_seed(25)
    
    #Training loop

    if training == True:
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

    if testing == True:
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

    if Saving_State == True:
        torch.save(model.state_dict(), 'model_save.pth')
                
#Execute main function (not as a child process)
if __name__ == '__main__':
    main()
