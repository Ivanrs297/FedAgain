import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss, train_acc = train(self.net, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"train_loss": train_loss, "train_acc": train_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, device=self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def train(net, trainloader, epochs, device, lr=0.01, momentum=0.9):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(trainloader)
    accuracy = correct / total
    return avg_loss, accuracy


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = nn.CrossEntropyLoss()
    net.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    return avg_loss, accuracy

