import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import copy


def train(net, trainloader, epochs, device, lr=0.01, momentum=0.9, proximal_mu=0.0, global_params=None):
    """Train the model on the training set.
    
    Args:
        proximal_mu: FedProx proximal term coefficient (0 = no proximal term, i.e., FedAvg)
        global_params: Global model parameters (needed for FedProx proximal term)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    
    # Store global parameters for FedProx proximal term
    if proximal_mu > 0 and global_params is not None:
        global_params_dict = {}
        for name, param in zip(net.state_dict().keys(), global_params):
            global_params_dict[name] = torch.tensor(param, device=device)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            # FedProx: Add proximal term ||w - w_global||^2
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for name, param in net.named_parameters():
                    if name in global_params_dict:
                        proximal_term += torch.sum((param - global_params_dict[name]) ** 2)
                loss += (proximal_mu / 2) * proximal_term
            
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

