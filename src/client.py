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


def _resnet_features(net, x):
    """Penultimate features of a torchvision ResNet (input to net.fc)."""
    x = net.conv1(x); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
    x = net.layer1(x); x = net.layer2(x); x = net.layer3(x); x = net.layer4(x)
    x = net.avgpool(x)
    return torch.flatten(x, 1)


def train_moon(net, trainloader, epochs, device, lr, momentum,
               global_net, prev_net, mu=1.0, temperature=0.5):
    """MOON: model-contrastive local training.

    Adds a contrastive term that pulls the local representation towards the
    global model's representation and away from the client's previous-round
    representation (Li et al., CVPR 2021). ``global_net`` and ``prev_net`` are
    frozen model copies.
    """
    criterion = nn.CrossEntropyLoss()
    cos = nn.CosineSimilarity(dim=-1)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    net.train()
    global_net.eval(); prev_net.eval()
    for p in global_net.parameters():
        p.requires_grad_(False)
    for p in prev_net.parameters():
        p.requires_grad_(False)

    total_loss, correct, total = 0.0, 0, 0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            z = _resnet_features(net, images)
            outputs = net.fc(z)
            loss = criterion(outputs, labels)

            with torch.no_grad():
                z_glob = _resnet_features(global_net, images)
                z_prev = _resnet_features(prev_net, images)
            sim_glob = cos(z, z_glob).unsqueeze(1)
            sim_prev = cos(z, z_prev).unsqueeze(1)
            logits_con = torch.cat([sim_glob, sim_prev], dim=1) / temperature
            labels_con = torch.zeros(images.size(0), dtype=torch.long, device=device)
            loss = loss + mu * criterion(logits_con, labels_con)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(trainloader), correct / total


def train_scaffold(net, trainloader, epochs, device, lr, momentum,
                   c_global, c_local):
    """SCAFFOLD: local training with control-variate correction.

    ``c_global`` / ``c_local`` are dicts {param_name: tensor}. Returns the loss,
    accuracy, the updated client control variate, and its delta (Karimireddy et
    al., ICML 2020).
    """
    criterion = nn.CrossEntropyLoss()
    # SCAFFOLD's control-variate update c_i = (x - y)/(K*lr) assumes plain SGD;
    # momentum would inflate the estimate and destabilize training.
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.0)
    net.train()
    x0 = {n: p.detach().clone() for n, p in net.named_parameters()}  # global weights

    total_loss, correct, total, steps = 0.0, 0, 0, 0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # SCAFFOLD correction: g <- g - c_local + c_global
            for n, p in net.named_parameters():
                if p.grad is not None:
                    p.grad.data += (c_global[n] - c_local[n])
            optimizer.step()
            steps += 1
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # c_i^+ = c_i - c + (x - y)/(K*lr);  delta = c_i^+ - c_i
    new_c_local, delta_c = {}, {}
    denom = max(steps, 1) * lr
    for n, p in net.named_parameters():
        new_c = c_local[n] - c_global[n] + (x0[n] - p.detach()) / denom
        delta_c[n] = new_c - c_local[n]
        new_c_local[n] = new_c
    return total_loss / len(trainloader), correct / total, new_c_local, delta_c


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

