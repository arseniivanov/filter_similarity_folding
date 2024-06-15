import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
import numpy as np

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Load pretrained SqueezeNet model
squeezenet = models.squeezenet1_0(weights='IMAGENET1K_V1')

# Modify the final layer for CIFAR-10 (10 classes)
squeezenet.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
squeezenet.num_classes = 10

squeezenet = squeezenet.to(device)

def get_layer_dict(model):
    layer_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filter_shape = (module.kernel_size, module.out_channels)
            if filter_shape not in layer_dict:
                layer_dict[filter_shape] = []
            layer_dict[filter_shape].append(module)
    return layer_dict


def filter_dict_by_size(d, min_size):
    return {k: v for k, v in d.items() if len(v) >= min_size}

def penalize_similar_filters_within_layer(layer, alpha):
    filters = layer.weight
    num_filters = filters.size(0)
    
    if num_filters < 2:
        return 0
    
    similarity_loss = 0.0
    for i in range(num_filters):
        filters_i = filters[i].view(-1)
        for j in range(i + 1, num_filters):
            filters_j = filters[j].view(-1)
            similarity = F.cosine_similarity(filters_i, filters_j, dim=0)
            similarity_loss += 1 - similarity

    if num_filters > 1:
        similarity_loss /= (num_filters * (num_filters - 1) / 2)
    
    return alpha * similarity_loss

def encourage_similar_filters_across_layers(layer_dict, beta):
    #High number = good similarity
    total_similarity_gain = 0.0
    for filter_shape, layers in layer_dict.items():
        if len(layers) < 2:
            continue
        
        for i in range(len(layers)):
            filters_i = layers[i].weight.view(layers[i].weight.size(1), -1)  # Shape: [num_filters_i, filter_size]
            for j in range(len(layers)):
                if i == j:
                    continue
                filters_j = layers[j].weight.view(layers[j].weight.size(1), -1)  # Shape: [num_filters_j, filter_size]
                # Compare each filter in filters_i with every filter in filters_j
                similarity_matrix = F.cosine_similarity(filters_i.unsqueeze(1), filters_j.unsqueeze(0), dim=2)
                total_similarity_gain += torch.sum(similarity_matrix)
    
    return beta * total_similarity_gain


def filter_similarity_loss(model, alpha=0.001, beta=0.001):
    layer_dict = get_layer_dict(model)
    layer_dict = filter_dict_by_size(layer_dict, min_size=2)
    total_similarity_loss = 0.0
    
    # Penalize similar filters within the same layer
    for layers in layer_dict.values():
        for layer in layers:
            total_similarity_loss += penalize_similar_filters_within_layer(layer, alpha)
    
    print("In-layer similarity loss: ", total_similarity_loss)

    # Encourage similar filters across different layers with the same shape
    cross_gain = encourage_similar_filters_across_layers(layer_dict, beta)
    print("Cross-layer similarity gain: ", cross_gain)
    total_similarity_loss -= cross_gain
    
    return total_similarity_loss

# Data transformations and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

# Define optimizer and scheduler
optimizer = optim.AdamW(squeezenet.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, train_loader, optimizer, scheduler, criterion, epochs=15, alpha=0.01, beta=0.0001):
    model.train()
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}")
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # After each epoch, calculate and apply the filter similarity loss
        optimizer.zero_grad()
        similarity_loss = filter_similarity_loss(model, alpha, beta)
        similarity_loss.backward()
        optimizer.step()
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}, Similarity Loss: {similarity_loss.item()}')
    
    print('Finished Training')

# Train the model
train(squeezenet, trainloader, optimizer, scheduler, criterion, epochs=10)

# Save the model
torch.save(squeezenet, 'squeezenet_cifar10_filter_similarity.pth')

