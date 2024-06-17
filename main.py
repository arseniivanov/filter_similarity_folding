import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from architecture import *
import os


def get_layer_dict(model):
    layer_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, DecomposedConv2d):  # Check for DecomposedConv2d instead of nn.Conv2d
            filter_shape = (module.kernel_size, module.out_channels)
            if filter_shape not in layer_dict:
                layer_dict[filter_shape] = []
            layer_dict[filter_shape].append(module)
    return layer_dict

def filter_dict_by_size(d, min_size):
    return {k: v for k, v in d.items() if len(v) >= min_size}

def encourage_similar_filters_across_layers(layer_dict, beta):
    total_similarity_gain = torch.tensor(0.0, device=device, requires_grad=True)
    for filter_shape, layers in layer_dict.items():
        if len(layers) < 2:
            continue
        
        for i in range(len(layers)):
            filters_i = layers[i].filters.view(layers[i].filters.size(1), -1)  # Shape: [num_filters_i, filter_size]
            for j in range(len(layers)):
                if i == j:
                    continue
                filters_j = layers[j].filters.view(layers[j].filters.size(1), -1)  # Shape: [num_filters_j, filter_size]
                
                # Compare each filter in filters_i with every filter in filters_j
                similarity_matrix = F.cosine_similarity(filters_i.unsqueeze(1), filters_j.unsqueeze(0), dim=2)
                total_similarity_gain = total_similarity_gain + torch.sum(similarity_matrix)  # High similarity should result in high reward
    
    return beta * total_similarity_gain

def filter_similarity_loss(model, beta=0.001):
    layer_dict = get_layer_dict(model)
    layer_dict = filter_dict_by_size(layer_dict, min_size=2)
    total_similarity_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Encourage similar filters across different layers with the same shape
    total_similarity_loss = total_similarity_loss - encourage_similar_filters_across_layers(layer_dict, beta)
    
    return total_similarity_loss
# Training function
def train(model, train_loader, optimizer, scheduler, criterion, epochs=15, alpha=0.001, beta=0.00001):
    model.train()
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}")
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # After each epoch, calculate and apply the filter similarity loss
        #optimizer.zero_grad()
        #similarity_loss = filter_similarity_loss(model, beta)
        #similarity_loss.backward()
        #optimizer.step()
        
        scheduler.step()
        #print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}, Similarity Loss: {similarity_loss.item()}')
        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}')
    
    print('Finished Training')

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")


model_path = 'squeezenet_cifar10_filter_similarity.pth'

# Check if the model file exists and load the model
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    squeezenet = torch.load(model_path)
else:
    print("No saved model found, starting from scratch.")
    squeezenet = DecomposedSqueezeNet(num_classes=10).to(device)

squeezenet = DecomposedSqueezeNet(num_classes=10).to(device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

# Define optimizer and scheduler
optimizer = optim.AdamW(squeezenet.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Train the model
train(squeezenet, trainloader, optimizer, scheduler, criterion, epochs=30)

# Save the entire model
torch.save(squeezenet, 'squeezenet_cifar10_filter_similarity_tuned.pth')
