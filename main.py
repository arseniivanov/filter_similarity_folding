import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

class DecomposedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DecomposedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.filters = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

    def forward(self, x):
        batch_size, _, height, width = x.size()
        output_height = (height - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_width = (width - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
        output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device)
        
        for i in range(self.out_channels):
            filter_i = self.filters[i:i+1, :, :, :]
            output[:, i:i+1, :, :] = F.conv2d(x, filter_i, stride=self.stride, padding=self.padding)
        
        return output


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = DecomposedConv2d(inplanes, squeeze_planes, kernel_size=(1, 1))
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = DecomposedConv2d(squeeze_planes, expand1x1_planes, kernel_size=(1, 1))
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = DecomposedConv2d(squeeze_planes, expand3x3_planes, kernel_size=(3, 3), padding=(1, 1))
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class DecomposedSqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DecomposedSqueezeNet, self).__init__()
        self.features = nn.Sequential(
            DecomposedConv2d(3, 96, kernel_size=(7, 7), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            DecomposedConv2d(512, num_classes, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x

# Load the decomposed SqueezeNet model
squeezenet = DecomposedSqueezeNet(num_classes=10).to(device)


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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

# Define optimizer and scheduler
optimizer = optim.AdamW(squeezenet.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, train_loader, optimizer, scheduler, criterion, epochs=15, alpha=0.001, beta=0.001):
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
        optimizer.zero_grad()
        similarity_loss = filter_similarity_loss(model, beta)
        similarity_loss.backward()
        optimizer.step()
        
        scheduler.step()
        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}, Similarity Loss: {similarity_loss.item()}')
    
    print('Finished Training')

# Train the model
train(squeezenet, trainloader, optimizer, scheduler, criterion, epochs=30)

# Save the entire model
torch.save(squeezenet, 'squeezenet_cifar10_filter_similarity.pth')
