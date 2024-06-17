import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from architecture import *

def train(model, train_loader, optimizer, scheduler, criterion, epochs=15, alpha=0.001, beta=0.00001):
    model.train()
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}")
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        #print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}, Similarity Loss: {similarity_loss.item()}')
        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}')
    
    print('Finished Training')

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

model = SimpleConvTestNet().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

# Define optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Train the model
train(model, trainloader, optimizer, scheduler, criterion, epochs=20)

# Save the entire model
torch.save(model, 'easy_net.pth')