import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from architecture import *
import os

def train(model, train_loader, optimizer, scheduler, criterion, epochs=15, alpha=0.001, beta=0.00001, pretrained=False, similarity_optimizer=None, testloader=None):
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

        if pretrained: 
            similarity_loss = model.compute_filter_similarity_loss() * 1000
            optimizer.zero_grad()
            similarity_loss.backward()
            optimizer.step()
            print("Similarity loss: ", similarity_loss)

        scheduler.step()
        #print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}, Similarity Loss: {similarity_loss.item()}')
        print(f'Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader)}')
        if testloader is not None:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for test_images, test_labels in testloader:
                    test_images, test_labels = test_images.to(device), test_labels.to(device)
                    test_outputs = model(test_images)
                    test_loss += criterion(test_outputs, test_labels).item()
                    _, predicted = torch.max(test_outputs, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            test_loss /= len(testloader)
            accuracy = 100 * correct / total
            print(f'Test Loss: {test_loss}, Test Accuracy: {accuracy}%')
            model.train()
    
    print('Finished Training')

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

if os.path.exists("easy_net.pth"):
    model = torch.load("easy_net.pth").to(device)
    pretrained = True
else:
    model = SimpleConvTestNet().to(device)
    pretrained = False

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2)

# Define optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
similarity_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Train the model
train(model, trainloader, optimizer, scheduler, criterion, epochs=20, pretrained=pretrained, similarity_optimizer=similarity_optimizer, testloader=testloader)

# Save the entire model
torch.save(model, 'easy_net.pth')