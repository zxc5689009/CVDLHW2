import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import RandomErasing


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = models.resnet50(pretrained=True)



num_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 預處理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
])


train_dataset = datasets.ImageFolder(root='./resnet_dataset/training_dataset', transform=train_transform)
val_dataset = datasets.ImageFolder(root='./resnet_dataset/validation_dataset', transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for data, targets in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        targets = targets.view_as(outputs) 
        loss = criterion(outputs, targets.type_as(outputs))
        train_loss += loss.item() * data.size(0)
        predicted = outputs.round()
        total += targets.size(0)
        correct += (predicted == targets.view_as(predicted)).sum().item()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss / len(train_loader.dataset))
    train_accuracies.append(correct / total)


    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            targets = targets.view_as(outputs) 
            loss = criterion(outputs, targets.type_as(outputs))
            val_loss += loss.item() * data.size(0)
            predicted = outputs.round()
            total += targets.size(0)
            correct += (predicted == targets.view_as(predicted)).sum().item()
    val_losses.append(val_loss / len(val_loader.dataset))
    val_accuracies.append(correct / total)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')


    torch.save(model.state_dict(), f'./improveresnet50epoch/improve_resnet50_epoch_{epoch+1}.pth')
