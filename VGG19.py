import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale, Lambda

import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm
def gray_to_rgb(img):
    return img.convert('RGB')

transform = Compose([
    Lambda(gray_to_rgb),  
    Resize((32, 32)),  
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
model = models.vgg19_bn(pretrained=False, num_classes=10).to(device)
model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.to(device)
#summary(model, input_size=(1, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for data, target in train_loader_tqdm:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        train_loader_tqdm.set_postfix(train_loss=train_loss/total, train_accuracy=100. * correct/total)
    train_losses.append(train_loss / len(train_loader.dataset))
    train_accuracies.append(correct / total)

    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_losses.append(test_loss / len(test_loader.dataset))
    test_accuracies.append(correct / total)
    torch.save(model.state_dict(), f'./VGG19epoch/model_epoch_{epoch+1}.pth')
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}')

fig, axs = plt.subplots(2, 1, figsize=(12, 10))  


axs[0].plot(train_losses, label='Train Loss')
axs[0].plot(test_losses, label='Test Loss')
axs[0].set_title('Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()


axs[1].plot(train_accuracies, label='Train Accuracy')
axs[1].plot(test_accuracies, label='Test Accuracy')
axs[1].set_title('Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.savefig('training_validation_loss_accuracy.jpg')
plt.show()