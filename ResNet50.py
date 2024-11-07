import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# 检测 CUDA 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练的 ResNet50 模型
model = models.resnet50(pretrained=True)

# 冻结模型的所有层，以避免在训练中更新它们的权重
for param in model.parameters():
    param.requires_grad = False

# 替换模型的最后一个全连接层
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='./resnet_dataset/training_dataset', transform=transform)
val_dataset = datasets.ImageFolder(root='./resnet_dataset/inference_dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 训练和验证
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets.type_as(outputs))
        train_loss += loss.item() * data.size(0)
        predicted = outputs.round()
        total += targets.size(0)
        correct += (predicted == targets.view_as(predicted)).sum().item()
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss / len(train_loader.dataset))
    train_accuracies.append(correct / total)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets.type_as(outputs))
            val_loss += loss.item() * data.size(0)
            predicted = outputs.round()
            total += targets.size(0)
            correct += (predicted == targets.view_as(predicted)).sum().item()
    val_losses.append(val_loss / len(val_loader.dataset))
    val_accuracies.append(correct / total)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')
    # 保存当前 epoch 的模型权重
    torch.save(model.state_dict(), f'./resnet50epoch/cat_dog_resnet50_epoch_{epoch+1}.pth')