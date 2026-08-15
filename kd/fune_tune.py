import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import  ImageFolder
from torchvision import transforms
from torchvision.models import resnet50
import mlflow
import mlflow.pytorch

# 超参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
WD = 1e-4
NUM_CLASSES = 120

# 训练集：强数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 测试集：标准预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 分别用不同 transform 加载（PyTorch 的 ImageFolder 不支持一个 dataset 两种 transform，
# 所以需要分别创建两个 Dataset，但底层图片是一样的）
train_dataset = ImageFolder(root='../datasets/stanford_dogs/Images', transform=train_transform)
test_dataset = ImageFolder(root='../datasets/stanford_dogs/Images', transform=test_transform)

# 8:2 划分，保证两个 dataset 的索引一致
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))

train_indices = indices[:train_size].tolist()
test_indices = indices[train_size:].tolist()

# 用 Subset 提取划分后的数据
from torch.utils.data import Subset
train_dataset = Subset(train_dataset, train_indices)
test_dataset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
# 模型：加载 SimCLR 预训练权重
def build_teacher_model(pretrained_path=None):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    if pretrained_path:
        # 加载 SimCLR 的 backbone 权重
        simclr_state = torch.load(pretrained_path, map_location='cpu')['state_dict']
        backbone_state = {k.replace('backbone.', ''): v for k, v in simclr_state.items() if 'backbone.' in k}
        model.load_state_dict(backbone_state, strict=False)
        print(f"Loaded SimCLR backbone from {pretrained_path}")

    return model.to(DEVICE)

# 训练
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

# 验证
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

# 主流程
mlflow.set_experiment("stanford_dogs_teacher")
with mlflow.start_run(run_name="resnet50_finetune"):
    mlflow.log_params({"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "pretrained": "simclr"})

    teacher = build_teacher_model('../runs/checkpoint_0100.pth.tar')
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing 是细粒度任务的小 trick
    optimizer = optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(teacher, train_loader, optimizer, criterion)
        test_loss, test_acc = eval_epoch(teacher, test_loader, criterion)
        scheduler.step()

        mlflow.log_metrics({
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc
        }, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(teacher.state_dict(), 'checkpoints/teacher_resnet50_best.pth')
            mlflow.pytorch.log_model(teacher, "teacher_best")

    print(f"Best Test Accuracy: {best_acc:.2f}%")
    mlflow.log_metric("best_test_acc", best_acc)