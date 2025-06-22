import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model import Week1DCNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train(
    model,
    X,
    y,
    batch_size=32,
    val_ratio=0.3,
    epochs=20,
    lr=1e-3,
    min_lr=1e-5,
    lr_scheduler='cosine',
    weight_decay=0.0,
    device='cpu',
    scheduler_step=10,
    scheduler_gamma=0.5,
    shuffle=True
):
    """
    X, y: numpy数组，X.shape=(N, 672, 2), y.shape=(N,)
    val_ratio: 验证集比例
    """
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=42, stratify=y)
    train_set = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_set = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = None
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_loss = total_loss / total
        train_losses.append(train_loss)
        acc = correct / total
        # 验证集loss
        val_loss, val_acc = evaluate(model, val_loader, device, return_loss=True)
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {cur_lr:.6f}")
    # 绘制loss曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()
    return model

def evaluate(model, loader, device='cpu', return_loss=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    avg_loss = total_loss / total if total > 0 else 0
    acc = correct / total if total > 0 else 0
    if return_loss:
        return avg_loss, acc
    return acc


# np.random.seed(42)
# torch.manual_seed(42)
# N = 100  # 样本数
# X = np.random.rand(N, 672, 2).astype(np.float32)
# y = np.random.randint(0, 3, size=(N,)).astype(np.int64)
# model = Week1DCNN(in_channels=2, num_classes=3)
# train(
#     model, X, y,
#     batch_size=16,
#     val_ratio=0.3,
#     epochs=5,
#     lr=1e-3,
#     min_lr=1e-5,
#     lr_scheduler='cosine',
#     weight_decay=1e-4,
#     device='cpu',
#     scheduler_step=2,
#     scheduler_gamma=0.7
# )
