import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model import Week1DCNN
import numpy as np
import matplotlib.pyplot as plt

def compute_class_weight(y):
    """
    计算每个类别的权重，类别样本数越少权重越大。
    返回: torch.tensor([w0, w1, w2], dtype=torch.float32)
    """
    import numpy as np
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes])
    weight = 1.0 / (counts + 1e-8)
    weight = weight / weight.sum() * len(classes)  # 归一化到类别数
    w_full = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    for c, w in zip(classes, weight):
        w_full[int(c)] = w
    import torch
    return torch.tensor(w_full, dtype=torch.float32)

def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=64,
    epochs=30,
    lr=1e-3,
    min_lr=1e-5,
    lr_scheduler='cosine',
    weight_decay=1e-3,  # 默认加大正则化
    device='cpu',
    scheduler_step=10,
    scheduler_gamma=0.5,
    shuffle=True,
    early_stopping_patience=5,
    verbose=True,
    use_class_weight=True
):
    """
    X_train, y_train: 训练集，numpy数组
    X_val, y_val: 验证集，numpy数组
    early_stopping_patience: 验证集loss超过patience个epoch不下降则提前停止
    """
    train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_set = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    model = model.to(device)
    if use_class_weight:
        class_weight = compute_class_weight(y_train).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        if verbose:
            print(f"使用类别权重: {class_weight.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = None
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
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
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {cur_lr:.6f}")
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break
    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
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
