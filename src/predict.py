import torch
import numpy as np
from src.model import Week1DCNN

def predict(model, X, device='cpu', batch_size=32):
    """
    输入：
        model: 已加载好权重的Week1DCNN模型
        X: numpy数组，shape=(N, 672, 2)
        device: 推理设备
        batch_size: 批量大小
    输出：
        pred_labels: shape=(N,) 的预测类别（int）
        pred_probs: shape=(N, 3) 的预测概率
    """
    model = model.to(device)
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X_tensor[i:i+batch_size].to(device)
            out = model(x_batch)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            preds.append(pred.cpu().numpy())
            probs.append(prob.cpu().numpy())
    pred_labels = np.concatenate(preds, axis=0)
    pred_probs = np.concatenate(probs, axis=0)
    return pred_labels, pred_probs



# from src.model import Week1DCNN
# from src.train import train
# import os
# np.random.seed(42)
# torch.manual_seed(42)
# N = 50
# X = np.random.rand(N, 672, 2).astype(np.float32)
# y = np.random.randint(0, 3, size=(N,)).astype(np.int64)
# # 情况1：训练好的模型
# model1 = Week1DCNN(in_channels=2, num_classes=3)
# train(model1, X, y, epochs=2, batch_size=8, val_ratio=0.2)
# pred_labels1, pred_probs1 = predict(model1, X)
# print('训练后模型的预测类别:', pred_labels1)
# print('训练后模型的预测概率:', pred_probs1)
# # # 保存模型
# # save_path = 'models/test_week1dcnn.pth'
# # os.makedirs(os.path.dirname(save_path), exist_ok=True)
# # torch.save(model1.state_dict(), save_path)
# # # 情况2：直接加载保存好的模型
# # model2 = Week1DCNN(in_channels=2, num_classes=3)
# # model2.load_state_dict(torch.load(save_path, map_location='cpu'))
# # pred_labels2, pred_probs2 = predict(model2, X)
# # print('加载保存模型的预测类别:', pred_labels2)
# # print('加载保存模型的预测概率:', pred_probs2)
