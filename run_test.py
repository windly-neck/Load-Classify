import os
import numpy as np
from src.model import Week1DCNN
from src.utils import concat_samples_with_labels, plot_classification_report, split_all_categories_train_val_test, get_file_label_from_samples
from src.train import train

if __name__ == '__main__':
    # 1. 以文件为单位划分训练集、验证集和测试集，避免数据泄漏
    # 假设 split_all_categories_train_val_test 返回 (X_train, y_train, X_val, y_val, test_file_list, test_label_list)
    X_train, y_train, X_val, y_val, test_file_list, test_label_list = split_all_categories_train_val_test(val_ratio=0.2, test_ratio=0.2, return_test_files=True)
    print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}, 测试集场站数: {len(test_file_list)}")
    print(f"训练集标签分布: {np.bincount(y_train)}，验证集标签分布: {np.bincount(y_val)}，测试集标签分布: {np.bincount(test_label_list)}")

    # 2.1 可选：直接加载已保存模型并预测
    import torch
    load_model = False  # 设置为True则直接加载模型，不重新训练
    model_path = 'models/test_cnn.pth'
    if load_model and os.path.exists(model_path):
        model = Week1DCNN(in_channels=2, num_classes=3)
        model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"已加载模型权重: {model_path}")
    else:
        # 2. 训练模型
        model = Week1DCNN(in_channels=2, num_classes=3)
        model = train(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=30, lr=2e-5, device='cuda' if torch.cuda.is_available() else 'cpu')
        # 3. 保存模型权重
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")

    # 4. 测试集以文件为单位聚合预测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    y_true, y_pred, file_names = [], [], []
    for fpath, true_label in zip(test_file_list, test_label_list):
        arr = np.load(fpath)
        if arr.size == 0:
            print(f"跳过空文件: {fpath}")
            continue
        arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(arr_tensor)
            pred_labels = outputs.argmax(dim=1).cpu().numpy()
            pred_probs = outputs.softmax(dim=1).cpu().numpy()
        vote_label, prob_label = get_file_label_from_samples(pred_labels, pred_probs)
        file_label = prob_label # 可选: 'vote' 或 'prob'
        y_true.append(true_label)
        y_pred.append(file_label)
        file_names.append(os.path.basename(fpath))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5. 结果分析与可视化
    class_names = ['business', 'civilian', 'industry']
    plot_classification_report(y_true, y_pred, class_names=class_names)

    # 6. 输出每个文件的预测与真实标签
    for fname, yt, yp in zip(file_names, y_true, y_pred):
        print(f"文件: {fname} | 真实: {class_names[yt]} | 预测: {class_names[yp]}")
