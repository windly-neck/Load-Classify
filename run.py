import os
import torch
import numpy as np
from src.model import Week1DCNN
from src.utils import classify_and_save_excel, shuffle_xy
from src.utils import concat_samples_with_labels, plot_classification_report, split_all_categories_train_val_test, get_file_label_from_samples

from src.train import train

if __name__ == '__main__':
    # 1. 以文件为单位划分训练集、验证集和测试集，避免数据泄漏
    # 假设 split_all_categories_train_val_test 返回 (X_train, y_train, X_val, y_val, test_file_list, test_label_list)
    X_train, y_train, X_val, y_val, test_file_list, test_label_list = split_all_categories_train_val_test(val_ratio=0.2,
                                                                                                          test_ratio=0,
                                                                                                          return_test_files=True)
    print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}, 测试集场站数: {len(test_file_list)}")
    print(
        f"训练集标签分布: {np.bincount(y_train)}，验证集标签分布: {np.bincount(y_val)}，测试集标签分布: {np.bincount(test_label_list)}")

    # 2.1 可选：直接加载已保存模型并预测
    import torch

    load_model = True  # 设置为True则直接加载模型，不重新训练
    model_path = 'models/work_cnn.pth'
    if load_model and os.path.exists(model_path):
        model = Week1DCNN(in_channels=2, num_classes=3)
        model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"已加载模型权重: {model_path}")
    else:
        # 2. 训练模型
        model = Week1DCNN(in_channels=2, num_classes=3)
        model = train(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=30, lr=2e-5,
                      device='cuda' if torch.cuda.is_available() else 'cpu')
        # 3. 保存模型权重
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")


    # 预测并保存结果
    toclassify_dir = 'data/Toclassify'
    output_excel = '分类结果.xlsx'
    classify_and_save_excel(model, toclassify_dir, output_excel)
    print("全部待分类数据已完成预测，结果已保存为Excel。")

