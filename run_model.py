import os
import numpy as np
import torch
from src.model import Week1DCNN
from src.utils import get_file_label_from_samples, plot_classification_report
from src.train import train
from src.utils import split_all_categories_train_val_test

RESULTS_DIR = 'results/10times'
MODEL_PATH_FMT = os.path.join(RESULTS_DIR, 'model_{}.pth')
PRED_PATH_FMT = os.path.join(RESULTS_DIR, 'pred_{}.npy')
TOCLASSIFY_DIR = 'data/Toclassify'
CLASS_NAMES = ['business', 'civilian', 'industry']
NUM_CLASSES = 3
NUM_RUNS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_save_models():
    for i in range(NUM_RUNS):
        X_train, y_train, X_val, y_val, _, _ = split_all_categories_train_val_test(val_ratio=0.2, test_ratio=0, return_test_files=True, random_state=i+42)
        # 统计训练集和验证集类别分布
        train_counts = np.bincount(y_train, minlength=NUM_CLASSES)
        val_counts = np.bincount(y_val, minlength=NUM_CLASSES)
        print(f"第{i+1}次训练集类别分布: business={train_counts[0]}, civilian={train_counts[1]}, industry={train_counts[2]}")
        print(f"第{i+1}次验证集类别分布: business={val_counts[0]}, civilian={val_counts[1]}, industry={val_counts[2]}")
        model = Week1DCNN(in_channels=2, num_classes=NUM_CLASSES)
        model = train(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=30, lr=2e-5, device='cuda' if torch.cuda.is_available() else 'cpu')
        torch.save(model.state_dict(), MODEL_PATH_FMT.format(i+1))
        print(f"模型{i+1}已保存: {MODEL_PATH_FMT.format(i+1)}")

def predict_with_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    npy_files = sorted([f for f in os.listdir(TOCLASSIFY_DIR) if f.endswith('.npy')], key=lambda x: int(x.split('.')[0]))
    all_preds = []
    for i in range(NUM_RUNS):
        model = Week1DCNN(in_channels=2, num_classes=NUM_CLASSES)
        model_path = MODEL_PATH_FMT.format(i + 1)
        model.load_state_dict(torch.load(model_path, map_location=device))  # 加载权重
        model.to(device)  # 保证模型和输入数据在同一设备
        model.eval()
        preds = []
        for fname in npy_files:
            arr = np.load(os.path.join(TOCLASSIFY_DIR, fname))
            if arr.size == 0:
                preds.append(-1)
                continue
            arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            with torch.no_grad():
                outputs = model(arr_tensor)
                pred_labels = outputs.argmax(dim=1).cpu().numpy()
                pred_probs = outputs.softmax(dim=1).cpu().numpy()
            vote_label, prob_label = get_file_label_from_samples(pred_labels, pred_probs)
            preds.append(prob_label)
        all_preds.append(preds)
        np.save(PRED_PATH_FMT.format(i+1), preds)
        print(f"第{i+1}次预测已保存: {PRED_PATH_FMT.format(i+1)}")
    return np.array(all_preds), npy_files

def vote_and_report(all_preds, npy_files):
    from collections import Counter
    final_preds = []
    vote_distributions = []
    for idx in range(all_preds.shape[1]):
        votes = all_preds[:, idx]
        vote_count = Counter(votes)
        vote_label = vote_count.most_common(1)[0][0]
        final_preds.append(vote_label)
        vote_distributions.append(dict(vote_count))
    # 输出投票分布和最终预测
    print("编号\t10次预测分布\t最终预测")
    for fname, dist, pred in zip(npy_files, vote_distributions, final_preds):
        print(f"{fname}\t{dist}\t{CLASS_NAMES[pred] if pred!=-1 else '无效'}")
    # 保存最终预测结果
    np.save(os.path.join(RESULTS_DIR, 'final_vote_preds.npy'), final_preds)
    print(f"最终投票预测已保存: {os.path.join(RESULTS_DIR, 'final_vote_preds.npy')}")
    # 若有真实标签，可对比精度
    # plot_classification_report(y_true, final_preds, class_names=CLASS_NAMES)

def main():
    print("=== 1. 训练10个模型 ===")
    # train_and_save_models()
    print("=== 2. 10个模型对Toclassify数据预测 ===")
    all_preds, npy_files = predict_with_models()
    print("=== 3. 投票法汇总与报告 ===")
    vote_and_report(all_preds, npy_files)

if __name__ == '__main__':
    main()

