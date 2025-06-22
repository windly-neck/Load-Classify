import psutil
import os
import time
import numpy as np
from src.preprocess import DayDataLoader

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"当前内存占用: {mem:.2f} MB")



_start_time = time.time()
def print_runtime():
    runtime = time.time() - _start_time
    print(f"当前程序运行时间: {runtime:.2f} 秒")

def concat_samples_with_labels(data_business, data_civilian, data_industry, label_business=0, label_civilian=1, label_industry=2):
    """
    输入：三个类别的样本数组（已加载，shape=(N_i, 672, 2)），以及对应标签
    输出：拼接后的X, y
    X.shape=(N, 672, 2), y.shape=(N,)
    label_business, label_civilian, label_industry为对应类别标签
    """
    X = np.concatenate([data_business, data_civilian, data_industry], axis=0)
    y = np.concatenate([
        np.full(len(data_business), label_business, dtype=np.int64),
        np.full(len(data_civilian), label_civilian, dtype=np.int64),
        np.full(len(data_industry), label_industry, dtype=np.int64)
    ], axis=0)
    return X, y

def get_file_label_from_samples(pred_labels, pred_probs):
    """
    输入：
        pred_labels: shape=(N,) 每个周样本的预测类别
        pred_probs: shape=(N, num_classes) 每个周样本的预测概率
    输出：
        vote_label: 投票法得到的整体负荷类型
        prob_label: 概率平均法得到的整体负荷类型
    """
    # 投票法
    from collections import Counter
    vote_label = Counter(pred_labels).most_common(1)[0][0]
    # 概率平均法
    prob_label = pred_probs.mean(axis=0).argmax()
    return vote_label, prob_label

def preprocess_and_save_all_files(input_dir, output_dir, loader=None, use_weekly=True):
    """
    遍历input_dir下所有文件，对每个文件做预处理，保存为output_dir下同名的.npy文件。
    use_weekly: True则用get_weekly_samples_from_file，否则用load_and_process
    loader: 可传入DayDataLoader实例，否则自动创建
    """
    if loader is None:
        loader = DayDataLoader()
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith('.xlsx'):
            continue
        fpath = os.path.join(input_dir, fname)
        if use_weekly:
            arr = loader.get_weekly_samples_from_file(fpath)
        else:
            arr = loader.load_and_process(fpath)
        out_name = os.path.splitext(fname)[0] + '.npy'
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, arr)
        print(f"保存: {out_path}, shape={arr.shape}")

