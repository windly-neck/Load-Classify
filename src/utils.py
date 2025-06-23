import psutil
import os
import time
import numpy as np
from src.preprocess import DayDataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"��前内存占用: {mem:.2f} MB")



_start_time = time.time()
def print_runtime():
    runtime = time.time() - _start_time
    print(f"当前程序运行时间: {runtime:.2f} 秒")

def shuffle_xy(X, y, random_state=42):
    idx = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    return X[idx], y[idx]

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
        if arr.size < 10:
            print(f"跳过过小样本: {fpath}")
            continue
        out_name = os.path.splitext(fname)[0] + '.npy'
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, arr)
        print(f"保存: {out_path}, shape={arr.shape}")

def plot_classification_report(y_true, y_pred, class_names=None, title='3-class Classification Report', save_path=None):
    """
    输入：
        y_true: 真实标签 (N,)
        y_pred: 预测标签 (N,)
        class_names: 类别名称列表，如['business', 'civilian', 'industry']
        save_path: 混淆矩阵图片保存路径（可选）
    输出：
        混���矩阵和分类报告可视化，并保存图片到result目录
    """
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = os.path.join('results', 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.show()
    print(f"混淆矩阵已保存到: {save_path}")
    print(classification_report(y_true, y_pred, target_names=class_names))

def get_test_folder_labels_and_preds(test_root, model, method='vote'):
    """
    遍历test_root下的business/civilian/industry三个子文件夹，对每个npy文件进行分类，
    返回所有文件的预测标签和真实标���。
    method: 'vote' 或 'prob'，决定用哪种方式作为文件的最终预测类别。
    返回：y_true, y_pred, file_names
    """
    from src.predict import predict
    y_true, y_pred, file_names = [], [], []
    class_map = {'business': 0, 'civilian': 1, 'industry': 2}
    for cls_name, cls_label in class_map.items():
        folder = os.path.join(test_root, cls_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith('.npy'):
                continue
            fpath = os.path.join(folder, fname)
            arr = np.load(fpath)
            if arr.size == 0:
                print(f"跳过空文件: {fpath}")
            pred_labels, pred_probs = predict(model, arr)  # (num_weeks,), (num_weeks, 3)
            vote_label, prob_label = get_file_label_from_samples(pred_labels, pred_probs)
            file_label = vote_label if method == 'vote' else prob_label
            y_true.append(cls_label)
            y_pred.append(file_label)
            file_names.append(f'{cls_name}/{fname}')
    return np.array(y_true), np.array(y_pred), file_names

def split_train_val_by_file(category, val_ratio=0.2, random_state=42):
    """
    对data/processed/{category}/下所有npy文件，按文件划分训练集和验证集。
    val_ratio: 验证集文件比例
    返回：X_train, y_train, X_val, y_val
    """
    import numpy as np
    import os
    np.random.seed(random_state)
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../data/processed/{category}/'))
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    n_files = len(files)
    n_val = max(1, int(n_files * val_ratio))
    idx = np.arange(n_files)
    np.random.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    def load_and_label(filelist, label):
        Xs = [np.load(os.path.join(folder, f)) for f in filelist]
        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 672, 2))
        y = np.full(X.shape[0], label, dtype=np.int64)
        return X, y
    label_map = {'business': 0, 'civilian': 1, 'industry': 2}
    label = label_map[category]
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    X_train, y_train = load_and_label(train_files, label)
    X_val, y_val = load_and_label(val_files, label)
    X_train, y_train = shuffle_xy(X_train, y_train)
    X_val, y_val = shuffle_xy(X_val, y_val)
    return X_train, y_train, X_val, y_val

def split_all_categories_train_val(val_ratio=0.2, random_state=42):
    """
    对三类数据分别按文件划分训练/验证集，并拼接返回
    返回：X_train, y_train, X_val, y_val
    """
    cats = ['business', 'civilian', 'industry']
    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
    for cat in cats:
        X_tr, y_tr, X_v, y_v = split_train_val_by_file(cat, val_ratio, random_state)
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_val_list.append(X_v)
        y_val_list.append(y_v)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    return X_train, y_train, X_val, y_val

def split_train_val_test_by_file(category, val_ratio=0.2, test_ratio=0.2, random_state=None, return_file_names=False):
    """
    对data/processed/{category}/下所有npy文件，按文件划分训练/验证/测试集。
    val_ratio: 验证集文件比例
    test_ratio: 测试集文件比例
    return_file_names: 若为True，返回测试集文件路径列表
    返回：X_train, y_train, X_val, y_val, X_test, y_test[, test_file_list]
    """
    import numpy as np
    import os
    if random_state is not None:
        np.random.seed(random_state)
    folder = os.path.join('data', 'processed', category)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    n_files = len(files)
    n_val = int(n_files * val_ratio)
    n_test = int(n_files * test_ratio)
    n_train = n_files - n_val - n_test
    idx = np.arange(n_files)
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    def load_and_label(filelist, label):
        Xs = [np.load(os.path.join(folder, f)) for f in filelist]
        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 672, 2))
        y = np.full(X.shape[0], label, dtype=np.int64)
        return X, y
    label_map = {'business': 0, 'civilian': 1, 'industry': 2}
    label = label_map[category]
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]
    X_train, y_train = load_and_label(train_files, label)
    X_val, y_val = load_and_label(val_files, label)
    X_test, y_test = load_and_label(test_files, label)
    X_train, y_train = shuffle_xy(X_train, y_train)
    X_val, y_val = shuffle_xy(X_val, y_val)
    X_test, y_test = shuffle_xy(X_test, y_test)
    test_files_rel = [os.path.join('data', 'processed', category, f) for f in test_files]
    if return_file_names:
        return X_train, y_train, X_val, y_val, X_test, y_test, test_files_rel
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test

def split_all_categories_train_val_test(val_ratio=0.2, test_ratio=0.2, random_state=None, return_test_files=False):
    """
    对三类数据分别按文件划分训练/验证/测试集，并拼接返回
    return_test_files: 若为True，则额外返回测试集文件路径和标签列表
    返回：X_train, y_train, X_val, y_val, X_test, y_test 或 (X_train, y_train, X_val, y_val, test_file_list, test_label_list)
    """
    cats = ['business', 'civilian', 'industry']
    X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list = [], [], [], [], [], []
    test_file_list, test_label_list = [], []
    for idx, cat in enumerate(cats):
        X_tr, y_tr, X_v, y_v, X_te, y_te, te_files = split_train_val_test_by_file(cat, val_ratio, test_ratio, random_state, return_file_names=True)
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        X_val_list.append(X_v)
        y_val_list.append(y_v)
        X_test_list.append(X_te)
        y_test_list.append(y_te)
        if return_test_files:
            test_file_list.extend(te_files)
            test_label_list.extend([idx]*len(te_files))
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    if return_test_files:
        return X_train, y_train, X_val, y_val, test_file_list, test_label_list
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test

def classify_and_save_excel(model, toclassify_dir, output_excel, device='cpu', batch_size=32, method='no-vote'):
    """
    遍历toclassify_dir下所有npy文件，调用模型预测类别，保存为excel。
    第一列为序号（文件名去掉扩展名），第二列为分类结果（中文）。
    """
    from src.predict import predict
    label_map = {0: '商业', 1: '居民', 2: '工业'}
    results = []
    files = sorted([f for f in os.listdir(toclassify_dir) if f.endswith('.npy')])
    for fname in files:
        fpath = os.path.join(toclassify_dir, fname)
        arr = np.load(fpath)
        if arr.size == 0:
            print(f"跳过空文件: {fpath}")
            continue
        pred_labels, pred_probs = predict(model, arr, device=device, batch_size=batch_size)
        from src.utils import get_file_label_from_samples
        vote_label, prob_label = get_file_label_from_samples(pred_labels, pred_probs)
        label = vote_label if method == 'vote' else prob_label
        label_cn = label_map.get(label, '未知')
        index = os.path.splitext(fname)[0]
        results.append([index, label_cn])
    df = pd.DataFrame(results, columns=['序号', '分类结果'])
    df.to_excel(output_excel, index=False)
    print(f"已保存分类结果到: {output_excel}")
