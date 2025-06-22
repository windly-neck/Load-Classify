import os
import numpy as np
from src.preprocess import DayDataLoader
from src.utils import print_memory_usage

def load_all_data_by_category(category, max_files=None):
    """
    category: "business" / "civilian" / "industry"
    max_files: 限制最多处理多少个文件，None为不限制
    返回: (总样本数, 96, 2) 的 numpy 数组
    """
    notebook_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    data_dir = os.path.join(notebook_dir, f'../data/raw/{category}/')
    data_dir = os.path.abspath(data_dir)
    files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') or f.endswith('.xls')]
    if max_files:
        files = files[:max_files]
    loader = DayDataLoader()
    all_samples = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        data = loader.get_weekly_samples_from_file(file_path)
        if data.shape[0] > 0:
            all_samples.append(data)
        print_memory_usage()
    if all_samples:
        return np.concatenate(all_samples, axis=0)
    else:
        return np.empty((0, 672, 2))

# 用法示例
# civilian_data = load_all_data_by_category("civilian")
# np.save('../data/classified/civilian_data.npy', civilian_data)
# print(civilian_data.shape)
# print(civilian_data[0])