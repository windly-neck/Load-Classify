from src.dataset import load_all_data_by_category
import numpy as np

civilian_data = load_all_data_by_category("civilian")
np.save('data/classfied/civilian_data.npy', civilian_data)
print(civilian_data.shape)
print(civilian_data[0])