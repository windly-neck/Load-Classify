from src.dataset import load_all_data_by_category
import numpy as np


civilian_data = load_all_data_by_category("business")
np.save('data/classified/business_data.npy', civilian_data)
print(civilian_data.shape)
print(civilian_data[0])