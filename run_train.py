from src.utils import preprocess_and_save_all_files

input_dir = "data/raw/test/civilian"  # 输入文件夹
output_dir = "data/processed/test/civilian"  # 输出文件夹
preprocess_and_save_all_files(input_dir, output_dir, use_weekly=True)