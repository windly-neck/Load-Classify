from src.utils import preprocess_and_save_all_files

input_dir = "data/raw/test/industry"  # 输入文件夹
output_dir = "data/processed/test/industry"  # 输出文件夹
preprocess_and_save_all_files(input_dir, output_dir, use_weekly=True)