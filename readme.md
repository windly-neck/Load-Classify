# 负荷分类项目说明

## 项目简介
本项目是上海交通大学机器学习课程的大作业。用于对电力负荷数据进行分类，支持商业（business）、民用（civilian）、工业（industry）三类负荷的自动识别。

## 使用方法

1. **准备数据**
   - 将原始数据文件（如Excel格式）放入项目目录下的`data/raw`文件夹中。
   - 按类别分别放入`business`、`civilian`、`industry`等子文件夹，例如：
     ```
     data/raw/business/xxx.xlsx
     data/raw/civilian/xxx.xlsx
     data/raw/industry/xxx.xlsx
     ```
   - 文件夹命名需为英文（business、civilian、industry）。

2. **数据预处理**
   - 在主目录下运行以下命令，自动将原始数据处理为标准格式：
     ```python
     from src.utils import preprocess_and_save_all_files
     preprocess_and_save_all_files('data/raw/business', 'data/processed/business')
     preprocess_and_save_all_files('data/raw/civilian', 'data/processed/civilian')
     preprocess_and_save_all_files('data/raw/industry', 'data/processed/industry')
     ```
   - 运行后会在`data/processed`下生成对应的npy文件。

3. **待分类数据处理**
   - 将待分类的原始Excel文件放入`data/待分类数据/`目录。
   - 可用`src.utils.preprocess_and_save_all_files('data/待分类数据', 'data/Toclassify')`将其批量转为npy格式，供模型分类使用。

4. **模型训练与预测**
   - 运行`run_model.py`，自动完成10次模型训练、预测、投票及结果保存，适合正式分类任务。
   - 运行`run.py`，只训练一次并输出一次分类结果，适合快速测试。
   - 运行`run_test.py`，对有标签数据进行训练和测试，便于调参和网络结构优化。

## 主要函数与脚本说明
- **数据预处理主函数**：`src.utils.preprocess_and_save_all_files`
- **批量分类主入口**：`run_model.py`（10次分类+投票）
- **单次分类主入口**：`run.py`（只分类一次）
- **有标签测试与调参**：`run_test.py`（用于有标签数据的训练/验证/测试）

## 目录结构示例
```
ML2/
├── data/
│   ├── raw/
│   │   ├── business/
│   │   ├── civilian/
│   │   └── industry/
│   ├── processed/
│   │   ├── business/
│   │   ├── civilian/
│   │   └── industry/
│   ├── 待分类数据/
│   └── Toclassify/
├── run_model.py
├── run.py
├── run_test.py
├── src/
│   └── utils.py
└── ...
```

## 依赖安装
```bash
pip install -r requirements.txt
```

## 其他说明
- 请确保原始数据文件夹命名规范。
- 结果文件和中间文件会自动生成在对应目录。
- 推荐先用`run_test.py`在有标签数据上调参，再用`run_model.py`批量分类。
