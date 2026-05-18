import os

# 1. 定义需要创建的文件夹和文件
structure = {
    "src": ["cleaner.py", "funnel.py", "rfm.py", "visualizer.py"],
    "notebooks": ["01_data_cleaning.ipynb", "02_funnel_analysis.ipynb", "03_rfm_segmentation.ipynb",
                  "04_dashboard.ipynb"],
    "data/raw": [],
    "data/processed": [],
    "outputs": []
}

# 2. 自动化循环创建
for folder, files in structure.items():
    # 创建文件夹（确保多层目录如 data/raw 也能顺利创建）
    os.makedirs(folder, exist_ok=True)
    print(f"已创建文件夹: {folder}")

    # 在对应文件夹下创建空文件
    for file in files:
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("")  # 写入空内容挂载文件
            print(f"  └─ 已创建文件: {file_path}")

# 3. 创建根目录的 .gitignore 和 README.md
base_files = {
    ".gitignore": "# 数据文件过滤\ndata/raw/\ndata/processed/\noutputs/\n*.csv\n*.xlsx\n*.parquet\n\n# 环境过滤\n__pycache__/\n.venv/\nvenv/\n.idea/\n.vscode/",
    "README.md": "# user-behavior-analysis\n基于电商用户行为数据的全链路分析"
}

for file_name, content in base_files.items():
    if not os.path.exists(file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已创建根目录文件: {file_name}")

print("\n🎉 所有项目目录与文件脚手架已全部生成成功！")