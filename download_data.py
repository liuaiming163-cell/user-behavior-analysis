# download_data.py
import os
import zipfile


def download_and_extract_olist():
    """下载并解压 Olist 数据集到目标目录"""
    raw_data_dir = "data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)

    # 检查环境变量是否配置成功
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise ValueError("❌ 错误：未检测到 KAGGLE_API_TOKEN 环境变量，请先在终端运行设置命令！")

    # 延迟导入 kaggle 确保它能读取到刚刚注入的环境变量
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    dataset_name = "olistbr/brazilian-ecommerce"
    print(f"🚀 开始从 Kaggle 下载 Olist 电商数据集: {dataset_name} ...")

    # 下载 zip 压缩包
    api.dataset_download_files(dataset_name, path=raw_data_dir, quiet=False)
    zip_path = os.path.join(raw_data_dir, "brazilian-ecommerce.zip")

    # 解压文件
    print("📦 下载完成，正在解压中...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_data_dir)

    # 清理删除不需要的 zip 压缩包
    os.remove(zip_path)
    print(f"🎉 成功！所有真实的电商脏数据已准备就绪，存放于: {raw_data_dir}/")


if __name__ == "__main__":
    download_and_extract_olist()