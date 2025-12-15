import os
import opendatasets as od

# Kaggle 数据集 URL
dataset_url = 'https://www.kaggle.com/splcher/animefacedataset'
DATA_DIR = './data/animefacedataset'

def download_dataset(url, target_dir):

    if os.path.exists(target_dir):
        print(f"Dataset already downloaded at: {target_dir}")
        return

    print("Downloading Anime Face Dataset...")

    try:
        od.download(url, data_dir=target_dir, force=False, quiet=False)  # quiet=False 会显示 tqdm
        print(f"Dataset downloaded successfully at: {target_dir}")
    except Exception as e:
        print("下载失败，请检查网络或 Kaggle Key 是否正确")
        print("错误信息：", e)
        print("你也可以手动下载数据集并解压到 './data/animefacedataset'")

if __name__ == "__main__":
    download_dataset(dataset_url, DATA_DIR)
