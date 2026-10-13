"""
直接下载 MNIST 原始文件到 ./data，不依赖 torch/torchvision
"""
import urllib.request
import gzip
import os
from pathlib import Path

MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
MIRROR_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def download_mnist(data_dir="./data"):
    raw_dir = Path(data_dir) / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        out_path = raw_dir / fname[:-3]   # 去掉 .gz
        if out_path.exists():
            print(f"  已存在: {out_path.name}")
            continue

        url = MIRROR_URL + fname
        gz_path = raw_dir / fname
        print(f"  下载: {fname} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception:
            url = MNIST_URL + fname
            urllib.request.urlretrieve(url, gz_path)

        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            f_out.write(f_in.read())
        gz_path.unlink()
        print(f"OK  ({out_path.stat().st_size // 1024} KB)")

    print("MNIST 下载完成")


if __name__ == "__main__":
    print("下载 MNIST 到 ./data ...")
    download_mnist()
