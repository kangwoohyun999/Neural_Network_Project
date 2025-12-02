# dataset/mnist.py
import os
import gzip
import shutil
import struct
import numpy as np
from urllib.request import urlretrieve

MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "t10k_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_mnist(dest_dir="dataset"):
    os.makedirs(dest_dir, exist_ok=True)
    for name, url in MNIST_URLS.items():
        filename = os.path.join(dest_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {url} -> {filename}")
            urlretrieve(url, filename)
    print("Downloaded files (if not present).")

def _gunzip(src_path, dst_path):
    with gzip.open(src_path, 'rb') as f_in:
        with open(dst_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def _load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows*cols)
        return data

def _load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def load_mnist(normalize=True, flatten=True, dest_dir="dataset"):
    # Ensure downloaded and extracted
    download_mnist(dest_dir)
    # Extract gz if necessary
    paths = {}
    for key, url in MNIST_URLS.items():
        gz = os.path.join(dest_dir, os.path.basename(url))
        raw = gz[:-3]  # remove .gz
        if not os.path.exists(raw):
            print(f"Extracting {gz} -> {raw}")
            _gunzip(gz, raw)
        paths[key] = raw

    x_train = _load_idx_images(paths["train_images"])
    t_train = _load_idx_labels(paths["train_labels"])
    x_test  = _load_idx_images(paths["t10k_images"])
    t_test  = _load_idx_labels(paths["t10k_labels"])

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test  = x_test.astype(np.float32) / 255.0

    if not flatten:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

    return (x_train, t_train), (x_test, t_test)

if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = load_mnist()
    print("train:", x_train.shape, t_train.shape)
    print("test:", x_test.shape, t_test.shape)
