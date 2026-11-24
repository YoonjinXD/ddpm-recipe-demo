import gzip
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch

MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist'

FILES = dict(
    train = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    test = ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
)

def exists(v):
    return v is not None

def read_idx(path, header):
    with gzip.open(path) as f:
        return np.frombuffer(f.read(), np.uint8, offset = header)

def mnist(split = 'train', digit = None, root = 'data/mnist'):
    """images in [-1, 1] with shape (n, 1, 28, 28), and their labels"""

    root = Path(root)
    root.mkdir(parents = True, exist_ok = True)

    for name in FILES[split]:
        if not (root / name).exists():
            urlretrieve(f'{MIRROR}/{name}', root / name)

    images_file, labels_file = FILES[split]
    images = read_idx(root / images_file, 16).reshape(-1, 1, 28, 28)
    labels = read_idx(root / labels_file, 8).astype(np.int64)

    if exists(digit):
        images, labels = images[labels == digit], labels[labels == digit]

    return torch.from_numpy(images).float() / 127.5 - 1., torch.from_numpy(labels)
