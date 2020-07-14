import os

import numpy as np
import torch
import torchvision as tv
from torch.utils import data

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

SAVE_DIR = './results/'
os.makedirs(SAVE_DIR, exist_ok=True)


def batch_images_to_one(batches_images):
    n_square_elements = int(np.sqrt(batches_images.shape[0]))
    rows_images = np.split(np.squeeze(batches_images), n_square_elements)
    return np.vstack([np.hstack(row_images) for row_images in rows_images])


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


train_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=128, shuffle=True,
    drop_last=True, num_workers=2,
)

test_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)
