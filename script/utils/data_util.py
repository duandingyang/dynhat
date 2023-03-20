import os
import numpy as np
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import pickle


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def loader(dataset='enron10', split_count=10):
    data_root = '../data/input/cached/{}/{}/'.format(dataset, split_count)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)
    # handel data function refers to https://github.com/marlin-codes/HTGN
    return torch.load(filepath)
