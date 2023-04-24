import csv

import numpy as np
import time
import torch
import os
from torch.utils.data import Dataset
import math


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.data = torch.stack(transposed_data[0], 0)
        self.mask = torch.stack(transposed_data[1], 0)
        self.label = torch.stack(transposed_data[2], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.mask = self.mask.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)


class BaseSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.skeleton_list = []

    def __getitem__(self, item):
        return item

    def __len__(self):
        raise NotImplementedError


class CustomSkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, path, batch_size, feature_dim, data_keys):
        super(CustomSkeletonDataset).__init__()
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.data_keys = data_keys
        self.dir_list = np.asarray([skeleton_file for skeleton_file in os.listdir(path)])
        floor, mod = divmod(len(self.dir_list), int(batch_size))
        if mod > 0:
            np.random.shuffle(self.dir_list)
            self.dir_list = self.dir_list[:-mod]
        # self.dir_list = np.reshape(np.asarray(self.dir_list), (floor, batch_size, self.dir_list.shape[-1]))
        self.start = 0
        self.end = len(self.dir_list)
        self._load_data()

    def shuffle(self):
        idx_batch = torch.randperm(self.data.shape[0])
        idx_seq = torch.randperm(self.data.shape[1])
        self.data = self.data[idx_batch][:, idx_seq]
        self.mask = self.mask[idx_batch][:, idx_seq]
        self.label = self.label[idx_batch][:, idx_seq]

    def _load_data(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            # random.shuffle(dir_list)
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        # print(self.dir_list[iter_start:iter_end])
        data_tuple = [load_file(batch, feature_dim=self.feature_dim, joint_keys=self.data_keys)
                      for batch in self.dir_list[iter_start:iter_end]]
        data = []
        mask = []
        label = []
        for patch in data_tuple:
            data.append(patch[0])
            mask.append(patch[1])
            label.append(patch[2])
        data = np.asarray(data).astype(np.float32)
        mask = np.asarray(mask).astype(np.float32)
        label = np.asarray(label).astype(np.int32)
        self.data = torch.as_tensor(data)
        self.mask = torch.as_tensor(mask)
        self.label = torch.as_tensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.label[idx]