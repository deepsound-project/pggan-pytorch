from torch.utils.data import Dataset
import numpy as np
import torch
import h5py
from utils import adjust_dynamic_range


class DepthDataset(Dataset):

    def __init__(self,
        h5_path,                                # e.g. 'cifar10-32.h5'
        depth_offset                = 0,
        resolution                  = None,     # e.g. 32 (autodetect if None)
        max_images                  = None,
        depth                       = 0,
        alpha                       = 1.0,
        range_in                    = (0, 255),
        range_out                   = (0, 1)):

        self.depth = depth
        self.alpha = alpha
        self.range_out = range_out

        # Open HDF5 file and select resolution.
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.resolution = resolution
        self.resolutions = sorted(list({v.shape[-1] for v in self.h5_file.values()}))
        self.resolution = self.resolutions[-1]
        self.depth_offset = depth_offset
        self.range_in = range_in
        self.h5_data = [self.h5_file['data{}x{}'.format(r, r)] for r in self.resolutions]

        # Look up shapes and dtypes.
        self.shape = self.h5_data[-1].shape
        if max_images is not None:
            self.shape = (min(self.shape[0], max_images),) + self.shape[1:]
        self.dtype = self.h5_data[0].dtype
        self.h5_data = [x[:self.shape[0]] for x in self.h5_data] # load everything into memory (!)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        data = self.h5_data[self.depth + self.depth_offset][item]
        if self.alpha < 1.0:
            c, h, w = data.shape
            t = data.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4)).repeat(2, 1).repeat(2, 2)
            data = (data + (t - data) * self.alpha)
        data = adjust_dynamic_range(data, self.range_in, self.range_out)
        return torch.from_numpy(data.astype('float32'))

    def close(self):
        self.h5_file.close()
