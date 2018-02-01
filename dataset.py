from torch.utils.data import Dataset
import numpy as np
import torch
import math
from utils import adjust_dynamic_range
try:
    from scipy.misc import imread
except ImportError as e:
    print('Unable to load scipy: {}\nDefaultImageFolderDataset won\'t work.'.format(e))
from functools import reduce
try:
    import h5py
except ImportError as e:
    print('Unable to load h5py: {}.\nOldH5Dataset won\'t work.'.format(e))
import os
LIBROSA_LOADED = False
try:
    import librosa as lbr
    sound_load_fun = lambda path, freq, dtype: lbr.load(path, freq, dtype=dtype)
    LIBROSA_LOADED = True
except ImportError as e:
    print('Unable to load librosa: {}.\nSoundImageDataset may work only in raw mode.')
try:
    import soundfile as sf
    sound_load_fun = lambda path, _, dtype: sf.read(path, dtype=dtype)  # sf does not support sr in read, but infers
except ImportError as e:
    errstr = 'Switching sound loading to librosa.load' if LIBROSA_LOADED else 'SoundImageDataset won\'t work at all.'
    print('Unable to load soundfile: {}.\n{}'.format(e, errstr))


class DepthDataset(Dataset):

    def __init__(self,
         model_dataset_depth_offset=2,  # we start with 4x4 resolution instead of 1x1
         model_initial_depth=0,
         alpha=1.0,
         range_in=(0, 255),
         range_out=(-1, 1)):

        self.model_depth = model_initial_depth
        self.alpha = alpha
        self.range_out = range_out
        self.model_dataset_depth_offset = model_dataset_depth_offset
        self.range_in = range_in

    @property
    def data(self):
        raise NotImplementedError()

    @property
    def shape(self):
        return self.data[-1].shape

    def alpha_fade(self, data):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        datapoint = self.data[self.model_depth + self.model_dataset_depth_offset][item]
        if self.alpha < 1.0:
            datapoint = self.alpha_fade(datapoint)
        # print(data[0])
        datapoint = adjust_dynamic_range(datapoint, self.range_in, self.range_out)
        # print(data[0])
        return torch.from_numpy(datapoint.astype('float32'))


class OldH5Dataset(DepthDataset):

    def __init__(self,
         h5_path='datasets/cifar10-32.h5',
         model_dataset_depth_offset  = 2,  # we start with 4x4 resolution instead of 1x1
         max_images                  = None,
         model_initial_depth         = 0,
         alpha                       = 1.0,
         range_in                    = (0, 255),
         range_out                   = (-1, 1)):

        super(OldH5Dataset, self).__init__(model_dataset_depth_offset, model_initial_depth, alpha, range_in, range_out)

        # Open HDF5 file and select resolution.
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.resolutions = sorted(list({v.shape[-1] for v in self.h5_file.values()}))
        self.resolution = self.resolutions[-1]
        self.h5_data = [self.h5_file['data{}x{}'.format(r, r)] for r in self.resolutions]
        self.h5_shape = self.h5_data[-1].shape
        if max_images is not None:
            self.h5_shape = (min(self.shape[0], max_images),) + self.shape[1:]
        self.dtype = self.h5_data[0].dtype
        self.h5_data = [x[:self.shape[0]] for x in self.h5_data] # load everything into memory (!)

    @property
    def data(self):
        return self.h5_data

    @property
    def shape(self):
        return self.h5_shape

    def __len__(self):
        return self.shape[0]

    def alpha_fade(self, datapoint):
        c, h, w = datapoint.shape
        t = datapoint.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4)).repeat(2, 1).repeat(2, 2)
        datapoint = (datapoint + (t - datapoint) * (1 - self.alpha))
        return datapoint

    def close(self):
        self.h5_file.close()


class FolderDataset(DepthDataset):

    def __init__(self,
         dir_path,  # e.g. 'samples/'
         max_dataset_depth           = None,
         create_unused_depths        = False,
         preload                     = False,
         model_dataset_depth_offset  = 2,  # we start with 4x4 resolution instead of 1x1
         model_initial_depth         = 0,
         alpha                       = 1.0,
         range_in                    = (0, 255),
         range_out                   = (-1, 1)):

        super(FolderDataset, self).__init__(model_dataset_depth_offset, model_initial_depth, alpha, range_in, range_out)
        self.dir_path = dir_path
        self.files = sorted(list(map(lambda x: os.path.join(dir_path, x), os.listdir(dir_path))))
        self.max_dataset_depth = max_dataset_depth
        if self.max_dataset_depth is None:
            self.max_dataset_depth = self.infer_max_dataset_depth(self.load_file(0))
        self.preload = preload
        self.min_dataset_depth = 0 if preload and create_unused_depths else self.model_dataset_depth_offset
        self.datas = [None] * (self.max_dataset_depth + 1)
        if self.preload:
            print('Preloading data...', self.max_dataset_depth)

            def get_datapoint(i, cur_depth):
                if cur_depth == self.max_dataset_depth:
                    return self.load_file(i)
                return self.get_datapoint_version(self.datas[cur_depth + 1][i], cur_depth + 1, cur_depth)
            for cur_depth in range(self.max_dataset_depth, self.min_dataset_depth - 1, -1):
                print('Preloading depth: {}'.format(cur_depth))
                tmp_data = None
                data_shape = None
                for i in range(len(self.files)):
                    print('Preloading item: {}/{} in depth: {}'.format(i, len(self.files), cur_depth))
                    datapoint = get_datapoint(i, cur_depth)
                    if not data_shape:
                        data_shape = datapoint.shape
                        shape = (len(self.files),) + data_shape
                        tmp_data = np.zeros(shape, dtype=datapoint.dtype)
                    else:
                        assert datapoint.shape == data_shape
                    tmp_data[i] = datapoint
                self.datas[cur_depth] = tmp_data
        self.description = {
            'len': len(self),
            'shape': self.datas[-1].shape if self.preload else 'unknown',
            'depth_range': ((self.min_dataset_depth if self.preload else 'unknown'), self.max_dataset_depth)
        }

    @property
    def data(self):
        if self.preload:
            return self.datas
        raise AttributeError('FolderDataset.data property only accessible if preload is on.')

    @property
    def shape(self):
        if self.preload:
            return super(FolderDataset, self).shape
        return (len(self),) + self.load_file(0).shape

    def __len__(self):
        return len(self.files)

    def get_datapoint_version(self, datapoint, datapoint_depth, target_depth):
        if datapoint_depth == target_depth:
            return datapoint
        return self.create_datapoint_from_depth(datapoint, datapoint_depth, target_depth)

    def create_datapoint_from_depth(self, datapoint, datapoint_depth, target_depth):
        raise NotImplementedError()

    def load_file(self, item):
        raise NotImplementedError()

    def infer_max_dataset_depth(self, datapoint):
        raise NotImplementedError()

    def __getitem__(self, item):
        if self.preload:  # we have access to the data attribute
            return super(FolderDataset, self).__getitem__(item)
        datapoint = self.load_file(item)
        datapoint = self.get_datapoint_version(datapoint, self.max_dataset_depth,
                                               self.model_depth + self.model_dataset_depth_offset)
        datapoint = self.alpha_fade(datapoint)
        datapoint = adjust_dynamic_range(datapoint, self.range_in, self.range_out)
        return torch.from_numpy(datapoint.astype('float32'))


class DefaultImageFolderDataset(FolderDataset):

    def __init__(self,
                 dir_path='datasets/images',
                 max_dataset_depth=None,
                 create_unused_depths=False,
                 preload=False,
                 model_dataset_depth_offset=2,  # we start with 4x4 resolution instead of 1x1
                 model_initial_depth=0,
                 alpha=1.0,
                 range_in=(0, 255),
                 range_out=(-1, 1),
                 imread_mode='L',
                 scale_factor=2):
        self.imread_mode = imread_mode
        self.scale_factor = scale_factor
        super(DefaultImageFolderDataset, self).__init__(dir_path, max_dataset_depth, create_unused_depths, preload,
                                                 model_dataset_depth_offset, model_initial_depth, alpha, range_in,
                                                 range_out)

    def load_file(self, item):
        im = imread(self.files[item], mode=self.imread_mode)
        if im.ndim == 2:
            im = im[np.newaxis]
        if im.ndim == 3:
            im = im.transpose(2, 0, 1)
        assert im.ndim == 3
        return im

    def alpha_fade(self, datapoint):
        c, h, w = datapoint.shape
        t = datapoint.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4)).repeat(2, 1).repeat(2, 2)
        datapoint = (datapoint + (t - datapoint) * (1 - self.alpha))
        return datapoint

    def create_datapoint_from_depth(self, datapoint, datapoint_depth, target_depth):
        datapoint = datapoint.astype(np.float32)
        depthdiff = (datapoint_depth - target_depth)
        datapoint = reduce(lambda acc, x: acc + datapoint[:, x[0]::(self.scale_factor**depthdiff),
                                          x[1]::(self.scale_factor**depthdiff)],
                           [(a,b) for a in range(self.scale_factor) for b in range(self.scale_factor)], 0)\
                    / (self.scale_factor ** 2)
        return np.uint8(np.clip(np.round(datapoint), self.range_in[0], self.range_in[1]))

    def infer_max_dataset_depth(self, datapoint):
        print(datapoint.shape)
        return int(math.log(datapoint.shape[-1], self.scale_factor))


class SoundImageDataset(DefaultImageFolderDataset):

    def __init__(self,
                 dir_path='datasets/piano',
                 max_dataset_depth=None,
                 create_unused_depths=False,
                 preload=False,
                 model_dataset_depth_offset=2,  # we start with 4x4 resolution instead of 1x1
                 model_initial_depth=0,
                 alpha=1.0,
                 range_in=(0, 255),
                 range_out=(-1, 1),
                 scale_factor=2,
                 n_fft=1024,       # these matter only for spectrogram img_mode
                 hop_length=128,   #
                 frequency=16000,  #
                 img_mode='abslog'
                 ):
        assert n_fft == 2 ** int(np.log2(n_fft))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frequency = frequency
        self.img_mode = img_mode
        super(SoundImageDataset, self).__init__(dir_path, max_dataset_depth, create_unused_depths, preload,
                                                model_dataset_depth_offset, model_initial_depth, alpha, range_in,
                                                range_out, scale_factor=scale_factor)

    def load_file(self, item):
        s, _ = sound_load_fun(self.files[item], self.frequency, dtype='float32')
        if s.ndim == 2:  # stereo to mono
            s = (s.sum(axis=1)) / 2
        if self.img_mode == 'raw':
            size = int(np.log2(np.sqrt(s.shape[0])))
            s = s[:(2 ** size)**2].reshape((2 ** size, 2 ** size))
        else:
            s = lbr.stft(s, self.n_fft, self.hop_length)
            s = s[:self.n_fft // 2, :self.n_fft // 2]
            if self.img_mode == 'abslog':
                s = np.log(1 + np.abs(s))
            else:
                s = np.log(1 + np.abs(s.real)) * np.sign(s)
        s = np.uint8(adjust_dynamic_range(s, (s.min(), s.max()), self.range_in))
        return s[np.newaxis]

    def create_datapoint_from_depth(self, datapoint, datapoint_depth, target_depth):
        if self.img_mode != 'raw':
            return super(SoundImageDataset, self).create_datapoint_from_depth(datapoint, datapoint_depth, target_depth)
        depthdiff = (datapoint_depth - target_depth)
        return datapoint[:, ::2**depthdiff, ::2**depthdiff]
