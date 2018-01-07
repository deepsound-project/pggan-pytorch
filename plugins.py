import matplotlib
matplotlib.use('Agg')

import PIL.Image
from network import Generator

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.trainer.plugins.plugin import Plugin
from torch.utils.trainer.plugins.monitor import Monitor
from torch.utils.trainer.plugins import LossMonitor, Logger
from torch.utils.data import DataLoader
from datetime import timedelta

from librosa.output import write_wav
from matplotlib import pyplot

import numpy as np

from glob import glob
import sys
import os
import pickle
import time
import utils


class DepthManager(Plugin):

    def __init__(self,
                 create_dataloader_fun,
                 create_rlg,
                 minibatch_default,
                 minibatch_overrides,
                 tick_kimg_default,
                 tick_kimg_overrides,
                 lod_training_nimg=100*1000,
                 lod_transition_nimg=100*1000):
        super(DepthManager, self).__init__([(1, 'iteration')])
        self.minibatch_default = minibatch_default
        self.minibatch_overrides = minibatch_overrides
        self.tick_kimg_default = tick_kimg_default
        self.tick_kimg_overrides = tick_kimg_overrides
        self.create_dataloader_fun = create_dataloader_fun
        self.create_rlg = create_rlg
        self.lod_training_nimg = lod_training_nimg
        self.lod_transition_nimg = lod_transition_nimg
        self.trainer = None
        self.depth = -1
        self.alpha = -1

    def register(self, trainer):
        self.trainer = trainer
        self.trainer.stats['minibatch_size'] = self.minibatch_default
        self.iteration()

    def iteration(self, *args):
        cur_nimg = self.trainer.cur_nimg
        full_passes, remaining_nimg = divmod(cur_nimg, self.lod_training_nimg + self.lod_transition_nimg)
        train_passes_rem, remaining_nimg = divmod(remaining_nimg, self.lod_training_nimg)
        depth = full_passes + train_passes_rem
        alpha = remaining_nimg / self.lod_transition_nimg if train_passes_rem > 0 else 1.0
        dataset = self.trainer.dataset
        if depth != self.depth:
            self.trainer.stats['minibatch_size'] = self.minibatch_default
            self.trainer.D.depth = self.trainer.G.depth = dataset.depth = depth
            self.depth = depth
            minibatch_size = self.minibatch_overrides.get(depth, self.minibatch_default)
            self.trainer.dataiter = iter(self.create_dataloader_fun(minibatch_size))
            self.trainer.random_latents_generator = self.create_rlg(minibatch_size)
            # print(self.trainer.random_latents_generator().size())
            tick_duration_kimg = self.tick_kimg_overrides.get(depth, self.tick_kimg_default)
            self.trainer.tick_duration_nimg = tick_duration_kimg * 1000
        if alpha != self.alpha:
            self.trainer.D.alpha = self.trainer.G.alpha = dataset.alpha = alpha
            self.alpha = alpha
        self.trainer.stats['depth'] = depth
        self.trainer.stats['alpha'] = alpha
        self.trainer.stats['lod'] = 9 - 2 - depth - alpha + 1


class LRScheduler(Plugin):

    def __init__(self,
                 lr_scheduler_d,
                 lr_scheduler_g):
        super(LRScheduler, self).__init__([(1, 'iteration')])
        self.lrs_d = lr_scheduler_d
        self.lrs_g = lr_scheduler_g

    def register(self, trainer):
        self.trainer = trainer
        self.iteration()

    def iteration(self, *args):
        self.lrs_d.step(self.trainer.cur_nimg / 1000.)
        self.lrs_g.step(self.trainer.cur_nimg / 1000.)


class EfficientLossMonitor(LossMonitor):

    def __init__(self, loss_no, stat_name):
        super(EfficientLossMonitor, self).__init__()
        self.loss_no = loss_no
        self.stat_name = stat_name

    def _get_value(self, iteration, *args):
        return args[self.loss_no] if self.loss_no < 2 else args[self.loss_no].mean()


class AbsoluteTimeMonitor(Plugin):

    stat_name = 'time'

    def __init__(self, base_time=0):
        super(AbsoluteTimeMonitor, self).__init__([(1, 'epoch')])
        self.base_time = base_time
        self.start_time = time.time()
        self.epoch_start = self.start_time
        self.start_nimg = None
        self.epoch_time = 0
        self.log_format = '{}'

    def register(self, trainer):
        self.trainer = trainer
        self.start_nimg = trainer.cur_nimg

    def epoch(self, epoch_index):
        cur_time = time.time()
        tick_time = cur_time - self.epoch_start
        self.epoch_start = cur_time
        kimg_time = tick_time / (self.trainer.cur_nimg - self.start_nimg) * 1000
        self.start_nimg = self.trainer.cur_nimg
        self.trainer.stats['time'] = timedelta(seconds=time.time() - self.start_time + self.base_time)
        self.trainer.stats['sec/tick'] = tick_time
        self.trainer.stats['sec/kimg'] = kimg_time


class SaverPlugin(Plugin):

    last_pattern = 'network-snapshot-{}-{:06}.dat'

    def __init__(self, checkpoints_path, keep_old_checkpoints, network_snapshot_ticks):
        super().__init__([(network_snapshot_ticks, 'epoch')])
        self.checkpoints_path = checkpoints_path
        self.keep_old_checkpoints = keep_old_checkpoints
        self._best_val_loss = float('+inf')

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        if not self.keep_old_checkpoints:
            self._clear(self.last_pattern.format('*', '*'))
        for model, name in [(self.trainer.G, 'generator'), (self.trainer.D, 'discriminator')]:
            torch.save(
                model,
                os.path.join(
                    self.checkpoints_path,
                    self.last_pattern.format(name, self.trainer.cur_nimg // 1000)
                )
            )

    def _clear(self, pattern):
        pattern = os.path.join(self.checkpoints_path, pattern)
        for file_name in glob(pattern):
            os.remove(file_name)


class SampleGenerator(Plugin):

    def __init__(self, samples_path, image_grid_size, drange, image_snapshot_ticks, resolution, sample_fn):
        super(SampleGenerator, self).__init__([(image_snapshot_ticks, 'epoch')])
        self.samples_path = samples_path
        self.image_grid_size = image_grid_size
        self.resolution = resolution
        self.sample_fn = sample_fn
        self.drange = drange

    def register(self, trainer):
        self.trainer = trainer
        self.G = trainer.G

    def gen_fn(self):
        x = Variable(self.sample_fn(np.prod(self.image_grid_size)))
        z = self.G.forward(x.cuda())
        if z.size(-1) < self.resolution:
            z = F.upsample(z, size=(self.resolution, self.resolution))
        z = z.cpu().data.numpy()
        return z

    # def first_gen(self, x):
    #     tmp = self.G.depth, self.G.alpha
    #     (self.G.depth, self.G.alpha) = self.max_depth, 1.0
    #     z = self.gen_fn(x)
    #     (self.G.depth, self.G.alpha) = tmp
    #     return z

    def _create_image_grid(self, images): # TODO test with 3-channel
        w, h = self.image_grid_size
        return np.vstack([
                    np.vstack([images[(j*w):(j+1)*w,:,i].ravel() for i in range(images.shape[-1])])
                    for j in range(h)
               ])

    def create_image_grid(self, images):
        assert images.ndim == 3 or images.ndim == 4
        num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
        grid_size = self.image_grid_size
        if grid_size is not None:
            grid_w, grid_h = tuple(grid_size)
        else:
            grid_w = max(int(np.ceil(np.sqrt(num))), 1)
            grid_h = max((num - 1) // grid_w + 1, 1)

        grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
        for idx in range(num):
            x = (idx % grid_w) * img_w
            y = (idx // grid_w) * img_h
            grid[..., y: y + img_h, x: x + img_w] = images[idx]
        return grid

    def convert_to_pil_image(self, image):
        format = 'RGB'
        print(image.shape)
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0]  # grayscale CHW => HW
                format = 'L'
            else:
                image = image.transpose(1, 2, 0)  # CHW -> HWC
                format = 'RGB'

        image = utils.adjust_dynamic_range(image, self.drange, (0, 255))
        image = image.round().clip(0, 255).astype(np.uint8)
        print(image.shape)
        print(image)
        print(format)
        return PIL.Image.fromarray(image, format)

    def epoch(self, epoch_index):
        out = self.gen_fn()
        im = self.create_image_grid(out)
        im = self.convert_to_pil_image(im)
        im.save(os.path.join(self.samples_path, 'fakes{:06}.png'.format(self.trainer.cur_nimg // 1000)))

#
# class BaseGeneratorPlugin(Plugin):
#
#     pattern = 'ep{}-s{}.wav'
#
#     def __init__(self, samples_path, sample_rate,
#                  sample_id_fn=(lambda i: i + 1)):
#         super().__init__([(1, 'epoch')])
#         self.samples_path = samples_path
#         self.sample_rate = sample_rate
#         self.sample_id_fn = sample_id_fn
#
#     def _generate(self):
#         raise NotImplementedError()
#
#     def epoch(self, epoch_index):
#         samples = self._generate().cpu().float().numpy()
#         (n_samples, _) = samples.shape
#         for i in range(n_samples):
#             write_wav(
#                 os.path.join(
#                     self.samples_path,
#                     self.pattern.format(epoch_index, self.sample_id_fn(i))
#                 ),
#                 samples[i, :], sr=self.sample_rate, norm=True
#             )
#
#
# class UnconditionalGeneratorPlugin(BaseGeneratorPlugin):
#
#     def __init__(self, samples_path, n_samples, sample_length, sample_rate):
#         super().__init__(samples_path, sample_rate)
#         self.n_samples = n_samples
#         self.sample_length = sample_length
#
#     def register(self, trainer):
#         self.generator = Generator(trainer.model.model, trainer.cuda)
#
#     def _generate(self):
#         return self.generator(self.n_samples, self.sample_length)


class CometPlugin(Plugin):

    def __init__(self, experiment, fields):
        super().__init__([(1, 'epoch')])

        self.experiment = experiment
        self.fields = [
            field if type(field) is tuple else (field, 'last')
            for field in fields
        ]

    def register(self, trainer):
        self.trainer = trainer

    def epoch(self, epoch_index):
        for (field, stat) in self.fields:
            self.experiment.log_metric(field, self.trainer.stats[field][stat])
        self.experiment.log_epoch_end(epoch_index)


class TeeLogger(Logger):

    def __init__(self, log_file, *args, **kwargs):
        super(TeeLogger, self).__init__(*args, **kwargs)
        self.log_file = open(log_file, 'a', 1)

    def log(self, msg):
        print(msg, flush=True)
        self.log_file.write(msg)

    def epoch(self, epoch_idx):
        self.iteration(epoch_idx)
