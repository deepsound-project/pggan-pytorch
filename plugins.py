import os
import time
from datetime import timedelta
from glob import glob

import PIL.Image
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.trainer.plugins import LossMonitor, Logger
from torch.utils.trainer.plugins.plugin import Plugin

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
                 lod_transition_nimg=100*1000,
                 max_lod=9,
                 depth_offset=2):
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
        self.max_lod = max_lod
        self.depth_offset = depth_offset

    def register(self, trainer):
        self.trainer = trainer
        self.trainer.stats['minibatch_size'] = self.minibatch_default
        self.trainer.stats['alpha'] = {'log_name': 'alpha', 'log_epoch_fields': ['{val:.2f}'], 'val': self.alpha}
        if self.max_lod is not None and self.depth_offset is not None:
            self.trainer.stats['lod'] = {'log_name': 'lod', 'log_epoch_fields': ['{val:.2f}'], 'val': self.lod}
        self.iteration()

    @property
    def lod(self):
        if self.max_lod is not None and self.depth_offset is not None:
            return self.max_lod - self.depth_offset - self.depth - self.alpha + 1
        return -1

    def iteration(self, *args):
        cur_nimg = self.trainer.cur_nimg
        full_passes, remaining_nimg = divmod(cur_nimg, self.lod_training_nimg + self.lod_transition_nimg)
        train_passes_rem, remaining_nimg = divmod(remaining_nimg, self.lod_training_nimg)
        depth = full_passes + train_passes_rem
        alpha = remaining_nimg / self.lod_transition_nimg if train_passes_rem > 0 else 1.0
        dataset = self.trainer.dataset
        if depth != self.depth:
            self.trainer.D.depth = self.trainer.G.depth = dataset.depth = depth
            self.depth = depth
            minibatch_size = self.minibatch_overrides.get(depth, self.minibatch_default)
            self.trainer.dataiter = iter(self.create_dataloader_fun(minibatch_size))
            self.trainer.random_latents_generator = self.create_rlg(minibatch_size)
            # print(self.trainer.random_latents_generator().size())
            tick_duration_kimg = self.tick_kimg_overrides.get(depth, self.tick_kimg_default)
            self.trainer.tick_duration_nimg = tick_duration_kimg * 1000
            self.trainer.stats['minibatch_size'] = minibatch_size
        if alpha != self.alpha:
            self.trainer.D.alpha = self.trainer.G.alpha = dataset.alpha = alpha
            self.alpha = alpha
        self.trainer.stats['depth'] = depth
        self.trainer.stats['alpha']['val'] = alpha
        if self.max_lod is not None and self.depth_offset is not None:
            self.trainer.stats['lod']['val'] = self.lod


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
        self.lrs_d.step(self.trainer.cur_nimg)
        self.lrs_g.step(self.trainer.cur_nimg)


class EfficientLossMonitor(LossMonitor):

    def __init__(self, loss_no, stat_name):
        super(EfficientLossMonitor, self).__init__()
        self.loss_no = loss_no
        self.stat_name = stat_name

    def _get_value(self, iteration, *args):
        val = args[self.loss_no] if self.loss_no < 2 else args[self.loss_no].mean()
        return val.data[0]


class AbsoluteTimeMonitor(Plugin):

    stat_name = 'time'

    def __init__(self, base_time=0):
        super(AbsoluteTimeMonitor, self).__init__([(1, 'epoch')])
        self.base_time = base_time
        self.start_time = time.time()
        self.epoch_start = self.start_time
        self.start_nimg = None
        self.epoch_time = 0

    def register(self, trainer):
        self.trainer = trainer
        self.start_nimg = trainer.cur_nimg
        self.trainer.stats['sec'] = {'log_format': ':.1f'}

    def epoch(self, epoch_index):
        cur_time = time.time()
        tick_time = cur_time - self.epoch_start
        self.epoch_start = cur_time
        kimg_time = tick_time / (self.trainer.cur_nimg - self.start_nimg) * 1000
        self.start_nimg = self.trainer.cur_nimg
        self.trainer.stats['time'] = timedelta(seconds=time.time() - self.start_time + self.base_time)
        self.trainer.stats['sec']['tick'] = tick_time
        self.trainer.stats['sec']['kimg'] = kimg_time


class SaverPlugin(Plugin):

    last_pattern = 'network-snapshot-{}-{}.dat'

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
                    self.last_pattern.format(name,
                        '{:06}'.format(self.trainer.cur_nimg // 1000))
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
        num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
        grid_size = self.image_grid_size

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
        self.log_file.write(msg + '\n')

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields')
