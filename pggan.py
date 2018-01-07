from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import config
from network import Generator, Discriminator
from functools import reduce, partial
from trainer import Trainer
from dataset import DepthDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from plugins import *
from utils import random_latents
torch.manual_seed(1337)


class InfiniteRandomSampler(RandomSampler):

    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


def train_gan(
    separate_funcs          = False,    # stays this way
    D_training_repeats      = 1,        # trainer
    G_learning_rate_max     = 0.0010,   # trainer
    D_learning_rate_max     = 0.0010,   # trainer
    G_smoothing             = 0.999,    # ???
    adam_beta1              = 0.0,      # trainer (init)
    adam_beta2              = 0.99,     # trainer (init)
    adam_epsilon            = 1e-8,     # trainer (init)
    minibatch_default       = 64,       # \
    minibatch_overrides     = {},       # /-> plugin to change DataLoader / Dataset in Trainer
    rampup_kimg             = 40,       # every batch - lr_scheduler.step() (plugin or directly)
    dataset_depth_offset    = 4,
    lod_training_kimg       = 4,        # trainer + plugin to change dataloader (/ network?)
    lod_transition_kimg     = 4,        # trainer + plugin to change dataloader (/ network?)
    total_kimg              = 100,      # trainer
    drange_net              = (-1,1),   # dataset
    image_grid_size         = None,     # plugin to create images
    tick_kimg_default       = 20,        # trainer
    tick_kimg_overrides     = {3:10, 4:10, 5:5, 6:2, 7:2, 8:1}, # trainer
    image_snapshot_ticks    = 2,        # plugin based on ticks for img snapshot
    network_snapshot_ticks  = 40,       # plugin based on ticks for network snapshot
    resume_network_pkl      = None,     # trainer ?
    resume_kimg             = 0,      # trainer
    resume_time             = 0,
    num_data_workers        = 16):     # plugin

    print('lod_training_kimg: {}'.format(lod_training_kimg))
    dataset = DepthDataset(os.path.join(config.data_dir, config.dataset), depth_offset=dataset_depth_offset, range_out=drange_net)
    resolution = dataset.shape[-1]
    num_channels = dataset.shape[1]
    if resume_network_pkl:
        print('Resuming', resume_network_pkl)
        G = torch.load(os.path.join(config.result_dir, resume_network_pkl.format('generator')))
        D = torch.load(os.path.join(config.result_dir, resume_network_pkl.format('discriminator')))
    else:
        G = Generator(num_channels, resolution, **config.G)
        D = Discriminator(num_channels, resolution, **config.D)

    G.cuda()
    D.cuda()
    latent_size = config.G['latent_size']

    # Setup snapshot image grid.
    if image_grid_size is None:
        w, h = (resolution,)*2
        image_grid_size = np.clip(1920 // w, 3, 16), np.clip(1080 // h, 2, 16)
        print(image_grid_size)
    # example_real_images = np.vstack(dataset[i] for i in range(np.prod(image_grid_size)))


    print(G)
    print('Total nuber of parameters in Generator: ', sum(map(lambda x: reduce(lambda a, b: a*b, x.size()), G.parameters())))
    print(D)
    print('Total nuber of parameters in Discriminator: ', sum(map(lambda x: reduce(lambda a, b: a*b, x.size()), D.parameters())))

    def get_dataloader(minibatch_size):
        return DataLoader(dataset, minibatch_size, sampler=InfiniteRandomSampler(dataset),
                          num_workers=num_data_workers, pin_memory=False, drop_last=True)

    def rl(bs):
        return lambda: random_latents(bs, latent_size)

    # Setting up learning rate and optimizers
    G_lrate = G_learning_rate_max
    D_lrate = D_learning_rate_max
    opt_g = Adam(G.parameters(), G_lrate, (adam_beta1, adam_beta2), adam_epsilon)
    opt_d = Adam(D.parameters(), D_lrate, (adam_beta1, adam_beta2), adam_epsilon)

    def rampup(epoch):
        if epoch < rampup_kimg * 1000:
            p = max(0.0, 1 - epoch / (rampup_kimg * 1000))
            return np.exp(-p * p * 5.0)
        else:
            return 1.0
    lr_scheduler_d = LambdaLR(opt_d, rampup)
    lr_scheduler_g = LambdaLR(opt_g, rampup)

    trainer = Trainer(D, G, opt_d, opt_g, dataset, iter(get_dataloader(minibatch_default)), rl(minibatch_default), D_training_repeats,
                      tick_nimg_default=tick_kimg_default * 1000, resume_nimg=resume_kimg*1000, resume_time=resume_time)
    # plugins
    trainer.register_plugin(DepthManager(get_dataloader, rl, minibatch_default, minibatch_overrides, tick_kimg_default,
                                         tick_kimg_overrides, lod_training_kimg*1000, lod_transition_kimg*1000))
    losses = ['G_loss', 'D_loss', 'D_real', 'D_fake']
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(EfficientLossMonitor(i, loss_name))
    trainer.register_plugin(SaverPlugin(config.result_dir, True, network_snapshot_ticks))
    trainer.register_plugin(SampleGenerator(config.result_dir, image_grid_size, drange_net,
                                            image_snapshot_ticks, resolution, lambda x: random_latents(x, latent_size)))
    trainer.register_plugin(AbsoluteTimeMonitor(resume_time))
    trainer.register_plugin(LRScheduler(lr_scheduler_d, lr_scheduler_g))
    trainer.register_plugin(TeeLogger(os.path.join(config.result_dir, 'log.txt'),
                                      [
                                          'tick_stat',
                                          'kimg_stat',
                                          'depth',
                                          'alpha',
                                          'lod',
                                          'minibatch_size',
                                          'time',
                                          'sec.tick',
                                          'sec.kimg'
                                      ]
                                      + losses,
                                      [(1, 'epoch')]))
    trainer.run(total_kimg)
    dataset.close()
    print('Done.')
    with open(os.path.join(config.result_dir, '_training-done.txt'), 'wt'):
        pass

if __name__ == "__main__":
    np.random.seed(config.random_seed)
    func_params = config.train
    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)