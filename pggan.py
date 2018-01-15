from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from network import Generator, Discriminator
from wgan_gp_loss import wgan_gp_G_loss, wgan_gp_D_loss
from functools import reduce, partial
from trainer import Trainer
from dataset import DepthDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from plugins import *
from utils import *
from argparse import ArgumentParser
from collections import OrderedDict
torch.manual_seed(1337)

default_params = OrderedDict(
    data_dir='datasets',
    dataset='specs512.h5',
    result_dir='results',
    exp_name='specs512',
    minibatch_size=16,
    lr_rampup_kimg=40,
    G_lr_max=0.001,
    D_lr_max=0.001,
    total_kimg=3000,
    tick_kimg_default=20,
    image_snapshot_ticks=3,
    network_snapshot_ticks=40,
    resume_network='',
    resume_time=0,
    num_data_workers=16,
    random_seed=1337,
    progressive_growing=True,
    comet_key='',
    comet_project_name='None',
    iwass_lambda=10.0,
    iwass_epsilon=0.001,
    iwass_target=1.0,
)


class InfiniteRandomSampler(RandomSampler):

    def __iter__(self):
        while True:
            it = super().__iter__()
            for x in it:
                yield x


def load_models(resume_network, result_dir, logger):
    logger.log('Resuming {}'.format(resume_network))
    G = torch.load(os.path.join(result_dir, resume_network.format('generator')))
    D = torch.load(os.path.join(result_dir, resume_network.format('discriminator')))
    return G, D


def init_comet(params, trainer):
    if params['comet_key']:
        from comet_ml import Experiment
        experiment = Experiment(api_key=params['comet_key'], project_name=params['comet_project_name'], log_code=False)
        hyperparams = {
            name: str(params[name]) for name in params
        }
        experiment.log_multiple_params(hyperparams)
        trainer.register_plugin(CometPlugin(
            experiment, [
                'G_loss.epoch_mean',
                'D_loss.epoch_mean',
                'D_real.epoch_mean',
                'D_fake.epoch_mean',
                'sec.kimg',
                'sec.tick',
                'kimg_stat'
            ] + (['depth', 'alpha'] if params['progressive_growing'] else [])
        ))


def main(params):      # plugin

    dataset = DepthDataset(os.path.join(params['data_dir'], params['dataset']),
                           **params['DepthDataset'])
    result_dir = create_result_subdir(params['result_dir'], params['exp_name'])
    resolution = dataset.shape[-1]
    num_channels = dataset.shape[1]

    losses = ['G_loss', 'D_loss', 'D_real', 'D_fake']
    stats_to_log = [
                  'tick_stat',
                  'kimg_stat',
    ]
    if params['progressive_growing']:
        stats_to_log.extend([
                  'depth',
                  'alpha',
                  'lod',
                  'minibatch_size'
        ])
    stats_to_log.extend([
        'time',
        'sec.tick',
        'sec.kimg'
    ] + losses)
    logger = TeeLogger(os.path.join(result_dir, 'log.txt'), stats_to_log, [(1, 'epoch')])
    logger.log(params_to_str(params))
    if params['resume_network']:
        G, D = load_models(params['resume_network'], params['result_dir'], logger)
    else:
        G = Generator(num_channels, resolution, **params['Generator'])
        D = Discriminator(num_channels, resolution, **params['Discriminator'])

    G.cuda()
    D.cuda()
    latent_size = params['Generator']['latent_size']

    logger.log(str(G))
    logger.log('Total nuber of parameters in Generator: {}'.format(
        sum(map(lambda x: reduce(lambda a, b: a*b, x.size()), G.parameters()))
    ))
    logger.log(str(D))
    logger.log('Total nuber of parameters in Discriminator: {}'.format(
        sum(map(lambda x: reduce(lambda a, b: a*b, x.size()), D.parameters()))
    ))

    def get_dataloader(minibatch_size):
        return DataLoader(dataset, minibatch_size, sampler=InfiniteRandomSampler(dataset),
                          num_workers=params['num_data_workers'], pin_memory=False, drop_last=True)

    def rl(bs):
        return lambda: random_latents(bs, latent_size)

    # Setting up learning rate and optimizers
    opt_g = Adam(G.parameters(), params['G_lr_max'], **params['Adam'])
    opt_d = Adam(D.parameters(), params['D_lr_max'], **params['Adam'])

    def rampup(cur_nimg):
        if cur_nimg < params['lr_rampup_kimg'] * 1000:
            p = max(0.0, 1 - cur_nimg / (params['lr_rampup_kimg'] * 1000))
            return np.exp(-p * p * 5.0)
        else:
            return 1.0
    lr_scheduler_d = LambdaLR(opt_d, rampup)
    lr_scheduler_g = LambdaLR(opt_g, rampup)

    mb_def = params['minibatch_size']
    D_loss_fun = partial(wgan_gp_D_loss, return_all=True, iwass_lambda=params['iwass_lambda'],
                         iwass_epsilon=params['iwass_epsilon'], iwass_target=params['iwass_target'])
    G_loss_fun = wgan_gp_G_loss
    trainer = Trainer(D, G, D_loss_fun, G_loss_fun,
                      opt_d, opt_g, dataset, iter(get_dataloader(mb_def)), rl(mb_def), **params['Trainer'])
    # plugins
    if params['progressive_growing']:
        trainer.register_plugin(DepthManager(get_dataloader, rl, **params['DepthManager']))
    for i, loss_name in enumerate(losses):
        trainer.register_plugin(EfficientLossMonitor(i, loss_name))
    trainer.register_plugin(SaverPlugin(result_dir, True, params['network_snapshot_ticks']))
    trainer.register_plugin(SampleGenerator(result_dir, lambda x: random_latents(x, latent_size), **params['SampleGenerator']))
    trainer.register_plugin(AbsoluteTimeMonitor(params['resume_time']))
    trainer.register_plugin(LRScheduler(lr_scheduler_d, lr_scheduler_g))
    trainer.register_plugin(logger)
    init_comet(params, trainer)
    trainer.run(params['total_kimg'])
    dataset.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    needarg_classes = [Trainer, Generator, Discriminator, DepthDataset, DepthManager, SaverPlugin, SampleGenerator, Adam]
    excludes = {'Adam': {'lr'}}
    default_overrides = {'Adam': {'betas': (0.0, 0.99)}}
    auto_args = create_params(needarg_classes, excludes, default_overrides)
    # default_params.update(auto_args)
    for k in default_params:
        parser.add_argument('--{}'.format(k), type=partial(generic_arg_parse, hinttype=type(default_params[k])))
    for cls in auto_args:
        for k in auto_args[cls]:
            name = '{}.{}'.format(cls, k)
            parser.add_argument('--{}'.format(name), type=generic_arg_parse)
            default_params[name] = auto_args[cls][k]
    parser.set_defaults(**default_params)
    params = get_structured_params(vars(parser.parse_args()))
    main(params)
