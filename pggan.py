import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import config
import misc
import os
import time
from dataset import load_dataset
from network import Generator, Discriminator
from wgan_gp_loss import wgan_gp_D_loss, wgan_gp_G_loss
from functools import reduce, partial
print = partial(print, flush=True)
torch.manual_seed(1337)


def random_latents(num_latents, latent_size):
    return Variable(torch.from_numpy(np.random.randn(num_latents, latent_size).astype(np.float32)))


def train_gan(
    separate_funcs          = False,
    D_training_repeats      = 1,
    G_learning_rate_max     = 0.0010,
    D_learning_rate_max     = 0.0010,
    G_smoothing             = 0.999,
    adam_beta1              = 0.0,
    adam_beta2              = 0.99,
    adam_epsilon            = 1e-8,
    minibatch_default       = 64,
    minibatch_overrides     = {},
    rampup_kimg             = 40,
    lod_initial_resolution  = 4,
    lod_training_kimg       = 4,
    lod_transition_kimg     = 4,
    total_kimg              = 100,
    drange_net              = (-1,1),
    drange_viz              = (-1,1),
    image_grid_size         = None,
    tick_kimg_default       = 2,
    tick_kimg_overrides     = {32:5, 64:2, 128:1, 256:5, 512:2, 1024:1},
    image_snapshot_ticks    = 2,
    network_snapshot_ticks  = 40,
    image_grid_type         = 'default',
    resume_network_pkl      = None,
    resume_kimg             = 0.0,
    resume_time             = 0.0):

    print('lod_training_kimg: {}'.format(lod_training_kimg))
    # Load dataset and build networks.
    training_set, drange_orig = load_dataset(config.data_dir, config.dataset)
    resolution = training_set.shape[-1]
    num_channels = training_set.shape[1]
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
    if image_grid_type == 'default':
        if image_grid_size is None:
            w, h = (resolution,)*2
            print(w, h)
            image_grid_size = np.clip(1920 // w, 3, 16), np.clip(1080 // h, 2, 16)
            print(image_grid_size)
        example_real_images, snapshot_fake_labels = training_set.get_random_minibatch(np.prod(image_grid_size), labels=True)
        snapshot_fake_latents = random_latents(np.prod(image_grid_size), latent_size)
    else:
        raise ValueError('Invalid image_grid_type', image_grid_type)

    # Misc init.
    resolution_log2 = int(np.round(np.log2(resolution)))
    initial_resolution_log2 = int(np.round(np.log2(lod_initial_resolution)))
    initial_lod = max(resolution_log2 - initial_resolution_log2, 0)

    def gen_fn(x):
        z = G.forward(x.cuda(), 1.0)
        if z.size(-1) < resolution:
            z = F.upsample(z, size=(resolution, resolution))
        z = z.cpu().data.numpy()
        return z

    def first_gen(x):
        tmp = G.depth
        G.depth = resolution_log2 - initial_resolution_log2
        z = gen_fn(x)
        G.depth = tmp
        return z
    # Save example images.
    snapshot_fake_images = first_gen(snapshot_fake_latents)
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    misc.save_image_grid(example_real_images, os.path.join(result_subdir, 'reals.png'), drange=drange_orig, grid_size=image_grid_size)
    misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=drange_viz, grid_size=image_grid_size)

    print(G)
    print('Total nuber of parameters in Generator: ', sum(map(lambda x: reduce(lambda a, b: a*b, x.size()), G.parameters())))
    print(D)
    print('Total nuber of parameters in Discriminator: ', sum(map(lambda x: reduce(lambda a, b: a*b, x.size()), D.parameters())))

    # Setting up learning rate and optimizers
    G_lrate = G_learning_rate_max
    D_lrate = D_learning_rate_max
    opt_g = Adam(G.parameters(), G_lrate, (adam_beta1, adam_beta2), adam_epsilon)
    opt_d = Adam(D.parameters(), D_lrate, (adam_beta1, adam_beta2), adam_epsilon)
    lrate_coef = lambda epoch: misc.rampup(epoch, rampup_kimg)
    lr_scheduler_d = LambdaLR(opt_d, lrate_coef)
    lr_scheduler_g = LambdaLR(opt_g, lrate_coef)

    # Training loop.
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []
    train_start_time = tick_start_time - resume_time
    # def f(cur_nimg, min_lod, max_lod, tick_train_out, tick_start_nimg, cur_tick, tick_start_time):
    while cur_nimg < total_kimg * 1000:

        # Calculate current LOD.
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / 1000.0) / (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) * (lod_training_kimg + lod_transition_kimg) / lod_transition_kimg, 0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)

        # Update learning rate.
        lr_scheduler_d.step(cur_nimg / 1000.0)
        lr_scheduler_g.step(cur_nimg / 1000.0)


        # Get next batch
        real_images_expr, _ = training_set.get_random_minibatch(minibatch_size, lod=cur_lod, shrink_based_on_lod=True,
                                                               labels=True)
        real_images_expr = misc.adjust_dynamic_range(real_images_expr, drange_orig, drange_net)
        fake_latents_in = random_latents(minibatch_size, latent_size).data.cuda()
        real_images_expr = torch.from_numpy(real_images_expr.astype('float32')).cuda()

        # Calculate current depth and fade percentage
        depth = resolution_log2 - int(cur_lod) - initial_resolution_log2
        alpha = float(1 - np.fmod(cur_lod, 1.0))

        # Calculate loss and optimize
        D_loss = D_real = D_fake = 0.
        for i in range(D_training_repeats):
            D_loss, D_real, D_fake = wgan_gp_D_loss(D, G, depth, alpha, real_images_expr, fake_latents_in, return_all=True, **config.loss)
            D_loss.backward()
            opt_d.step()

        G_loss = wgan_gp_G_loss(G, D, depth, alpha, fake_latents_in)
        G_loss.backward()
        opt_g.step()

        tick_train_out.append((G_loss, D_loss, D_real, D_fake))
        cur_nimg += minibatch_size

        # Perform maintenance operations once per tick.
        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            fromcpu = lambda x: x.cpu().data.numpy()
            fcmean = lambda x: fromcpu(x).mean()
            tick_train_out = [(fromcpu(gc), fromcpu(dc), fcmean(dr), fcmean(df))
                               for gc, dc, dr, df in tick_train_out]
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []

            # Print progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-9.1f sec/kimg %-6.1f Dgdrop %-8.4f Gloss %-8.4f Dloss %-8.4f Dreal %-8.4f Dfake %-8.4f' % (
                (cur_tick, cur_nimg / 1000.0, cur_lod, minibatch_size, misc.format_time(cur_time - train_start_time), tick_time, tick_time / tick_kimg, 0.0) + tick_train_avg))

            # Visualize generated images.
            if cur_tick % image_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                snapshot_fake_images = gen_fn(snapshot_fake_latents)
                misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg // 1000)), drange=drange_viz, grid_size=image_grid_size)

            # Save network snapshot every N ticks.
            if cur_tick % network_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                torch.save(G, os.path.join(result_subdir, 'network-snapshot-generator-%06d.dat' % int(cur_nimg // 1000)))
                torch.save(D, os.path.join(result_subdir, 'network-snapshot-discriminator-%06d.dat' % int(cur_nimg // 1000)))

    torch.save(G, os.path.join(result_subdir, 'network-final-generator.dat'))
    torch.save(D, os.path.join(result_subdir, 'network-final-discriminator.dat'))
    training_set.close()
    print('Done.')
    with open(os.path.join(result_subdir, '_training-done.txt'), 'wt'):
        pass

if __name__ == "__main__":
    np.random.seed(config.random_seed)
    func_params = config.train
    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)