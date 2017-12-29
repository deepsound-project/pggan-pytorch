import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import config
import misc
import os
import time
import dataset
torch.manual_seed(1337)


class PGConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1,
                 pixelnorm=True, wscale='paper', act=F.leaky_relu,
                 winit='paper'):
        super(PGConv2d, self).__init__()

        if winit == 'paper':
            init = lambda x: nn.init.normal(x, 0, 1)
        elif winit == 'impl':
            init = lambda x: nn.init.kaiming_normal(x)
        else:
            init = lambda x: x
        self.conv = nn.Conv2d(ch_in, ch_out, ksize, stride, pad)
        init(self.conv.weight)
        if wscale:
            if wscale == 'paper':
                self.c = np.sqrt(2 / (ch_in * ksize * ksize))
            elif wscale == 'impl':
                self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2))
            self.conv.weight.data /= self.c
        else:
            self.c = 1.
        self.eps = 1e-8

        self.pixelnorm = pixelnorm
        self.act = act
        self.conv.cuda()

    def forward(self, x):
        h = x * self.c
        h = self.act(self.conv(h))
        if self.pixelnorm:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        return h


class GFirstBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels):
        super(GFirstBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out, 4, 1, 3)
        self.c2 = PGConv2d(ch_out, ch_out)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False,
                         winit=None, wscale=None, act=lambda x:x)
        # print('no elo', num_channels)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            return self.toRGB(x)
        return x


class GBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels):
        super(GBlock, self).__init__()
        self.c1 = PGConv2d(ch_in, ch_out)
        self.c2 = PGConv2d(ch_out, ch_out)
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False,
                         winit=None, wscale=None, act=lambda x:x)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            x = self.toRGB(x)
        return x


class Generator(nn.Module):
    def __init__(self,
        num_channels        = 1,        # Overridden based on dataset.
        resolution          = 32,       # Overridden based on dataset.
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 256,
        latent_size         = None,
        normalize_latents   = True,
        use_wscale          = True,
        use_pixelnorm       = True,
        use_leakyrelu       = True):
        super(Generator, self).__init__()

        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        if latent_size is None:
            latent_size = nf(0)

        self.normalize_latents = normalize_latents

        # print('no siemix', num_channels)
        self.block0 = GFirstBlock(latent_size, nf(1), num_channels)
        self.blocks = nn.ModuleList([
            GBlock(nf(i-1), nf(i), num_channels)
            for i in range(2, R)
            # GBlock(512, 256, (16, 16)),
            # GBlock(256, 128, (32, 32)),
            # GBlock(128, 64, (64, 64)),
            # GBlock(64, 32, (128, 128)),
            # GBlock(32, 16, (256, 256)),
        ])

        self.depth = 0
        self.eps = 1e-8

    def forward(self, x, alpha=0.0):
        h = x.unsqueeze(2).unsqueeze(3)
        # print('raz', h.size())
        if self.normalize_latents:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        # print('dwa', h.size())
        h = self.block0(h, self.depth == 0)
        # print('tri', h.size())
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = F.upsample(h, scale_factor=2)
                h = self.blocks[i](h)
            h = F.upsample(h, scale_factor=2)
            ult = self.blocks[self.depth - 1](h, True)
            # print('trololol', ult.size(), h.size(), self.blocks[self.depth - 2])
            if alpha > 0.0 and alpha < 1.0:
                if self.depth > 1:
                    preult_rgb = self.blocks[self.depth - 2].toRGB(h)
                else:
                    preult_rgb = self.block0.toRGB(h)
                # print('preult_rgb {}, ult {}'.format(preult_rgb.size(), ult.size()))
            else:
                preult_rgb = 0
            h = preult_rgb * (1-alpha) + ult * alpha
        # print('Gen final shape for depth {} and alpha {}: {}'.format(self.depth, alpha, h.size()))
        return h


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels):
        super(DBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False,
                              winit=None, wscale=None, act=lambda x: x)
        self.c1 = PGConv2d(ch_in, ch_in, pixelnorm=False)
        self.c2 = PGConv2d(ch_in, ch_out, pixelnorm=False)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class DLastBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels):
        super(DLastBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False,
                              winit=None, wscale=None, act=lambda x: x)
        self.stddev = MinibatchStddev()
        self.c1 = PGConv2d(ch_in + 1, ch_in, pixelnorm=False)
        self.c2 = PGConv2d(ch_in, ch_out, 4, 1, 0, pixelnorm=False)

    def forward(self, x, first=False):
        if first:
            x = self.fromRGB(x)
        x = self.stddev(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


def allmean(t):
    from functools import reduce
    return reduce(lambda acc, x: torch.mean(acc, dim=x, keepdim=True),
                  list(range(t.dim())), t)


def Tstdeps(val):
    return torch.sqrt(allmean((val - allmean(val))**2) + 1.0e-8)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.eps = 1.0

    def forward(self, x):
        stddev_mean = Tstdeps(x)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2), x.size(3))
        # print(new_channel.size(), x.size())
        h = torch.cat((x, new_channel), dim=1)
        return h


class Discriminator(nn.Module):
    def __init__(self,
        num_channels        = 1,        # Overridden based on dataset.
        resolution          = 32,       # Overridden based on dataset.
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 256,
        use_wscale          = True,
        use_pixelnorm       = True,
        use_leakyrelu       = True):
        super(Discriminator, self).__init__()

        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        self.R = R

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.blocks = nn.ModuleList([
            DBlock(nf(i), nf(i-1), num_channels)
            for i in range(R-1, 1, -1)
            # GBlock(512, 256, (16, 16)),
            # GBlock(256, 128, (32, 32)),
            # GBlock(128, 64, (64, 64)),
            # GBlock(64, 32, (128, 128)),
            # GBlock(32, 16, (256, 256)),
        ] + [DLastBlock(nf(1), nf(0), num_channels)])

        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.eps = 1e-8

    def forward(self, x, alpha=0.0):
        blockno = self.R - self.depth - 2
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        # if self.depth > 0: print('entered with highres: {} and out of first: {}'.format(xhighres.size(), h.size()))
        if self.depth > 0:
            h = F.avg_pool2d(h, 2)
            # print('h, ', h.size())
            if alpha > 0.0:
                # print('alpha > 0')
                xlowres = F.avg_pool2d(xhighres, 2)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                # print('sizes: h {}, preult {}'.format(h.size(), preult_rgb.size()))
                # print(type(alpha))
                melted = h * alpha
                # print(type(melted))
                # print('preult_rgb, ', preult_rgb.size())
                h = h * alpha + (1 - alpha) * preult_rgb
                # print('dunn')

        for i in range(self.depth, 0, -1):
            # print('inserting prev to next', i, self.blocks[-1])
            # print(h.size())
            h = self.blocks[-i](h)
            # print(h.size())
            if i > 1:
                h = F.avg_pool2d(h, 2)
        # if self.depth > 0: print('linear', self.linear, h.size())
        h = self.linear(h.squeeze(-1).squeeze(-1))
        # if self.depth > 0: print('go!')
        return h


def load_dataset(dataset_spec=None, verbose=False, **spec_overrides):
    if verbose: print('Loading dataset...')
    if dataset_spec is None: dataset_spec = config.dataset
    dataset_spec = dict(dataset_spec) # take a copy of the dict before modifying it
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = os.path.join(config.data_dir, dataset_spec['h5_path'])
    if 'label_path' in dataset_spec: dataset_spec['label_path'] = os.path.join(config.data_dir, dataset_spec['label_path'])
    training_set = dataset.Dataset(**dataset_spec)
    if verbose: print('Dataset shape =', np.int32(training_set.shape).tolist())
    drange_orig = training_set.get_dynamic_range()
    if verbose: print('Dynamic range =', drange_orig)
    return training_set, drange_orig


def load_dataset_for_previous_run(result_subdir, **kwargs):
    dataset = None
    with open(os.path.join(result_subdir, 'config.txt'), 'rt') as f:
        for line in f:
            if line.startswith('dataset = '):
                exec(line)
    return load_dataset(dataset, **kwargs)


def random_latents(num_latents, latent_size):
    return Variable(torch.from_numpy(np.random.randn(num_latents, latent_size).astype(np.float32)))


def calc_gradient_penalty(D, alpha, real_data, fake_data, iwass_lambda, iwass_target):
    mixing_factors = torch.cat([torch.from_numpy(np.random.uniform(size=(1,1)).astype('float32')).expand(1, *real_data.size()[1:]) for _ in range((real_data.size(0)))]).cuda()
    # print('depth: sizes in loss: {} {} {}'.format(D.depth, real_data.size(), fake_data.size(), mixing_factors.size()))
    mixed_data = Variable(real_data * (1 - mixing_factors) + fake_data * mixing_factors, requires_grad=True)
    # print('dupa', mixed_data.size())
    mixed_scores = D(mixed_data, alpha)

    gradients = autograd.grad(outputs=mixed_scores, inputs=mixed_data,
                              grad_outputs=torch.ones(mixed_scores.size()).cuda(),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - iwass_target) ** 2) * iwass_lambda / (iwass_target ** 2)

    return gradient_penalty


def evaluate_loss(
    G, D, depth, alpha, real_images_in,
    fake_latents_in, opt_g, opt_d,
    type            = 'iwass',
    L2_fake_weight  = 0.1,
    iwass_lambda    = 10.0,
    iwass_epsilon   = 0.001,
    iwass_target    = 1.0): # set cond_tweak_G=0.1 to match original improved Wasserstein implementation

    # Helpers.
    def L2(a, b): return 0 if a is None or b is None else torch.mean((a - b)**2)

    D.zero_grad()
    G.zero_grad()

    G.depth = depth
    D.depth = depth

    real_data_v = Variable(real_images_in)
    # train with real
    D_real = D(real_data_v, alpha)
    # D_real = D_real.mean()
    D_real_loss = -D_real + D_real**2 * iwass_epsilon
    # D_real_loss.backward()

    # train with fake
    noisev = Variable(fake_latents_in, volatile=True)  # totally freeze netG
    # print(noisev.size())
    fake = Variable(G(noisev, alpha).data)
    inputv = fake
    D_fake = D(inputv, alpha)
    D_fake_loss = D_fake
    # D_fake_loss.backward()
    # D_fake = D_fake.mean()
    # D_fake.backward(one)

    # l2v = Variable(D_real.data)
    # D_l2 = L2(l2v, 0) * iwass_epsilon
    # D_l2_loss = -D_l2.mean()
    # D_l2_loss.backward()
    # D_l2.backward(mone)

    # train with gradient penalty
    gradient_penalty = calc_gradient_penalty(D, alpha, real_data_v.data, fake.data, iwass_lambda, iwass_target)
    gp = gradient_penalty
    # gp.backward()

    D_cost = (D_fake_loss + D_real_loss + gp).mean()
    D_cost.backward()
    Wasserstein_D = D_real - D_fake
    opt_d.step()

    ############################
    # (2) Update G network
    ###########################
    D.zero_grad()
    G.zero_grad()

    noisev = Variable(fake_latents_in)
    # fake = G(noisev, alpha)
    # G_res = D(fake)
    # G_res = G_res.mean()
    G_new = G(noisev, alpha)
    # D_new = G_new
    # G_cost = D_new
    D_new = -D(G_new, alpha)
    G_cost = D_new.mean()
    G_cost.backward()

    opt_g.step()

    # print('...', end=' ')
    # G.depth = depth
    # fake_images_out = G(fake_latents_in, alpha)
    # # Mix reals and fakes through linear crossfade.
    # mixing_factors = Variable(torch.from_numpy(np.random.uniform(size=(real_images_in.size(0), 1, 1, 1)).astype(np.float32)))
    # print('depth: sizes in loss: {} {} {}'.format(depth, real_images_in.size(), fake_images_out.size(), mixing_factors.size()))
    # mixed_images_out = real_images_in * (1 - mixing_factors) + fake_images_out * mixing_factors
    #
    # # Evaluate discriminator.
    # D.depth = depth
    # real_scores_out = D(real_images_in,   alpha)
    # fake_scores_out = D(fake_images_out,  alpha)
    # mixed_scores_out = D(mixed_images_out, alpha)

    # if type == 'iwass': # Improved Wasserstein
    #     G_loss = -(fake_scores_out.mean())
    #     gp = calc_gradient_penalty(D, mixed_images_out, mixed_scores_out, iwass_lambda, iwass_target)
    #     gp.backward()
    #     D_loss = ((fake_scores_out).mean() - (real_scores_out).mean()) + gp
    #     D_loss += L2(real_scores_out, 0) * iwass_epsilon # additional penalty term to keep the scores from drifting too far from zero
    #     fake_scores_out = fake_scores_out - real_scores_out # reporting tweak
    #     real_scores_out = 0 # reporting tweak

    # if type == 'lsgan': # LSGAN
    #     G_loss = L2(fake_scores_out, 0)
    #     D_loss = L2(real_scores_out, 0) + L2(fake_scores_out, 1) * L2_fake_weight
    #
    # # update disc
    # opt_d.zero_grad()
    # D_loss.backward()
    # opt_d.step()
    #
    # # update generator
    # opt_g.zero_grad()
    # G_loss.backward()
    # opt_g.step()

    return G_cost, D_cost, D_real, D_fake


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
    rampdown_kimg           = 0,
    lod_initial_resolution  = 32,
    lod_training_kimg       = 4,
    lod_transition_kimg     = 4,
    total_kimg              = 100,
    drange_net              = [-1,1],
    drange_viz              = [-1,1],
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
    training_set, drange_orig = load_dataset()
    resolution = training_set.shape[-1]
    num_channels = training_set.shape[1]
    if resume_network_pkl:
        print('Resuming', resume_network_pkl)
        G = torch.load(os.path.join(config.result_dir, resume_network_pkl.format('generator')))
        D = torch.load(os.path.join(config.result_dir, resume_network_pkl.format('discriminator')))
    else:
        G = Generator(num_channels, resolution, **config.G)
        D = Discriminator(num_channels, resolution, **config.D)
    print(G)
    print(D)
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

    # Theano input variables and compile generation func.
    print('Setting up Theano... not!')
    G_lrate = 0.001
    D_lrate = 0.001
    def gen_fn(x):
        # tmp = G.depth
        # G.depth = resolution_log2 - 2
        z = G.forward(x.cuda(), 1.0)
        print(z.size(), resolution)
        if z.size(-1) < resolution:
            z = F.upsample(z, size=(resolution, resolution))
        z = z.cpu().data.numpy()
        print(z.shape)
        # G.depth = tmp
        # print(z)
        return z
    def first_gen(x):
        tmp = G.depth
        G.depth = resolution_log2 - 2
        z = gen_fn(x)
        G.depth = tmp
        return z
    from torch.optim import Adam
    print(sum(map(lambda x: sum(x.size()), G.parameters())))
    print(sum(map(lambda x: sum(x.size()), D.parameters())))
    opt_g = Adam(G.parameters(), G_lrate, (adam_beta1, adam_beta2), adam_epsilon)
    opt_d = Adam(D.parameters(), D_lrate, (adam_beta1, adam_beta2), adam_epsilon)

    # Misc init.
    resolution_log2 = int(np.round(np.log2(resolution)))
    initial_lod = max(resolution_log2 - int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

    print ('aaa', resolution_log2, initial_lod)

    # Save example images.
    snapshot_fake_images = first_gen(snapshot_fake_latents)
    result_subdir = misc.create_result_subdir(config.result_dir, config.run_desc)
    misc.save_image_grid(example_real_images, os.path.join(result_subdir, 'reals.png'), drange=drange_orig, grid_size=image_grid_size)
    misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % 0), drange=drange_viz, grid_size=image_grid_size)

    print ('batches',minibatch_default,minibatch_overrides)
    # Training loop.
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []
    train_start_time = tick_start_time - resume_time
    lrate_coef = lambda epoch: misc.rampup(epoch, rampup_kimg)
    lr_scheduler_d = LambdaLR(opt_d, lrate_coef)
    lr_scheduler_g = LambdaLR(opt_g, lrate_coef)
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

        # print ('cur_lod', cur_lod)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)

        # print ('cur_res', cur_res)

        # Update network config.
        lr_scheduler_d.step(cur_nimg / 1000.0)
        lr_scheduler_g.step(cur_nimg / 1000.0)

        # lrate_coef = misc.rampup(cur_nimg / 1000.0, rampup_kimg)
        # lrate_coef *= misc.rampdown_linear(cur_nimg / 1000.0, total_kimg, rampdown_kimg)
        # G_lrate = np.float32(lrate_coef * G_learning_rate_max)
        # D_lrate = np.float32(lrate_coef * D_learning_rate_max)
        # G.depth = int(lod_initial_resolution - cur_lod)

        # Setup training func for current LOD.
        new_min_lod, new_max_lod = int(np.floor(cur_lod)), int(np.ceil(cur_lod))
        real_images_expr, _ = training_set.get_random_minibatch(minibatch_size, lod=cur_lod, shrink_based_on_lod=True,
                                                               labels=True)
        real_images_expr = misc.adjust_dynamic_range(real_images_expr, drange_orig, drange_net)
        # print ('lody', new_min_lod, new_max_lod, min_lod, max_lod)
        if min_lod != new_min_lod or max_lod != new_max_lod:
            print('Compiling training funcs... not! Its not Theano anymore!\nNotice, though, that depth and alpha are changing.')
            min_lod, max_lod = new_min_lod, new_max_lod

            # # Pre-process reals.
            # print('adjusting range...')
            #
            # print('done.')
            # if min_lod > 0: # compensate for shrink_based_on_lod
            #     print(real_images_expr)
            #     real_images_expr = real_images_expr.repeat(2**min_lod, 2)
            #     real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=3)

            # Optimize loss.

        depth = resolution_log2 - int(cur_lod) - 2
        alpha = float(1 - cur_lod % 1)
        fake_latents_in = random_latents(minibatch_size, latent_size).data.cuda()
        real_images_expr = torch.from_numpy(real_images_expr.astype('float32')).cuda()
        # print('depth {}, alpha {}, real_images_expr.shape {}'.format(depth, alpha, real_images_expr.size()))
        G_loss, D_loss, real_scores_out, fake_scores_out = evaluate_loss(G, D, depth, alpha, real_images_expr, fake_latents_in, opt_g, opt_d, **config.loss)

        tick_train_out.append((G_loss, D_loss, real_scores_out, fake_scores_out))
        cur_nimg += minibatch_size
        # Perform maintenance operations once per tick.
        # print ('cur_nimg', cur_nimg, 'tick_start_nimg', tick_start_nimg, 'tick_duration_kimg', tick_duration_kimg)
        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            fromcpu = lambda x: x.cpu().data.numpy()
            fcmean = lambda x: fromcpu(x).mean()
            tick_train_out = [ (fromcpu(gc), fromcpu(dc), fcmean(dr), fcmean(df))
                               for gc, dc, dr, df in tick_train_out ]
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []
            # print(tick_train_avg)
            # print(tick_train_out)
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
                torch.save(D, os.path.join(result_subdir, 'network-snapshot-discriimnator-%06d.dat' % int(cur_nimg // 1000)))
            # break
    # import cProfile
    # cProfile.runctx('f(cur_nimg, min_lod, max_lod, tick_train_out, tick_start_nimg, cur_tick, tick_start_time)', globals(), locals())
    # Write final results.
    # torch.save(G, os.path.join(result_subdir, 'network-final-generator.dat'))
    # torch.save(D, os.path.join(result_subdir, 'network-final-discriimnator.dat'))
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