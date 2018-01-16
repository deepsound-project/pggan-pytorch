import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PGConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, ksize=3, stride=1, pad=1,
                 pixelnorm=True, wscale='impl', act='lrelu',
                 winit='impl'):
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
                self.c = np.sqrt((ch_in * ksize * ksize) / 2)
            elif wscale == 'impl':
                self.c = np.sqrt(torch.mean(self.conv.weight.data ** 2))
            self.conv.weight.data /= self.c
        else:
            self.c = 1.
        self.eps = 1e-8

        self.pixelnorm = pixelnorm
        if act is not None:
            self.act = nn.LeakyReLU(0.2) if act == 'lrelu' else nn.ReLU()
        else:
            self.act = None
        self.conv.cuda()

    def forward(self, x):
        h = x * self.c
        h = self.conv(h)
        if self.act is not None:
            h = self.act(h)
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
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)
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
        self.toRGB = PGConv2d(ch_out, num_channels, ksize=1, pad=0, pixelnorm=False, act=None)

    def forward(self, x, last=False):
        x = self.c1(x)
        x = self.c2(x)
        if last:
            x = self.toRGB(x)
        return x


class Generator(nn.Module):
    def __init__(self,
        dataset_shape, # Overriden based on the dataset
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 512,
        latent_size         = 512,
        normalize_latents   = True,
        use_wscale          = True,
        use_pixelnorm       = True,
        use_leakyrelu       = True):
        super(Generator, self).__init__()

        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        print(dataset_shape)
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
        ])

        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-8
        self.max_depth = len(self.blocks)

    def forward(self, x):
        h = x.unsqueeze(2).unsqueeze(3)
        if self.normalize_latents:
            mean = torch.mean(h * h, 1, keepdim=True)
            dom = torch.rsqrt(mean + self.eps)
            h = h * dom
        h = self.block0(h, self.depth == 0)
        if self.depth > 0:
            for i in range(self.depth - 1):
                h = F.upsample(h, scale_factor=2)
                h = self.blocks[i](h)
            h = F.upsample(h, scale_factor=2)
            ult = self.blocks[self.depth - 1](h, True)
            if 0.0 < self.alpha < 1.0:
                if self.depth > 1:
                    preult_rgb = self.blocks[self.depth - 2].toRGB(h)
                else:
                    preult_rgb = self.block0.toRGB(h)
            else:
                preult_rgb = 0
            h = preult_rgb * (1-self.alpha) + ult * self.alpha
        return h


class DBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_channels):
        super(DBlock, self).__init__()
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
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
        self.fromRGB = PGConv2d(num_channels, ch_in, ksize=1, pad=0, pixelnorm=False)
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


def Tstdeps(val):
    return torch.sqrt(((val - val.mean())**2).mean() + 1.0e-8)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
        self.eps = 1.0

    def forward(self, x):
        stddev_mean = Tstdeps(x)
        new_channel = stddev_mean.expand(x.size(0), 1, x.size(2), x.size(3))
        h = torch.cat((x, new_channel), dim=1)
        return h


class Discriminator(nn.Module):
    def __init__(self,
        dataset_shape, # Overriden based on dataset
        fmap_base           = 4096,
        fmap_decay          = 1.0,
        fmap_max            = 512,
        use_wscale          = True,
        use_pixelnorm       = True,
        use_leakyrelu       = True):
        super(Discriminator, self).__init__()

        resolution = dataset_shape[-1]
        num_channels = dataset_shape[1]
        R = int(np.log2(resolution))
        assert resolution == 2 ** R and resolution >= 4
        self.R = R

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.blocks = nn.ModuleList([
            DBlock(nf(i), nf(i-1), num_channels)
            for i in range(R-1, 1, -1)
        ] + [DLastBlock(nf(1), nf(0), num_channels)])

        self.linear = nn.Linear(nf(0), 1)
        self.depth = 0
        self.alpha = 1.0
        self.eps = 1e-8
        self.max_depth = len(self.blocks) - 1

    def forward(self, x):
        xhighres = x
        h = self.blocks[-(self.depth + 1)](xhighres, True)
        if self.depth > 0:
            h = F.avg_pool2d(h, 2)
            if self.alpha > 0.0:
                xlowres = F.avg_pool2d(xhighres, 2)
                preult_rgb = self.blocks[-self.depth].fromRGB(xlowres)
                h = h * self.alpha + (1 - self.alpha) * preult_rgb

        for i in range(self.depth, 0, -1):
            h = self.blocks[-i](h)
            if i > 1:
                h = F.avg_pool2d(h, 2)
        h = self.linear(h.squeeze(-1).squeeze(-1))
        return h