import os
import glob
import torch
import numpy as np


def adjust_dynamic_range(data, range_in, range_out):
    if range_in != range_out:
        (min_in, max_in) = range_in
        (min_out, max_out) = range_out
        scale_factor = (max_out - min_out) / (max_in - min_in)
        data = (data - min_in) * scale_factor + min_out
    return data


def random_latents(num_latents, latent_size):
    return torch.from_numpy(np.random.randn(num_latents, latent_size).astype(np.float32))