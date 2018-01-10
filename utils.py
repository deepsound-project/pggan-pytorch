import torch
import numpy as np
import os


def adjust_dynamic_range(data, range_in, range_out):
    if range_in != range_out:
        (min_in, max_in) = range_in
        (min_out, max_out) = range_out
        scale_factor = (max_out - min_out) / (max_in - min_in)
        data = (data - min_in) * scale_factor + min_out
    return data


def random_latents(num_latents, latent_size):
    return torch.from_numpy(np.random.randn(num_latents, latent_size).astype(np.float32))


def create_result_subdir(results_dir, experiment_name, dir_pattern='{new_num:03}-{exp_name}'):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fnames = os.listdir(results_dir)
    max_num = max(map(int, filter(lambda x: all(y.isdigit() for y in x), (x.split('-')[0] for x in fnames))),
                  default=0)
    path = os.path.join(results_dir, dir_pattern.format(new_num=max_num+1, exp_name=experiment_name))
    os.makedirs(
        path,
        exist_ok=False
    )
    return path

