import torch
import numpy as np
import os
import inspect
from pickle import load, dump


def generate_samples(generator, gen_input):
    out = generator.forward(gen_input)
    out = out.cpu().data.numpy()
    return out


def save_pkl(fname, obj):
    with open(fname, 'wb') as f:
        dump(obj, f)


def load_pkl(fname):
    with open(fname, 'rb') as f:
        return load(f)


def adjust_dynamic_range(data, range_in, range_out):
    if range_in != range_out:
        (min_in, max_in) = range_in
        (min_out, max_out) = range_out
        scale_factor = (max_out - min_out) / (max_in - min_in)
        data = (data - min_in) * scale_factor + min_out
    return data


def numpy_upsample_nearest(x, n_last_dims, size=None, scale_factor=None):
    try:
        shape = x.shape[-n_last_dims:]
        if size is not None:
            if type(size) is int:
                size = (size,) * n_last_dims
            for i in range(n_last_dims):
                if size[i] % shape[i] != 0:
                    raise Exception('Incompatible sizes: {} and {}.'.format(x.shape, size))
            scale_factor = tuple((target_s // source_s for source_s, target_s in zip(shape, size)))
        if scale_factor is None:
            raise Exception('Either size or scale_factor must be specified.')
        if type(scale_factor) is int:
            scale_factor = (scale_factor,) * n_last_dims
        for i in range(n_last_dims):
            if scale_factor[i] > 1:
                x = x.repeat(scale_factor[i], axis=-n_last_dims + i)
        return x
    except Exception as e:
        print('Args or shapes in numpy_upsample: x {} size {} scale_factor {}'.format(x.shape, size, scale_factor))
        raise e


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


def get_all_classes(module):
    return [getattr(module, name) for name in dir(module)
            if inspect.isclass(getattr(module, name, 0))]


def generic_arg_parse(x, hinttype=None):
    if hinttype is int or hinttype is float or hinttype is str:
        return hinttype(x)
    try:
        for _ in range(2):
            x = x.strip('\'').strip("\"")
        __special_tmp = eval(x, {}, {})
    except:  # the string contained some name - probably path, treat as string
        __special_tmp = x  # treat as string
        print('Treating value: {} as str.'.format(x))
    return __special_tmp


def create_params(classes, excludes=None, overrides=None):
    params = {}
    if not excludes:
        excludes = {}
    if not overrides:
        overrides = {}
    for cls in classes:
        nm = cls.__name__
        params[nm] = {
            k: (v.default if nm not in overrides or k not in overrides[nm] else overrides[nm][k])
            for k, v in dict(inspect.signature(cls.__init__).parameters).items()
            if v.default != inspect._empty and
            (nm not in excludes or k not in excludes[nm])
        }
    return params


def get_structured_params(params):
    new_params = {}
    for p in params:
        if '.' in p:
            [cls, attr] = p.split('.', 1)
            if cls not in new_params:
                new_params[cls] = {}
            new_params[cls][attr] = params[p]
        else:
            new_params[p] = params[p]
    return new_params


def params_to_str(params):
    s = '{\n'
    for k, v in params.items():
        s += '\t\'{}\': {},\n'.format(k, repr(v))
    s += '}'
    return s