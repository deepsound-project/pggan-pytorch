import torch
import output_postprocess
from torch.autograd import Variable
from utils import *
from argparse import ArgumentParser
from functools import partial
from output_postprocess import *


default_params = {
    'generator_path': '',
    'num_samples': 6,
    'postprocessors': [],
    'description': 'unknown',
}


def output_samples(generator_path, num_samples, postprocessors, description):
    G = torch.load(generator_path)
    G.cuda()
    latent_size = getattr(G, 'latent_size', 512)  # yup I just want to use old checkpoints
    print('Sampling noise...')
    gen_input = Variable(random_latents(num_samples, latent_size)).cuda()
    print('Generating...')
    output = generate_samples(G, gen_input)
    print('Done.')
    for proc in postprocessors:
        print('Outputting for postprocessor: {}'.format(proc))
        proc(output, description)
    print('Done.')


if __name__ == '__main__':
    parser = ArgumentParser()
    needarg_classes = get_all_classes(output_postprocess)
    auto_args = create_params(needarg_classes)
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
    postprocessors = [ globals()[x](**params[x]) for x in params['postprocessors'] ]
    output_samples(params['generator_path'], params['num_samples'], postprocessors, params['description'])