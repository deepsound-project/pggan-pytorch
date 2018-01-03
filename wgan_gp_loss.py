import torch
import numpy as np
from torch.autograd import Variable, grad

mixing_factors = None
grad_outputs = None


def mul_rowwise(a, b):
    s = a.size()
    return (a.view(s[0], -1) * b).view(s)


def calc_gradient_penalty(D, depth, alpha, real_data, fake_data, iwass_lambda, iwass_target):
    global mixing_factors, grad_outputs
    if mixing_factors is None or real_data.size(0) != mixing_factors.size(0):
        mixing_factors = torch.cuda.FloatTensor(real_data.size(0), 1)
    mixing_factors.uniform_()
    # mixing_factors = torch.cat([torch.rand((1,1)).cuda().expand(1, *real_data.size()[1:]) for _ in range((real_data.size(0)))])

    # print('depth: sizes in loss: {} {} {}'.format(D.depth, real_data.size(), fake_data.size(), mixing_factors.size()))
    mixed_data = Variable(mul_rowwise(real_data, 1 - mixing_factors) + mul_rowwise(fake_data, mixing_factors), requires_grad=True)
    # print('dupa', mixed_data.size())
    D.depth = depth
    mixed_scores = D(mixed_data, alpha)
    if grad_outputs is None or mixed_scores.size(0) != grad_outputs.size(0):
        grad_outputs = torch.cuda.FloatTensor(mixed_scores.size())
        grad_outputs.fill_(1.)

    gradients = grad(outputs=mixed_scores, inputs=mixed_data,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - iwass_target) ** 2) * iwass_lambda / (iwass_target ** 2)

    return gradient_penalty


def wgan_gp_D_loss(D, G, depth, alpha, real_images_in, fake_latents_in,
    iwass_lambda    = 10.0,
    iwass_epsilon   = 0.001,
    iwass_target    = 1.0,
    return_all      = True):

    D.zero_grad()
    G.zero_grad()

    G.depth = depth
    D.depth = depth

    real_data_v = Variable(real_images_in)
    # train with real
    D_real = D(real_data_v, alpha)
    D_real_loss = -D_real + D_real ** 2 * iwass_epsilon

    # train with fake
    noisev = Variable(fake_latents_in, volatile=True)  # totally freeze netG
    fake = Variable(G(noisev, alpha).data)
    inputv = fake
    D_fake = D(inputv, alpha)
    D_fake_loss = D_fake

    # train with gradient penalty
    gradient_penalty = calc_gradient_penalty(D, depth, alpha, real_data_v.data, fake.data, iwass_lambda, iwass_target)
    gp = gradient_penalty
    # gp.backward()

    D_cost = (D_fake_loss + D_real_loss + gp).mean()
    if return_all:
        return D_cost, D_real_loss, D_fake_loss
    return D_cost


def wgan_gp_G_loss(G, D, depth, alpha, fake_latents_in):
    G.zero_grad()

    G.depth = depth
    noisev = Variable(fake_latents_in)
    G_new = G(noisev, alpha)
    D_new = -D(G_new, alpha)
    G_cost = D_new.mean()
    return G_cost


def evaluate_loss(
    G, D, depth, alpha, real_images_in,
    fake_latents_in, opt_g, opt_d,
    iwass_lambda    = 10.0,
    iwass_epsilon   = 0.001,
    iwass_target    = 1.0): # set cond_tweak_G=0.1 to match original improved Wasserstein implementation

    D.zero_grad()
    G.zero_grad()

    G.depth = depth
    D.depth = depth

    real_data_v = Variable(real_images_in)
    # train with real
    D_real = D(real_data_v, alpha)
    D_real_loss = -D_real + D_real**2 * iwass_epsilon

    # train with fake
    noisev = Variable(fake_latents_in, volatile=True)  # totally freeze netG
    fake = Variable(G(noisev, alpha).data)
    inputv = fake
    D_fake = D(inputv, alpha)
    D_fake_loss = D_fake

    # train with gradient penalty
    gradient_penalty = calc_gradient_penalty(D, alpha, real_data_v.data, fake.data, iwass_lambda, iwass_target)
    gp = gradient_penalty
    # gp.backward()

    D_cost = (D_fake_loss + D_real_loss + gp).mean()
    D_cost.backward()
    opt_d.step()

    ############################
    # (2) Update G network
    ###########################
    G.zero_grad()

    noisev = Variable(fake_latents_in)
    G_new = G(noisev, alpha)
    D_new = -D(G_new, alpha)
    G_cost = D_new.mean()
    G_cost.backward()

    opt_g.step()

    return G_cost, D_cost, D_real, D_fake
