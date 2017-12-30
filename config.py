# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results'


#----------------------------------------------------------------------------
# Baseline configuration from
# Appendix A.1: "1024x1024 networks used for CelebA-HQ".

run_desc = 'TBD'
random_seed = 1000
dataset = None

train = dict(                               # Training parameters:
    func                    = 'train_gan',  # Main training func.
    separate_funcs          = True,         # Alternate between training generator and discriminator?
    D_training_repeats      = 1,            # n_{critic}
    G_learning_rate_max     = 0.001,        # \alpha
    D_learning_rate_max     = 0.001,        # \alpha
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    adam_beta1              = 0.0,          # \beta_1
    adam_beta2              = 0.99,         # \beta_2
    adam_epsilon            = 1e-8,         # \epsilon
    minibatch_default       = 16,           # Minibatch size for low resolutions.
    minibatch_overrides     = {256:14, 512:6,  1024:3}, # Minibatch sizes for high resolutions.
    rampup_kimg             = 40,           # Learning rate rampup.
    lod_initial_resolution  = 4,            # Network resolution at the beginning.
    lod_training_kimg       = 150,          # Thousands of real images to show before doubling network resolution.
    lod_transition_kimg     = 150,          # Thousands of real images to show when fading in new layers.
    total_kimg              = 3000,        # Thousands of real images to show in total.
)

G = dict(                                   # Generator architecture:
    # func                    = 'G_paper',    # Configurable network template.
    fmap_base               = 512,         # Overall multiplier for the number of feature maps.
    fmap_decay              = 1.0,          # log2 of feature map reduction when doubling the resolution.
    fmap_max                = 512,          # Maximum number of feature maps on any resolution.
    latent_size             = 512,          # Dimensionality of the latent vector.
    normalize_latents       = True,         # Normalize latent vector to lie on the unit hypersphere?
    use_wscale              = True,         # Use equalized learning rate?
    use_pixelnorm           = True,         # Use pixelwise normalization?
    use_leakyrelu           = True,         # Use leaky ReLU?
)

D = dict(                                   # Discriminator architecture:
    # func                    = 'D_paper',    # Configurable network template.
    fmap_base               = 512,         # Overall multiplier for the number of feature maps.
    fmap_decay              = 1.0,          # log2 of feature map reduction when doubling the resolution.
    fmap_max                = 512,          # Maximum number of feature maps on any resolution.
    use_wscale              = True,         # Use equalized learning rate?
)

loss = dict(                                # Loss function:
    # type                    = 'iwass',      # Improved Wasserstein (WGAN-GP).
    iwass_lambda            = 10.0,         # \lambda
    iwass_epsilon           = 0.001,        # \epsilon_{drift}
    iwass_target            = 1.0,          # \alpha
)

#----------------------------------------------------------------------------
# Configuration overrides for individual experiments.

# Section 6.3: "High-resolution image generation using CelebA-HQ dataset"
if 1:
    run_desc = 'specs128'
    dataset = dict(h5_path='specs128.h5', resolution=128, max_labels=0, mirror_augment=False)