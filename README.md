# pggan-pytorch

A PyTorch simplified implementation of [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).

![A visual representation of the PGGAN architecture](http://deepsound.io/images/pggan/fig1.png)

It's based on the reference implementation in Theano: https://github.com/tkarras/progressive_growing_of_gans and a few other implementations named in acknowledgements below. It's more modularized and specifically designed for training the Progressively Growing GAN networks with WGAN-GP loss. Main focus for this implementation in terms of domain of generation were spectrograms and other visual representations of audio and thus it contains a few helpers for these tasks (but also works for any other kind of image). For more details on results in audio generation, see our blog post: http://deepsound.io/samplernn_pytorch.html.

The model and implementation proved itself to be succesful - and as for now, the authors are not aware of any other succesful GAN application to audio signal generation.

## Dependencies

This code requires Python 3.5+ and was tested under PyTorch 0.2+. Other requirements depend on the usage, most important for sound generation being Librosa 0.4.3+. Installation instructions for PyTorch are available on their website: http://pytorch.org/. You can install the rest of the dependencies by running `pip install -r requirements.txt`.

## Datasets

Training on new data is as simple as implementing your own DepthDataset (see dataset.py) with all required methods. A few useful Datasets were already implemented:
    * OldH5Dataset - allows use of (fitting in memory!) datasets from original implementation
    * DefaultImageFolderDataset - takes any folder with images and creates a proper dataset
    * SoundImageDataset - takes any folder with .wav files of length at least (n_fft\*hop_length/2 + n_fft) (where n_fft and hop_length are desired parameters of stft). For example, a good start is to use 16000 hz, 5 second audio created by script from our [SampleRNN PyTorch implementation](https://github.com/deepsound/samplernn-pytorch/). This class can be a dataset of either magnitude spectrograms, "raw" data or real-part stfts, with best results obtained with magnitude spectrograms.

## Training

To train the model you need to run `train.py`. All model and training hyperparameters are settable in the command line in a very generic way - you can essentially pass ClassName.init_param_name for every class that will be instantiated during the training process. Sometimes you need to specify which classes you may want to instantiate, e.g. which kind of dataset or which output postprocessors. Most hyperparameters have sensible default values, so you don't need to provide all of them. Run `python train.py -h` for details. With piano dataset from our script mentioned before, you can run:

```
python pggan.py --exp_name TEST --dataset_class SoundImageDataset --SoundImageDataset.dir_path PATH_TO_DATASET_FOLDER --SoundImageDataset.preload True --SoundImageDataset.img_mode abslog --save_dataset abs.pkl --postprocessors ['ImageSaver', 'SoundSaver'] --ImageSaver.samples_path images --SoundSaver.samples_path sound
```

If you would like to keep lod information from original implementation in logging, add --DepthManager.max_lod and --DepthManager.depth_offset parameters.

The results - training log, model checkpoints and postprocessed generated samples will be saved in `results_dir/NUMEXP-exp_name/`. NUMEXP is number one greater than maximal number of experiments in results directory.

We also have an option to monitor the metrics using [CometML](https://www.comet.ml/). To use it, just pass your API key as `--comet_key` parameter to `train.py`. For newer versions of CometML, you may also pass sensible `--comet_project_name`.

## Main differences compared to original implementation

Main difference is substitution of LOD parameter in exchange for two, more related to the paper parameters: `depth` and `alpha`. The inspiration for this is also the Chainer implementation (see acknowledgements). Also, this implementation is completely agnostic of a term of `resolution` and everything can be specified just by proper `depth`, `max_depth` and `depth_offset` parameters. In this implementation, parameter `DepthManager.max_lod` corresponds to `log2(resolution)` and `model_dataset_depth_offset` (in the selected dataset) corresponds to `log2(initial_resolution)`. This is motivated by the following logic: neural models should always have their parameter `depth` corresponding to the index of deepest layer trainable, while datasets can also provide versions of data for other depths (it works this way also in original implementation, where dataset always creates inputs downsampled down to 2^0 x 2^0). If you provide `--DepthManager.max_lod` and `--DepthManager.depth_offset` parameters, you can have old-style lod displayed in lod, equal to `max_lod - depth_offset - depth + 1 - alpha`.

There is a lot of unimplemented features of the original paper, such as conditional training, a few network parameters etc. which we didn't find useful (for now!) during spectrogram generation which is main purpose of this implementation. All the default parameters for generating the 1024x1024 faces from the original paper are implemented.

The training process in Trainer class resembles the original implementation notion of "nimg", "kimg" and "ticks" instead of "iterations" or "epochs". For the ease of use of several built-in plugins and to satisfy both PyTorch and the original implementation, "tick" is equivalent to "epoch".

Take note, that with simple proper initializations of depth / plugin parameters, you can also train regular (non-progressive) GANs with this framework or your own progressively growing networks, which only need to specify their max_depth and make use of depth and alpha parameters.

## Acknowledgments

Original implementation: https://github.com/tkarras/progressive_growing_of_gans
Few other inspirations:
    * chainer-PGGAN: https://github.com/joisino/chainer-PGGAN
    * PyTorch - progressive growing of gans: https://github.com/github-pengge/PyTorch-progressive_growing_of_gans
    * wgan-gp pytorch - https://github.com/caogang/wgan-gp

We want to highlight, that these were strong, but still just an inspiration for rewriting the code from scratch, even though a clear resemblence of the code from mentioned implementation can be seen.
