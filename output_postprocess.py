import os
import utils
import numpy as np
try:
    import PIL.Image
except ImportError as e:
    print('Unable to load PIL.Image: {}.\nExporting images won\'t work.'.format(e))
try:
    import librosa as lbr
except ImportError as e:
    print('Unable to load librosa: {}.\nExporting sound won\'t work.'.format(e))
from utils import adjust_dynamic_range, numpy_upsample_nearest


class Postprocessor(object):

    def __init__(self, samples_path='.'):
        self.samples_path = samples_path


class ImageSaver(Postprocessor):

    output_file_format = 'fakes_{}.png'

    def __init__(self, samples_path='.', drange=(-1,1), resolution=512, create_subdirs=True):
        super(ImageSaver, self).__init__(samples_path)
        self.samples_path = samples_path
        if create_subdirs:
            os.makedirs(self.samples_path, exist_ok=True)
        # Setup snapshot image grid.
        self.resolution = resolution
        self.drange = drange
        self.mode = None

    def create_image_grid(self, images):
        (count, channels, img_h, img_w) = images.shape

        grid_w = max(int(np.ceil(np.sqrt(count))), 1)
        grid_h = max((count - 1) // grid_w + 1, 1)

        grid = np.zeros((channels,) + (grid_h * img_h, grid_w * img_w), dtype=images.dtype)
        for i in range(count):
            x = (i % grid_w) * img_w
            y = (i // grid_w) * img_h
            grid[:, y: y + img_h, x: x + img_w] = images[i]
        return grid

    def convert_to_pil_image(self, image):
        format = 'RGB'
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0]
                format = 'L'
            else:
                image = image.transpose(1, 2, 0)
                format = 'RGB'

        image = utils.adjust_dynamic_range(image, self.drange, (0, 255))

        image = image.round().clip(0, 255).astype(np.uint8)
        return PIL.Image.fromarray(image, format)

    def __call__(self, output, description):
        if self.resolution is not None:
            output = numpy_upsample_nearest(output, 2, size=self.resolution)
        im = self.create_image_grid(output)
        im = self.convert_to_pil_image(im)
        fname = self.output_file_format
        if type(description) is int:
            fname = fname.format('{:06}')
        im.save(os.path.join(self.samples_path, fname.format(description)))


class SoundSaver(Postprocessor):

    output_file_format = 'fakes_sound_{}_{}.wav'

    def __init__(self, samples_path='.', drange=(-1, 1), resolution=512, mode='abslog', sample_rate=16000,
                 hop_length=128, create_subdirs=True, verbose=False, griffin_lim_iter=100):
        super(SoundSaver, self).__init__(samples_path)
        self.samples_path = samples_path
        if create_subdirs:
            os.makedirs(self.samples_path, exist_ok=True)
        self.drange = drange
        self.mode = mode
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.verbose = verbose
        self.resolution = resolution
        self.griffin_lim_iter = griffin_lim_iter

    def reconstruct_from_magnitude(self, stft_mag):
        n_fft = (stft_mag.shape[0] - 1) * 2
        x = np.random.randn((stft_mag.shape[1] - 1) * self.hop_length)
        for i in range(self.griffin_lim_iter):
            stft_rec = lbr.stft(x, n_fft=n_fft, hop_length=self.hop_length)
            angle = np.angle(stft_rec)
            my_stft = stft_mag * np.exp(1.0j * angle)
            if self.verbose: # and i == it - 1:
                prev_x = x
            x = lbr.istft(my_stft, hop_length=self.hop_length)
            if self.verbose:  # and i == it - 1:
                mse = np.sqrt(np.square(x - prev_x).sum())  # logmse would be more appropriate?
                print('MSE between sub- and ultimate iteration: {}'.format(mse))
        return x

    def image_to_sound(self, image):
        if self.mode == 'reallog' or self.mode == 'abslog':
            x = np.zeros((image.shape[0] + 1, image.shape[1]))  # real spectrograms have 2**i + 1 freq bins
            # x.fill(image.mean())
            x[:image.shape[0], :image.shape[1]] = image
            if self.mode == 'reallog':
                signed = adjust_dynamic_range(x, self.drange, (-1, 1))
                sgn = np.sign(signed)
                real_pt_stft = (np.exp(np.abs(signed)) - 1) * sgn
                signal = lbr.istft(real_pt_stft, self.hop_length)
            else:
                x = adjust_dynamic_range(x, self.drange, (0, 255))
                signal = self.reconstruct_from_magnitude(x)
        elif self.mode == 'raw':
            signal = image.ravel()
        else:
            raise Exception(
                'image_to_sound: unrecognized mode: {}. Available modes are: reallog, abslog, raw.'.format(self.mode)
            )
        signal = signal / np.abs(signal).max()
        return signal

    def output_wav(self, signal, samples_description, ith):
        fname = self.output_file_format
        if type(samples_description) is int:
            fname = fname.format('{:06}', '{:02}')
        else:
            fname = fname.format('{}', '{:02}')
        try:
            lbr.output.write_wav(
                os.path.join(self.samples_path, fname.format(samples_description, ith)),
                signal,
                self.sample_rate,
                norm=True
            )
        except Exception as e:
            with open(os.path.join(self.samples_path, 'error_{}_{}.txt'.format(samples_description, ith)), 'w') as f:
                f.write('Exception trying to save sound: {}'.format(e))

    def __call__(self, output, samples_description):
        times_smaller = self.resolution // output.shape[-1]
        if self.mode == 'raw':
            times_smaller *= times_smaller
        for i, img in enumerate(output):
            signal = self.image_to_sound(img[0])
            signal = numpy_upsample_nearest(signal, 1, scale_factor=times_smaller)
            self.output_wav(signal, samples_description, i)
