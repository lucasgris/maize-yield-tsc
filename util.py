import random
import glob
import sys

import concurrent.futures
import os
import time
import glob
import shutil
import numpy as np
import subprocess

from cv2 import *
from PIL import Image
from tifffile import imsave

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np


DEFAULT_SHAPE = (53, 53)
# RESCALE_PERC_BEFORE_ROT = 1000


def remove_no_data(mat, default='zeros'):
    mat[mat <= 0] = np.NaN
    if default == 'zeros':
        mat[np.isnan(mat)] = 0
    elif default == 'ones':
        mat[np.isnan(mat)] = 1
    return mat


def read_tif(path, adjust_shape=None, **kwargs):  # ONLY 1 CHANNEL!!
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    im_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if adjust_shape is not None:
        im_array = adjust_mat(im_array, adjust_shape)
    return remove_no_data(im_array, **kwargs)


def auto_rotate_crop(im_array):
    pass


def crop_mat(mat, shape=DEFAULT_SHAPE, start=None,):
    crop_w = shape[1]
    crop_h = shape[0]
    if type(start) == tuple:
        cropw_s = start[0]
        croph_s = start[1]
    elif start == 'random':
        cropw_s = random.randint(0, crop_w)
        croph_s = random.randint(0, crop_h)
    elif start == 'center':
        cropw_s = crop_w // 2
        croph_s = crop_h // 2
    else:
        cropw_s = 0
        croph_s = 0
    mat = mat[croph_s:crop_h, cropw_s:crop_w]
    return mat


def pad_mat(mat, shape=DEFAULT_SHAPE, start=None, value=0):
    pad_w = shape[1] - mat.shape[1]
    pad_h = shape[0] - mat.shape[0]
    if type(start) == tuple:
        padwl = start[0]
        padhb = start[1]
    elif start == 'random':
        padwl = random.randint(0, pad_w)
        padhb = random.randint(0, pad_h)
    elif start == 'center':
        padwl = pad_w // 2
        padhb = pad_h // 2
    else:
        padwl = pad_w
        padhb = 0
    padwr = pad_w - padwl
    padht = pad_h - padhb
    return np.pad(mat, pad_width=((padhb, padht), (padwr, padwl)),
                  mode='constant', constant_values=value)


def adjust_mat(mat, shape=DEFAULT_SHAPE, **kwargs):
    if mat.shape == shape:
        return mat
    # Crop first
    if mat.shape[0] > shape[0] and mat.shape[1] <= shape[1]:
        mat = crop_mat(mat, (shape[0], mat.shape[1]))
    elif mat.shape[1] > shape[1] and mat.shape[0] <= shape[0]:
        mat = crop_mat(mat, (mat.shape[0], shape[1]))
    elif mat.shape[0] > shape[0] and mat.shape[1] > shape[1]:
        mat = crop_mat(mat, shape)
    # Pad
    return pad_mat(mat, shape, **kwargs)


def rand_pad_image(im_array, shape=DEFAULT_SHAPE, **kwargs):
    pad_w = shape[1] - im_array[:, :, 0].shape[1]
    pad_h = shape[0] - im_array[:, :, 0].shape[0]
    padwl = random.randint(0, pad_w)
    padhb = random.randint(0, pad_h)
    return np.asarray([pad_mat(im_array[:, :, i], shape, start=(padwl, padhb),
                               **kwargs)
                       for i in range(im_array.shape[-1])])


def export(im_array, out_path, npy=True, png=False, tif=True,
           verbose_level=0):
    if verbose_level > 1:
        print('Exporting', out_path)
    if npy:
        np.save(out_path, im_array)
    if png:
        impng = 255 * (1.0 - im_array)
        impng = Image.fromarray(impng.astype(np.uint8))
        impng.save(f"{out_path}.png")
    if tif:
        imsave(f"{out_path}.tif", im_array)


def merge_channels(channels):
    multichannel = np.zeros((channels[0].shape[0], channels[0].shape[1],
                             len(channels)))
    for i, ch in enumerate(channels):
        multichannel[:, :, i] = ch
    return multichannel


def rot90(img, verbose_level=1, **kwargs):
    if verbose_level > 0 and img.shape[0] != img.shape[1]:
        print("Warning: it is not possible to rotate using this method: np.rot90. "
              f"Invalid shape: {img.shape}. Skiping.")
        return img
    elif verbose_level == 0 and img.shape[0] != img.shape[1]:
        raise ValueError("It is not possible to rotate using this method: np.rot90. "
                         f"Invalid shape: {img.shape}")
    img = img.copy()
    for i in range(img.shape[-1]):
        img[:, :, i] = np.rot90(img[:, :, i], **kwargs)
    return img


def rotation(img, angle):
    img = img.copy()
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


def vertical_flip(img, flag):
    img = img.copy()
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


def horizontal_flip(img, flag):
    img = img.copy()
    if flag:
        return cv2.flip(img, 1)
    else:
        return img


def flip(img):
    im_flip_hor = horizontal_flip(img, True)
    im_flip_diag = vertical_flip(im_flip_hor, True)
    return im_flip_diag


def fill(img, h, w):
    img = img.copy()
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def zoom(img, value, start=None):
    img = img.copy()
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    # value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    if start == 'center':
        h_start = h-h_taken // 2
        w_start = w-w_taken // 2
    elif start == 'random':
        h_start = random.randint(0, h-h_taken)
        w_start = random.randint(0, w-w_taken)
    elif type(start) == tuple:
        h_start, w_start = start
    else:
        h_start = w_start = 0
    if len(img.shape) > 2:
        img = img[h_start:h_start+h_taken, w_start:w_start+w_taken:]
    else:
        img = img[h_start:h_start+h_taken, w_start:w_start+w_taken]
    img = fill(img, w, h)
    return img


def vertical_shift(img, ratio=0.0):
    img = img.copy()
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    # ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if len(img.shape) > 2:
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :, :]
    else:
        if ratio > 0:
            img = img[:int(h-to_shift), :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :]
    img = fill(img, w, h)
    return img


def horizontal_shift(img, ratio=0.0):
    img = img.copy()
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    # ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio

    if len(img.shape) > 2:
        if ratio > 0:
            img = img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = img[:, int(-1*to_shift):, :]
    else:
        if ratio > 0:
            img = img[:, :int(w-to_shift)]
        if ratio < 0:
            img = img[:, int(-1*to_shift):]

    img = fill(img, w, h)
    return img

#  550, 660, 735 e 790


def ndvi(im):
    im = (im[:, :, 3]-im[:, :, 1]) / (im[:, :, 3]+im[:, :, 1])
    return np.expand_dims(im, axis=-1)


def rdvi(im):
    im = (im[:, :, 3]-im[:, :, 1]) / (im[:, :, 3]+im[:, :, 1])**(1/2)
    return np.expand_dims(im, axis=-1)


def osavi(im):
    im = ((1.16)*(im[:, :, 3]-im[:, :, 1])) / (im[:, :, 3]+im[:, :, 1]+0.16)
    return np.expand_dims(im, axis=-1)


def msr(im):
    im = (im[:, :, 3]/im[:, :, 1]) - 1/((im[:, :, 3]/im[:, :, 1])**(1/2)) + 1
    return np.expand_dims(im, axis=-1)


def mcari1(im):
    im = 1.2*(2.5*((im[:, :, 3] - im[:, :, 1])) -
              1.3*(im[:, :, 3]/im[:, :, 0]))
    return np.expand_dims(im, axis=-1)


def mcari2(im):
    m1 = 1.2*(2.5*((im[:, :, 3] - im[:, :, 1])) -
              1.3*(im[:, :, 3]/im[:, :, 0]))
    im = m1 / ((2*im[:, :, 3]+1)**(1/2) -
               (6*im[:, :, 3]-5*(im[:, :, 1]**(1/2))) - .5)**1/2
    return np.expand_dims(im, axis=-1)


def pssra(im):
    im = im[:, :, 3]/im[:, :, 1]
    return np.expand_dims(im, axis=-1)


def g_rgb_ratio(im):
    im = im[:, :, 0]/im[:, :, 1]
    return np.expand_dims(im, axis=-1)


def augment_image(im_array, crop, instance, out_dir, augment_type='default',
                  export=False, verbose_level=0):
    if verbose_level > 0:
        print('Performing data augmentation of', instance)
        print('Type of augmentation:', augment_type)

    # print(im_array.shape)
    augmented_images = []
    if augment_type == 'simple' or augment_type == 'default' or augment_type == 'advanced':
        augmented_images.append((flip(im_array), "__FLIP"))
    if augment_type == 'default' or augment_type == 'advanced':
        augmented_images.append((vertical_flip(im_array, True), "__VFLIP"))
        augmented_images.append((horizontal_flip(im_array, True), "__HFLIP"))
        if im_array.shape[0] == im_array.shape[1]:
            augmented_images.append((rot90(im_array), "__NPROT90"))
            augmented_images.append((rot90(im_array, k=1), "__NPROT90K1"))
        elif verbose_level > 2:
            print("Warning: 90 degree rotation not performed because the "
                  "shape of the array is not a square.")
    if augment_type == 'advanced':
        augmented_images.append((rotation(im_array, 30), "__ROT30"))
        augmented_images.append((rotation(im_array, 120), "__ROT120"))
        augmented_images.append((rotation(im_array, 160), "__ROT160"))
        augmented_images.append((zoom(im_array, .2), "__ZOOM2"))
        augmented_images.append((zoom(im_array, .3), "__ZOOM3"))
        augmented_images.append((zoom(im_array, .6), "__ZOOM6"))
        augmented_images.append(
            (horizontal_shift(im_array, .25), "__HSHIFT25"))
        augmented_images.append(
            (horizontal_shift(im_array, .50), "__HSHIFT50"))
        augmented_images.append(
            (horizontal_shift(im_array, .60), "__HSHIFT60"))
        augmented_images.append((vertical_shift(im_array, .25), "__VSHIFT25"))
        augmented_images.append((vertical_shift(im_array, .50), "__VSHIFT50"))
        augmented_images.append((vertical_shift(im_array, .60), "__VSHIFT60"))

    if export:

        out_npy = os.path.join(out_dir, 'NPY', crop, instance)
        # out_tif = os.path.join(out_dir, 'TIFF', crop, instance)

        for im_aug in augmented_images:
            export(im_aug[0], os.path.join(out_npy, f'{instance}_{im_aug[1]}'),
                   npy=True, tif=False, png=False, verbose_level=verbose_level)
    return [pair[0] for pair in augmented_images]


def _process_bands(bands, data, out_dir=None, im_shape=DEFAULT_SHAPE, augment_type=None,
                   export=False, verbose_level=0, transform_f=None, **kwargs):
    if augment_type is not None and augment_type != 'simple':
        if len(bands) > 1:
            fullim = merge_channels(bands)
        else:
            fullim = np.asarray(bands[0])
            fullim = np.expand_dims(fullim, axis=-1)
            # print(fullim.shape)
        bands = rand_pad_image(fullim, shape=im_shape)
    else:
        bands = [pad_mat(b, shape=im_shape) for b in bands]

    if len(bands) > 1:
        fullim = merge_channels(bands)
    else:
        fullim = np.asarray(bands[0])
        fullim = np.expand_dims(fullim, axis=-1)

    if transform_f is not None:
        fullim = transform_f(fullim)
        fullim[np.isnan(fullim)] = 0        

    instance = data['Instance']
    crop = data['Crop']

    if export and out_dir is not None:
        out_npy = os.path.join(out_dir, 'NPY', crop, instance)
        os.makedirs(out_npy, exist_ok=True)
        export(fullim, os.path.join(out_npy, instance), npy=True,
               tif=False, png=False, verbose_level=verbose_level)

    if augment_type is not None:
        images = augment_image(fullim, crop, instance, out_dir, augment_type,
                               export=export, verbose_level=verbose_level)
    else:
        images = []
    images.append(fullim)

    return instance, images


def process(data, out_dir=None, im_shape=DEFAULT_SHAPE, augment_type=None,
            export=False, verbose_level=0, transform_f=None, **kwargs):
    bands = []
#     for fp in data.iloc[2:6]:
    for fp in [data['B1File'], data['B2File'], data['B3File'], data['B4File']]:
        if isinstance(fp, pd.Series):
            fp = fp.values[0]
        if verbose_level > 1:
            print('Opening band', fp)
        bands.append(read_tif(fp, adjust_shape=im_shape))

    return _process_bands(bands,
                          data,
                          out_dir=out_dir,
                          im_shape=im_shape,
                          augment_type=augment_type,
                          export=export,
                          verbose_level=verbose_level,
                          transform_f=transform_f,
                          **kwargs)


def process_one_band(data, band='B1File', out_dir=None, im_shape=DEFAULT_SHAPE,
                     augment_type='default', export=False, verbose_level=0,
                     transform_f=None, **kwargs):
    bands = []
    for fp in [data['B1File']]:
        if isinstance(fp, pd.Series):
            fp = fp.values[0]
        if verbose_level > 1:
            print('Opening band', fp)
        bands.append(read_tif(fp, adjust_shape=im_shape))

    return _process_bands(bands,
                          data,
                          out_dir=out_dir,
                          im_shape=im_shape,
                          augment_type=augment_type,
                          export=export,
                          verbose_level=verbose_level,
                          transform_f=transform_f,
                          **kwargs)


def process_rt(data, out_dir=None, im_shape=DEFAULT_SHAPE, augment_type='default', temp_dir=False,
               verbose_level=0, **kwargs):
    pass


def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'B2File' in df:
        return df[['Instance', 'Crop', 'B1File', 'B2File', 'B3File',
                   'B4File']]
    else:
        return df[['Instance', 'Crop', 'B1File']]


if __name__ == '__main__':
    import argparse

    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates the dataset.')
    parser.add_argument('src', help='Input Directory')
    parser.add_argument('out', help='Output Directory')
    parser.add_argument('csv', help='CSV file data (see create_csv_data)')
    parser.add_argument('--shape', help='Shape of image',
                        default=DEFAULT_SHAPE)
    # parser.add_argument('--rescale_percent_before_rot',
    #                     help='Reescales the image before performing the auto'
    #                     ' rotation with the provided value (percent)',
    #                     default=RESCALE_PERC_BEFORE_ROT)
    parser.add_argument('--auto_rotate', help='Auto rotate image and crop to '
                        'fill all area.',
                        action='store_true', default=False)
    parser.add_argument('--skip_merge_channels', help='Skip merge channels',
                        action='store_true', default=False)
    parser.add_argument('--skip_pad', help='Skip padding images with shape'
                        '(Can cause different shapes in data)',
                        action='store_false', default=True)
    parser.add_argument('--workers', help='Number of workers (multiprocessing)',
                        type=int, default=1)
    parser.add_argument('--verbose', help='Verbose level',
                        type=int, default=0)
    aug_args = parser.add_argument_group('Data augmentation options')
    aug_args.add_argument('--basic',
                          help='Generate new images with flipping and padding',
                          action='store_true')
    aug_args.add_argument('--advanced',
                          help='Generate new images with flipping, zooming, '
                          'rotations and shifting, as well as the basic '
                          'operations.',
                          action='store_true')

    arguments = parser.parse_args()
    os.makedirs(arguments.out, exist_ok=True)

    def process_files(input_path, dataframe, out_path, augmentation, im_shape,
                      n_workers=1, verbose_level=0, **kwargs):
        #         dataframe['Path'] = dataframe['Crop'].apply(lambda c: os.path.join(
        #             input_path, c))
        for key in ['B1File', 'B2File', 'B3File', 'B4File']:
            dataframe[key] = dataframe.apply(
                lambda x: os.path.join(input_path, x['Crop'], x[key]), axis=1)
        if n_workers == 1:
            for i, data in dataframe.iterrows():
                process(data, out_path, im_shape, augment=augmentation,
                        verbose_level=verbose_level, export=True, **kwargs)
                if verbose_level > 0:
                    print('Processing', data['Instance'])
        else:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=arguments.workers) as executor:

                futures = [executor.submit(process, data, out_path, im_shape,
                                           augment=augmentation,
                                           export=True,
                                           verbose_level=verbose_level,
                                           **kwargs)
                           for i, data in dataframe.iterrows()]

                for f in concurrent.futures.as_completed(futures):
                    if verbose_level > 0:
                        print('Done', f.result()[0])

    process_files(input_path=arguments.src,
                  dataframe=parse_csv(arguments.csv),
                  out_path=arguments.out,
                  augmentation=('basic' if arguments.basic else
                                'advanced' if arguments.advanced
                                else None),
                  im_shape=arguments.shape,
                  n_workers=arguments.workers,
                  skip_pad=arguments.skip_pad,
                  verbose_level=arguments.verbose)
