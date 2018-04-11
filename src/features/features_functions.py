#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:35:19 2017

@author: peter
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.signal import istft as istft_
import os

from src.data.data_functions import load_soundfile, get_SNR_dB, mod_hann


#==============================================================================
# dataloaders
#==============================================================================


def walk_dir(file_dir, extention=None):
    """"
    returns naturally sorted filepaths
    could be made a class just need a lenht
    """
    import re

    def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(_nsre, s)]

    file_paths = []
    for (dirpath, dirnames, filenames) in os.walk(file_dir):
        if extention is not None:
            filenames = [s for s in filenames if s.endswith(extention)]
        file_paths.extend([os.path.join(dirpath, s) for s in filenames])
    return sorted(file_paths, key=natural_sort_key)


def makedir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def make_identical_dir_structure(input_dir, output_dir):
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        makedir(os.path.join(output_dir, os.path.relpath(dirpath, input_dir)))


def format_paths(paths, old_basedir=None, new_basedir=None, extention=None):
    # replace pre path
    if (old_basedir is not None) or (new_basedir is not None):
        paths = [os.path.join(new_basedir, os.path.relpath(s, old_basedir)) for s in paths]
    if extention is not None:
        paths = [s.rsplit('.', 1)[0] + extention for s in paths]
    return paths


def save_audio_(data, path):
    data *= (2**15 - 1) / max(abs(data.min()), data.max())
    data = np.int16(np.around(data))
    wavfile.write(path, 8000, data)


class SubsetSampler(Sampler):
    """Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices=None, slice_=None):
        if slice_:
            indices = indices[slice_]
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def shuffle_cycle_dataset(length, samples_per_epoch, seed):
    # Enables us to use arbitrary epoch sizes
    rng = np.random.RandomState(seed)
    a = []
    while True:
        if len(a) < samples_per_epoch:
            a = a + rng.permutation(length).tolist()
        else:
            b, a = a[:samples_per_epoch], a[samples_per_epoch:]
            yield b


class Datafolder_soundfiles(Dataset):
    """
    """

    def __init__(self, x_paths=None, y_paths=None, x_scale=None, y_scale=None, standardize=True, transform=None):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.standardize = standardize
        self.transform = transform

        self.length = len(self.y_paths)

    def __getitem__(self, index):
        # example of x_path, 'data/processed/dataset_1/val/x/sample_{}.wav'

        if self.y_paths is not None:
            y = load_soundfile(self.y_paths[index])
            y = y.astype(np.float)
            if self.standardize:
                y = y / y.std()
            if self.y_scale is not None:  # normalize y and keep correct ratio for x and y
                y = y * self.y_scale[index]
        else:
            y = np.nan

        if self.x_paths is not None:
            x = load_soundfile(self.x_paths[index])
            x = x.astype(np.float)
            if self.standardize:
                x = x / x.std()
            if self.x_scale:  # normalize x and keep correct ratio for x and y
                x = x * self.x_scale[index]
        else:
            x = np.nan

        if self.transform is not None:
            transformed_data = self.transform(x=x, y=y)
            return index, transformed_data
        else:
            return index, x, y

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__doc__


#==============================================================================
#
#==============================================================================

def load_dataset_(x_dir, y_dir, transform=None):
    x_paths = walk_dir(x_dir)
    x_scale = []
    for s in x_paths:
        index = int(s.rsplit('.', 1)[0].rsplit('_', 1)[-1])
        SNR_dB = get_SNR_dB(index)
        SNR = 10**(SNR_dB / 10.)
        x_scale.append((SNR / (SNR + 1))**.5)

    dataset = Datafolder_soundfiles(
        x_paths=walk_dir(x_dir), y_paths=walk_dir(y_dir),
        x_scale=x_scale, y_scale=None, standardize=True, transform=transform)
    return dataset


def load_dataset_2_train(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_2', 'train', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_2', 'train', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


def load_dataset_2_val(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


def load_dataset_3_train(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_3', 'train', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_3', 'train', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


def load_dataset_3_val(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


def load_dataset_4_train(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_4', 'train', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_4', 'train', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


def load_dataset_4_val(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


def load_dataset_5_train(transform=None):
    x_dir = os.path.join('data', 'processed', 'dataset_5', 'train', 'x')
    y_dir = os.path.join('data', 'processed', 'dataset_5', 'train', 'y')
    return load_dataset_(x_dir, y_dir, transform=transform)


# def Load_Chime4(test=False, transform=None):
#     fix chime4 path
#     pass


#==============================================================================
# data trainsforms
#==============================================================================


class feature_transform_0:
    def __init__(self, normalize_dataset_dir=None):
        self.mel = np.loadtxt(os.path.join('data', 'external', 'mel_8000_256_100.csv'))
        if normalize_dataset_dir is not None:
            self.mel_mean = np.loadtxt(
                os.path.join(normalize_dataset_dir, 'transform_0_mean.txt'))[:, None]
            self.mel_std = np.loadtxt(
                os.path.join(normalize_dataset_dir, 'transform_0_std.txt'))[:, None]
        else:
            self.mel_mean = 0
            self.mel_std = 1

    def __call__(self, Y):
        Y_m = np.abs(Y)
        Y_mmel = np.dot(self.mel, Y_m)
        np.log1p(Y_mmel, out=Y_mmel)
        Y_mmel -= self.mel_mean
        Y_mmel /= self.mel_std
        return Y_mmel


class transform_tf:
    """
    """
    def __init__(self, feature_transform, target_transform=None):
        self.feature_transform = feature_transform
        self.target_transform = target_transform
        self.awin = mod_hann(256)  # use sqrt hann. window (scypy does the sqrt)

    def __call__(self, x=None, y=None):
        f, t, Y = stft(y, fs=8000, window=self.awin, nperseg=256,
                       noverlap=128, return_onesided=True)
        data = self.feature_transform(Y=Y)
        data = torch.from_numpy(data.astype(np.float32))

        if self.target_transform is None:
            return data, np.abs(Y), np.angle(Y), len(y)

        f, t, X = stft(x, fs=8000, window=self.awin, nperseg=256,
                       noverlap=128, return_onesided=True)
        targetX = self.target_transform(X=X, Y=Y)
        targetX = torch.from_numpy(targetX.astype(np.float32))
        Y_m = torch.from_numpy(np.abs(Y).astype(np.float32))
        return data, Y_m, targetX


# class transform_0:
#     """
#     """
#     def __init__(self, normalize=True, mode=None, dataset_dir=None):
#         assert mode in ['labels', 'runtime', 'debug', 'labels_psa']
#         self.mode = mode
#         self.awin = mod_hann(256)  # use sqrt hann. window (scypy does the sqrt)
#         self.mel = np.loadtxt(os.path.join('data', 'external', 'mel_8000_256_100.csv'))
#
#         if normalize:
#             self.mel_mean = np.loadtxt(os.path.join(dataset_dir, 'transform_0_mean.txt'))[:, None]
#             self.mel_std = np.loadtxt(os.path.join(dataset_dir, 'transform_0_std.txt'))[:, None]
#         else:
#             self.mel_mean = 0
#             self.mel_std = 1
#
#     def __call__(self, x=None, y=None):
#         f, t, Y = stft(y, fs=8000, window=self.awin, nperseg=256,
#                        noverlap=128, return_onesided=True)
#
#         Y_m = np.abs(Y)
#         Y_mmel = np.dot(self.mel, Y_m)
#         np.log1p(Y_mmel, out=Y_mmel)
#         Y_mmel -= self.mel_mean
#         Y_mmel /= self.mel_std
#         data = torch.from_numpy(Y_mmel.astype(np.float32))
#
#         if self.mode == 'runtime':
#             Y_a = np.angle(Y)
#             length = len(y)
#             return data, Y_m, Y_a, length
#
#         f, t, X = stft(x, fs=8000, window=self.awin, nperseg=256,
#                        noverlap=128, return_onesided=True)
#         X_m = np.abs(X)
#
#         if self.mode == 'labels':
#             targetX = torch.from_numpy(X_m.astype(np.float32))
#             targetY = torch.from_numpy(Y_m.astype(np.float32))
#             return data, targetY, targetX
#         elif self.mode == 'labels_psa':
#             # targetX = np.real(X / Y) * np.abs(Y)
#             targetX = X_m * np.cos(np.angle(X) - np.angle(Y))
#             targetX = torch.from_numpy(targetX.astype(np.float32))
#             targetY = torch.from_numpy(Y_m.astype(np.float32))
#             return data, targetY, targetX
#         else:
#             X_a, Y_a = np.angle(X), np.angle(Y)
#             return data, X_m, Y_m, X_a, Y_a, x, y


def istft(X_m, X_a, new_length=None):
    awin = mod_hann(256)
    X = X_m * np.exp(1j * X_a)

    _, x = istft_(X, fs=8000, window=awin, nperseg=256, noverlap=128, input_onesided=True)

    if new_length <= len(x):
        x = x[:new_length]
    else:
        print(new_length, len(x))
        raise NotImplementedError('padding is not implemented only cutting')

    return x


if __name__ == '__main__':
    pass
