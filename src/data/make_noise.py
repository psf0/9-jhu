#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 16:33:02 2018

@author: peter
"""

import numpy as np
from scipy.io import wavfile
import os

from src.features.features_functions import makedir


def make_noise_files(overlap=False):
    # project path
    annot_dir = os.path.join('data', 'raw', 'lre17_dev_annot')
    soundfiledir = os.path.join('data', 'processed', 'lre17_dev')
    noisefiledir = os.path.join('data', 'processed', 'noise')
    if overlap:
        noisefiledir = os.path.join('data', 'processed', 'noise2')
    makedir(noisefiledir)

    for (dirpath, dirnames, filenames) in os.walk(annot_dir):
        break
    filenames = sorted(filenames)

    for name in filenames:
        with open(os.path.join(dirpath, name), 'r') as markfile:
            lines = markfile.readlines()

        fs, x = wavfile.read(os.path.join(soundfiledir, name[:-4] + 'wav'))

        I = np.zeros((len(lines) + 1, 2), dtype=np.int)
        for k, s in enumerate(lines):
            i, j = s.replace('\n', '').split()[1:]
            i, j = float(i), float(j)
            i, j = int(i * 8000), int((i + j) * 8000)
            I[k, 1], I[k + 1, 0] = i, j
        I[-1, 1] = len(x)

        """perhaps trow away some """
        I = I[(np.diff(I) > 1000).flatten()]

        if len(I) == 0:
            continue

        n = np.zeros(np.sum(np.diff(I)))
        w1, w2 = np.split(np.hanning(256), 2)
        k = 0
        for i, j in I:
            k2 = k + j - i
            n[k:k2] = x[i:j]
            n[k:k + 128] *= w1
            n[k2 - 128:k2] *= w2
            k = k2 if overlap is False else k2 - 128
        if overlap:
            n = n[:k2 + 128]

        n = n.astype(np.int16)
        wavfile.write(os.path.join(noisefiledir, name[:-4] + 'wav'), fs, n)


if __name__ == '__main__':
    # make_noise_files(overlap=False)
    make_noise_files(overlap=True)
