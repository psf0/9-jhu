#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:35:19 2017

@author: peter
"""

from torch.utils.data import Dataset
import numpy as np
from scipy.io import wavfile
import os
import subprocess

#==============================================================================
#
#==============================================================================


def get_speech_list(test_percent=0.10, swb=True, fisher=False, lre_train=False):
    """
    """

    s = {'swbd_cellular1': '/export/corpora5/LDC/LDC2001S13',
         'swbd_cellular2': '/export/corpora5/LDC/LDC2004S07',
         'swbd2_phase1': '/export/corpora3/LDC/LDC98S75',
         'swbd2_phase2': '/export/corpora5/LDC/LDC99S79',
         #     'swbd2_phase3' : '/export/corpora5/LDC/LDC2002S06/',
         }

    f = {'fisher1t': '/export/corpora3/LDC/LDC2004T19',
         'fisher2t': '/export/corpora3/LDC/LDC2005T19',
         'fisher1s': '/export/corpora3/LDC/LDC2004S13',
         'fisher2s': '/export/corpora3/LDC/LDC2005S13',
         }

    l = {'lre_train': '/export/corpora/LDC/LDC2017E22/LDC2017E22_2017_NIST_Language_Recognition_Evaluation_Training_Data'
         }

    patch_dict = {}
    if swb:
        patch_dict = {**patch_dict, **s}
    if fisher:
        patch_dict = {**patch_dict, **f}
        raise NotImplementedError('is fisher working? check first')
    if lre_train:
        patch_dict = {**patch_dict, **l}

    speechlist = []
    for sp in patch_dict.values():
        for (dirpath, dirnames, filenames) in os.walk(sp):
            a = [dirpath + '/' + f for f in filenames if f.endswith('.sph')]
            if a:
                speechlist.extend(a)
    try:  # known broken files
        speechlist.remove('/export/corpora3/LDC/LDC98S75/swb2/sw_13263.sph')
    except:
        pass

    # split into test and train
    J = np.random.RandomState(42).permutation(len(speechlist))
    i = int(len(J) * test_percent)
    test_speechlist = [speechlist[j] for j in J[:i]]
    train_speechlist = [speechlist[j] for j in J[i:]]

    return train_speechlist, test_speechlist


def get_noise_w(noisefilepath, test_percent=0.10):
    """
    each noise file has a wight that is porportional to its sample count.
    """
    for (dirpath, dirnames, noisefilenames) in os.walk(noisefilepath):
        break
    noisefilenames = sorted(noisefilenames)

    w = []
    l = 0
    for name in noisefilenames:
        fs, n = wavfile.read(os.path.join(noisefilepath, name))
        w.append(len(n))
        l += len(n)
#    print(l/8000/3600) # hours of noise data

    # test and train split
    w = np.array(w, dtype=np.float)
    test_sample_count = w.sum() * test_percent
    J = np.random.RandomState(42).permutation(len(noisefilenames))
    count = 0  # count to .15 of the samples
    for i, j in enumerate(J):
        count += w[j]
        if count > test_sample_count:
            break

    test_noisefilenames = [noisefilenames[j] for j in J[:i]]
    train_noisefilenames = [noisefilenames[j] for j in J[i:]]

    test_w = w[J[:i]]
    train_w = w[J[i:]]

    if test_noisefilenames:
        test_w = np.cumsum(test_w)
        test_w /= test_w[-1]

    train_w = np.cumsum(train_w)
    train_w /= train_w[-1]

    return (train_noisefilenames, train_w), (test_noisefilenames, test_w)
#    w = np.cumsum()
#    w /= w[-1]
#    return noisefilenames, w

#==============================================================================
#
#==============================================================================


def read_sample_count(filname):
    with open(filname, 'rb') as f:
        f = f.read(1024)
    ii = f.find(b'sample_count')
    if ii != -1:
        jj = f.find(b'\n', ii)
        length = int(f[ii:jj].split()[-1])
        return length
    else:
        raise ValueError('Failed to read NIST file header')


def nist_file_reader(speechfile_name, start='', stop=''):
    inputs_ = [
        "/export/b16/pfredericks/AAU/9-jhu/src/external/sph2pipe_v2.5/sph2pipe",
        '-p',     # 16-bit linear pmc
        '-c1',    # first channel
    ]
    #'-s :10', #samples
    if (start != '' or stop != ''):
        inputs_ += ['-s {}:{}'.format(start, stop)]
    inputs_ += [speechfile_name]

    try:
        a = subprocess.run(inputs_, stdout=subprocess.PIPE)
    except:
        inputs_[0] = get_base_path() + "src/external/sph2pipe_v2.5/sph2pipe"
        a = subprocess.run(inputs_, stdout=subprocess.PIPE)
    x = np.frombuffer(a.stdout[1024:], dtype=np.int16)

    if not len(x):
        raise ValueError('Warning filelengt is zero', speechfile_name, start, stop)
    if (start != '' and stop != '' and len(x) != stop - start):
        raise ValueError('Warning filelengt is not the right length', speechfile_name, start, stop)

    return x


def load_soundfile(path, start='', stop=''):
    if (start != '' or stop != '' or not path.endswith('.wav')):
        x = nist_file_reader(path, start=start, stop=stop)
    else:
        fs, x = wavfile.read(path)
    return x


def read_random_sound_segment(speechfile_name, sample_length, random_generator):
    #sound, fs = sf.read(self.speechlist[i],start=j,stop=j+self.sample_len,dtype=np.int16)
    try:
        length = read_sample_count(speechfile_name)
    except:
        length = -1

    if length > sample_length:
        j = random_generator.randint(0, length - sample_length + 1)
        x = nist_file_reader(speechfile_name, start=j, stop=j + sample_length)
        x = x.astype(np.float64)
    else:
        x = nist_file_reader(speechfile_name)
        x = cut_signal(x, sample_length, random_generator)
    return x


def cut_signal(signal, newlength, random_generator):  # gen=np.random.RandomState()
    """
    cut or zeropad a signal to a new length
    """
    if len(signal) > newlength:
        j = random_generator.randint(0, len(signal) - newlength + 1)
        x = signal[j:j + newlength].astype(np.float64)
    else:
        j = random_generator.randint(0, newlength - len(signal) + 1)
        x = np.zeros(newlength, dtype=np.float64)
        x[j:j + len(signal)] = signal
    return x


def speech_add_noise(x, n, SNR_dB=None, vad_len=200, normalize_y=True):
    """VAD improvent?
    made for 8000Hz
    Keep the speech at the same signal power. Cheeks for empthy speech and noise.
    """
    SNR = 10**(SNR_dB / 10.)

    n -= n.mean()
    x -= x.mean()
    n_std = n.std()
    x_std = x.std()

    # Handle faulty data
    #raise ValueError(length,j,x.shape,self.speechlist[i])
    if (n_std == 0) or (x_std == 0):
        return None, None

    # simple vad to acount for silence segments
    vad = (x[:(len(x) // vad_len) * vad_len]**2).reshape(-1, vad_len).mean(axis=-1)
    x_std = vad[vad > (0.001 * x_std**2)].mean()**.5

    if normalize_y:
        # remove std, and rescale to required power st. E[std(y)]==1
        x_scale = (1 / x_std) * (SNR / (SNR + 1))**.5
        n_scale = (1 / n_std) * (1 / (SNR + 1))**.5
        x = x_scale * x
        y = x + n_scale * n
    else:
        n_scale = (x_std / n_std) * (1 / SNR)**.5
        y = x + n_scale * n
    return x, y


def mod_hann(M):
    """
    Generate Hann window that adds up to one with 50% overlap, when M is even.
    The standard Hann window adds up to one with 50% overlap, when M is odd.
    """
    intermediate = (M - 1) / 2
    win = .5 + .5 * np.cos(2 * np.pi * (np.arange(M) - intermediate) / M)
    return win


def get_SNR_dB(index):
    return (index % 7 - 1) * 3  # -3,0,...,15 dB


#==============================================================================
# Dataset
#==============================================================================


class Data_synthesis_1(Dataset):
    """
    First version of the data synthesis
    """

    def __init__(self, length, test=False, normalize_y=True, transform=None,
                 speech_list='switchbord', noise_version=1):

        if noise_version == 1:
            self.noisefilepath = os.path.join('data', 'processed', 'noise')
        elif noise_version == 2:
            self.noisefilepath = os.path.join('data', 'processed', 'noise2')
        else:
            raise ValueError(noise_version)

        self.fs = 8000
        self.sample_len = self.fs * 5
        self.length = length
        self.normalize_y = normalize_y
        self.transform = transform

#        self.noisefilenames, self.w = get_noise_w(self.noisefilepath)
        ((train_noisefilenames, train_w),
         (test_noisefilenames, test_w)) = get_noise_w(self.noisefilepath)

        if speech_list == 'switchbord':
            train_speechlist, test_speechlist = get_speech_list()
        elif speech_list == 'lre_train':
            train_speechlist, test_speechlist = get_speech_list(
                swb=False, fisher=False, lre_train=True)
        else:
            raise NotImplementedError('speech_list', speech_list)

        self.test = test
        if test:
            self.noisefilenames = test_noisefilenames
            self.w = test_w
            self.speechlist = test_speechlist
        else:
            self.noisefilenames = train_noisefilenames
            self.w = train_w
            self.speechlist = train_speechlist

#    @profile
    def __getitem__(self, index):
        # each sample id is the seed for the sample
        random_generator = np.random.RandomState(index)

        # Noise
        i = np.searchsorted(self.w, random_generator.rand())
        fs, noise = wavfile.read(os.path.join(self.noisefilepath, self.noisefilenames[i]))
        n = cut_signal(noise, self.sample_len, random_generator)

        # Speech
        i = random_generator.randint(0, len(self.speechlist))
        x = read_random_sound_segment(self.speechlist[i], self.sample_len, random_generator)

        # normalize mix the signals
#        SNR_db = random_generator.randint(-1,6)*3 #-3,0,...,15 dB
        SNR_dB = get_SNR_dB(index)

        x, y = speech_add_noise(x, n, SNR_dB=SNR_dB, normalize_y=self.normalize_y)

        if x is None:  # broken redo
            print('Redoing Sample ' + str(index), 'test=' + str(self.test))
            return self.__getitem__(index + self.length)

        if self.transform is not None:
            transformed_data = self.transform(x, y)
            return transformed_data  # data, targetY, targetX
        else:
            return x, y

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__doc__


if __name__ == '__main__':
    pass
