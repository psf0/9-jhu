#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:50:15 2018

@author: peter
"""

from torch.utils.data import DataLoader
import os

from src.data.data_functions import Data_synthesis_1, save_audio_
from src.features.features_functions import makedir
from src.datasets.make_dataset_1 import save_preproccesing_parameters


def make_dataset_5(output_dir):
    """
    Dataset 1 consists of 6*5 hours of synthetic noisy speech. About the same length
    as the noise, however it has been sampled with It has been sampled with
    replacemnt. The Clean files are from switchboard and the noise is anotated
    noise only segments from the lre17_dev set. Each sample is 5 seconds long.
    SNR
        10 % is reserved for the validation set
    """
    # int(119.2 *10**9 /2/8000/5)/int(6.6 * .9 * 3600 / 5 * 5)
    # output_dir a = Path('.') / 'data' / 'processed' / 'dataset_1'
    # lre17_train is 119.2 GB which is 119.2 *10**9 /2/8000 seconds
    # almost 70 times larger
    # train_len = int(6.6 * .9 * 3600 / 5 * 5)  # synthesise 1 times the train noise
    train_len = int(119.2 * 10**9 / 2 / 8000 / 5)  # synthesise equel to lre17_train
    # test_len = int(6.6 * .1 * 3600 / 5 * 5)  # synthesise 5 times the test noise

    train_set = Data_synthesis_1(
        length=train_len, speech_list='lre_train', noise_version=2)
    training_data_loader = DataLoader(train_set, batch_size=1, num_workers=2)
    t_path_str_x = os.path.join(output_dir, 'train', 'x', 'sample_{}.wav')
    t_path_str_y = os.path.join(output_dir, 'train', 'y', 'sample_{}.wav')

    list_ = ((t_path_str_x, t_path_str_y, training_data_loader),
             )

    for path_str_x, path_str_y, data_loader in list_:
        makedir(os.path.dirname(path_str_x))
        makedir(os.path.dirname(path_str_y))
        for i, (x, y) in enumerate(data_loader):
            x, y = x.numpy()[0], y.numpy()[0]
            save_audio_(x, path_str_x.format(i))
            save_audio_(y, path_str_y.format(i))


if __name__ == '__main__':
    dataset_dir = os.path.join(*['data', 'processed', 'dataset_5'])

    make_dataset_5(dataset_dir)
    save_preproccesing_parameters(dataset_dir)
