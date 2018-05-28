#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:50:15 2018

@author: peter
"""

from torch.utils.data import DataLoader
import os

from src.data.data_functions import Data_synthesis_1
from src.features.features_functions import makedir, save_audio_
from src.data.make_dataset_1 import save_preproccesing_parameters


def make_dataset_3(output_dir):
    """
    Dataset 1 consists of 6*5 hours of synthetic noisy speech. About the same length
    as the noise, however it has been sampled with It has been sampled with
    replacemnt. The Clean files are from switchboard and the noise is anotated
    noise only segments from the lre17_dev set. Each sample is 5 seconds long.
    SNR
        10 % is reserved for the validation set
    """

#    output_dir a = Path('.') / 'data' / 'processed' / 'dataset_1'

    train_len = int(6.6 * .9 * 3600 / 5 * 5)  # synthesise 5 times the train noise
    test_len = int(6.6 * .1 * 3600 / 5 * 5)  # synthesise 5 times the test noise

    train_set = Data_synthesis_1(length=train_len, speech_list='lre_train')
    training_data_loader = DataLoader(train_set, batch_size=1, num_workers=2)
    t_path_str_x = os.path.join(output_dir, 'train', 'x', 'sample_{}.wav')
    t_path_str_y = os.path.join(output_dir, 'train', 'y', 'sample_{}.wav')

    validation_set = Data_synthesis_1(length=test_len, test=True, speech_list='lre_train')
    validation_data_loader = DataLoader(validation_set, batch_size=1, num_workers=2)
    v_path_str_x = os.path.join(output_dir, 'val', 'x', 'sample_{}.wav')
    v_path_str_y = os.path.join(output_dir, 'val', 'y', 'sample_{}.wav')

    list_ = ((t_path_str_x, t_path_str_y, training_data_loader),
             (v_path_str_x, v_path_str_y, validation_data_loader)
             )

    for path_str_x, path_str_y, data_loader in list_:
        makedir(os.path.dirname(path_str_x))
        makedir(os.path.dirname(path_str_y))
        for i, (x, y) in enumerate(data_loader):
            x, y = x.numpy()[0], y.numpy()[0]
            save_audio_(x, path_str_x.format(i))
            save_audio_(y, path_str_y.format(i))


if __name__ == '__main__':
    dataset_dir = os.path.join(*['data', 'processed', 'dataset_3'])

    make_dataset_3(dataset_dir)
    save_preproccesing_parameters(dataset_dir)
