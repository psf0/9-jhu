#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:19:51 2018

@author: peter
"""

import argparse
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from multiprocessing import Pool

from src.features.features_functions import (
    istft, walk_dir, Datafolder_soundfiles, save_audio_,
    SubsetSampler, make_identical_dir_structure, format_paths)
from src.models.model_functions import load_to_cpu


def estimate_mask(model, data, cuda=False):
    # works on a batch
    data = Variable(data, volatile=True)
    if cuda:
        data = data.cuda(async=True)
    mask = model(data)  # prediction
    mask = mask.data
    if cuda:
        mask = mask.cpu()
    return mask


def clean_sample_(model, mask, Y_m, Y_a, length, save_path):
    Xh_m = model.output_transform(mask, Y_m)
    y = model.inverse_transform(Xh_m, Y_a, new_length=length)
    return y


def enhance_Datafolder(model, input_dir, output_dir, batch_size, cuda=False):
    # Load dataset
    make_identical_dir_structure(input_dir, output_dir)
    output_paths = format_paths(test_set.y_paths, input_dir, output_dir, extention='.wav')
    y_paths = walk_dir(input_dir)
    enhance_soundfiles(model, y_paths, output_paths, batch_size, cuda)


def enhance_soundfiles(model, y_paths, output_paths, batch_size, cuda=False):
    if cuda:
        model.cuda()
    model.eval()

    test_set = Datafolder_soundfiles(y_paths=y_paths, transform=model.transform)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2,
                                  pin_memory=cuda)

    for (indexs, (data, Y_m, Y_a, length)) in test_data_loader:
            try:
                mask = estimate_mask(model, data, cuda=cuda)
            except:
                for index in indexs:
                    print(index, output_paths[index], 'failed')
                continue

            mask, Y_m, Y_a, length = (k.numpy() for k in (mask, Y_m, Y_a, length))

            for I, index in enumerate(indexs):
                y = clean_sample_(model, mask[I], Y_m[I], Y_a[I], length[I], output_paths[index])
                save_audio_(y, save_path)


                """
                compute the masks here and ....
                """



    print('Finished clearning', input_dir)


def clean_dataset_4_eval(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'y')
    output_dir = os.path.join('data', 'interim', 'dataset_4_val', model.experiment_name)
    enhance_Datafolder(model, input_dir, output_dir,
                       batch_size=10, cuda=cuda, samples=samples)


if __name__ == '__main__':
    # experiment_name = 'BLSTM_A5_27'
    # loadpath = os.path.join('models', 'BLSTM_A5',
    #                         'BLSTM_A5_epoch_27.state')
    # enhance_with_model(experiment_name, loadpath)

    experiment_name, loadpath, cuda=False, samples=None

    checkpoint = load_to_cpu(loadpath)
    p = checkpoint['p']
    model = p['model_class'](**p['model_kwargs'])
    model.load_state_dict(checkpoint['state_dict'])

    model.transform = p['input_transform']
    model.transform.target_transform = None
    model.output_transform = p['output_transform']
    model.inverse_transform = istft
    model.experiment_name = experiment_name  # this is a hack

    clean_dataset_4_eval(model, cuda=cuda, samples=samples)
