#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:19:51 2018

@author: peter
"""


import os

from src.features.features_functions import istft
from src.evaluation.enhance_with_model import (
    clean_lre17_dev, clean_lre17_eval, clean_dataset_4_eval,
    clean_lre17tel_dev, clean_lre17tel_eval)
from src.models.model_functions import apply_mask, load_to_cpu


def clean_overfit(experiment_name, loadpath):
    from src.models.BLSTM_A5 import p

    model = p['model_class'](**p['model_kwargs'])
    model.load_state_dict(load_to_cpu(loadpath)['state_dict'])

    model.transform = p['input_transform']
    model.transform.target_transform = None
    model.output_transform = p['output_transform']
    model.inverse_transform = istft
    model.experiment_name = experiment_name  # this is a hack

    samples = None  # slice(-1,None,-1)
    cuda = True
    if cuda:
        model.cuda()

    clean_lre17_dev(model, cuda=cuda, samples=samples)
    clean_lre17_eval(model, cuda=cuda, samples=samples)
    clean_lre17tel_dev(model, cuda=cuda, samples=samples)
    clean_lre17tel_eval(model, cuda=cuda, samples=samples)
    clean_dataset_4_eval(model, cuda=cuda, samples=samples)


if __name__ == '__main__':
    # epoch_numbers = [8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 26, 27, 28, 49, 68]
    # epoch_numbers = [5, 6, 7, 9, 10, 11, 21, 22, 23, 25, 29, 30]
    epoch_numbers = [1, 2, 3, 4, 31, 32, 33, 34, 35]
    for epoch in epoch_numbers:
        clean_overfit('BLSTM_A5_' + str(epoch),
                      os.path.join('models', 'BLSTM_A5', 'BLSTM_A5_epoch_{}.state').format(str(epoch)))
