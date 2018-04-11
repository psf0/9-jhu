#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:19:51 2018

@author: peter
"""


import os

from src.features.features_functions import istft
from src.evaluation.eval_BLSTM_A0 import (
    clean_lre17_dev, clean_lre17_eval, clean_dataset_4_eval,
    clean_lre17tel_dev, clean_lre17tel_eval)
from src.models.model_functions import apply_mask, load_to_cpu


def clean_overfit(experiment_name, loadpath):
    from src.models.BLSTM_A5 import p

    model = p['model_class'](**p['model_kwargs'])
    model.load_state_dict(load_to_cpu(loadpath)['state_dict'])

    model.experiment_name = experiment_name
    model.transform = p['input_transform']
    model.transform.mode = 'runtime'

    model.apply_mask = apply_mask
    model.inverse_transform = istft

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
    # epoch_numbers = [8, 19, 49, 68, 15, 24, 14, 16, 18, 20, 26, 28]
    # for epoch in epoch_numbers:
    #     clean_overfit('BLSTM_A5_' + str(epoch),
    #                   os.path.join('models', 'BLSTM_A5', 'BLSTM_A5_epoch_{}.state').format(str(epoch)))

    epoch_numbers = [13, 12, 17]
    for epoch in epoch_numbers:
        clean_overfit('BLSTM_A5_' + str(epoch),
                      os.path.join('models', 'BLSTM_A5', 'BLSTM_A5_epoch_{}.state').format(str(epoch)))
