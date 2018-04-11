#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:19:51 2018

@author: peter
"""


import os

from src.features.features_functions import istft
from src.evaluation.eval_BLSTM_A0 import (
    clean_lre17_dev, clean_lre17_eval, clean_dataset_3_eval, clean_dataset_4_eval,
    clean_lre17tel_dev, clean_lre17tel_eval, clean_lre17tel_train)
from src.models.model_functions import apply_mask, load_to_cpu

if __name__ == '__main__':
    from src.models.BLSTM_A9 import p
    model = p['model_class'](**p['model_kwargs'])
    loadpath = os.path.join('models', 'BLSTM_A9',
                            'BLSTM_A9_epoch_56.state')

    model.load_state_dict(load_to_cpu(loadpath)['state_dict'])

    model.transform = p['input_transform']
    model.transform.mode = 'runtime'

    model.apply_mask = apply_mask
    model.inverse_transform = istft
    model.experiment_name = p['experiment_name']  # this is a hack

    samples = None  # slice(-1,None,-1)
    cuda = True
    clean_lre17_dev(model, cuda=cuda, samples=samples)
    clean_lre17_eval(model, cuda=cuda, samples=samples)
    clean_dataset_4_eval(model, cuda=cuda, samples=samples)
    clean_lre17tel_dev(model, cuda=cuda, samples=samples)
    clean_lre17tel_eval(model, cuda=cuda, samples=samples)
    # clean_lre17tel_train(model, cuda=cuda, samples=samples)
