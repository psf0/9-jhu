#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:19:51 2018

@author: peter
"""

import torch
import os

from src.features.features_functions import transform_0, istft
from src.evaluation.eval_BLSTM_A0 import (
    clean_lre17_dev, clean_lre17_eval, clean_dataset_2_eval, clean_dataset_3_eval,
    clean_lre17tel_dev, clean_lre17tel_eval, clean_lre17tel_train)
from src.models.model_functions import apply_mask

if __name__ == '__main__':
    from src.models.BLSTM_A3 import BLSTM_A3
    model = BLSTM_A3()
    model.experiment_name = 'BLSTM_A3'
    loadpath = os.path.join('models', 'BLSTM_A3',
                            'BLSTM_A3_epoch_18.state')

    model.load_state_dict(torch.load(loadpath))

    dataset_dir = os.path.join('data', 'processed', 'dataset_3')
    model.transform = transform_0(mode='runtime', dataset_dir=dataset_dir)
    model.apply_mask = apply_mask
    model.inverse_transform = istft

    samples = None  # slice(-1,None,-1)
    cuda = True
    clean_lre17_dev(model, cuda=cuda, samples=samples)
    clean_lre17_eval(model, cuda=cuda, samples=samples)
    clean_dataset_2_eval(model, cuda=cuda, samples=samples)
    clean_dataset_3_eval(model, cuda=cuda, samples=samples)

    clean_lre17tel_dev(model, cuda=cuda, samples=samples)
    clean_lre17tel_eval(model, cuda=cuda, samples=samples)
    clean_lre17tel_train(model, cuda=cuda, samples=samples)
