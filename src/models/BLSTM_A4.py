#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:39:11 2017

@author: peter
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import time

from src.features.features_functions import (
    load_dataset_4_train, load_dataset_4_val, transform_tf, feature_transform_0,
    SubsetSampler, shuffle_cycle_dataset)
from src.features.mask_functions import ideal_amplitude_target
from src.models.model_functions import apply_mask, BLSTM_A, save_checkpoint


#==============================================================================
# Paremeters
#==============================================================================

# The model in the interspeech paper, smaller version dataset_4
p = {
    'experiment_name': 'BLSTM_A4',
    'model_class': BLSTM_A,
    'model_kwargs': {'input_size': 100,
                     'output_size': 129,
                     'hidden_size': 384,
                     'LSTM_layers': 2},
    'input_transform': transform_tf(
        feature_transform_0(os.path.join('data', 'processed', 'dataset_4')),
        ideal_amplitude_target),
    'output_transform': apply_mask,
    'training_set': load_dataset_4_train,
    'validation_set': load_dataset_4_val,
    'batch_size': 10,
    'epochs_max': 25,
    'samples_per_epoch': None,  # length of dataset_3 and 4
    'criterion': nn.MSELoss(),
    'optimizer': optim.Adam,
    'optimizer_kwargs': {},
    'cuda': True,
    'seed': 42}


if __name__ == '__main__':
    experiment_name = p['experiment_name']
    cuda = p['cuda']
    torch.manual_seed(p['seed'])
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('Building model')
    model = p['model_class'](**p['model_kwargs'])
    criterion = p['criterion']
    optimizer = p['optimizer'](model.parameters(), **p['optimizer_kwargs'])

    if cuda:
        torch.cuda.manual_seed(p['seed'])
        model = model.cuda()
        criterion = criterion.cuda()

    print('Loading datasets')
    training_set = p['training_set'](transform=p['input_transform'])
    validation_set = p['validation_set'](transform=p['input_transform'])
    output_transform = p['output_transform']

    batch_size = p['batch_size']
    samples_per_epoch = p['samples_per_epoch'] if p['samples_per_epoch'] is not None else training_set.length
    batch_per_epoch = int(np.ceil(samples_per_epoch / batch_size))
    print('One epoch is {:.1f} hours of data, the training set is {:.1f} hours \nand validations set is {:.1f} hours'.format(
        samples_per_epoch * 5 / 3600,
        training_set.length * 5 / 3600,
        validation_set.length * 5 / 3600,
    ))

    indecies_genrator = shuffle_cycle_dataset(
        training_set.length, samples_per_epoch, p['seed'])

    # check save path
    logpath = os.path.join('models', experiment_name, 'training_log.txt')
    save_str = os.path.join('models', experiment_name, experiment_name + '_epoch_{}.state')
    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    # log header
    print('epoch', 'train_loss', 'train_loss_std', 't', 'val_loss', 'val_loss_std',
          sep=',', file=open(logpath, 'a'))

#==============================================================================
# Training code
#==============================================================================


#@profile
def train():
    """
    training
    """
    model.train()
    epoch_loss, t0 = [], time.time()

    training_data_loader = DataLoader(
        training_set, batch_size=batch_size, num_workers=2, pin_memory=cuda,
        sampler=SubsetSampler(indices=sample_indecies))

    for i_batch, (indexs, (data, targetY, targetX)) in enumerate(training_data_loader, 1):
        data, targetY, targetX = Variable(data), Variable(targetY), Variable(targetX)
        if cuda:
            data = data.cuda(async=True)
            targetY = targetY.cuda(async=True)
            targetX = targetX.cuda(async=True)

        optimizer.zero_grad()
        mask = model(data)  # prediction
        loss = criterion(apply_mask(mask, targetY), targetX)
        epoch_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()
        print("===> Epoch {:2} {:4.1f}% Loss: {:.4e}".format(
            epoch, i_batch / batch_per_epoch * 100, loss.data[0]))

    # assume loss is emperical mean of the batch and i.i.d
    loss, loss_std, t = np.mean(epoch_loss), np.std(epoch_loss) * batch_size**.5, int(time.time() - t0)
    print("Epoch {} Complete: Avg. Loss: {:.4e} {:.4e} {}".format(epoch, loss, loss_std, int(t / 60)))
    print(epoch, loss, loss_std, t,
          sep=',', end=',', file=open(logpath, 'a'))


def validation():
    model.eval()
    val_loss = []

    validation_data_loader = DataLoader(
        validation_set, batch_size=batch_size, num_workers=2, pin_memory=cuda)

    for j_batch, (indexs, (data, targetY, targetX)) in enumerate(validation_data_loader, 1):
        data, targetY, targetX = Variable(data), Variable(targetY), Variable(targetX)

        if cuda:
            data = data.cuda(async=True)
            targetY = targetY.cuda(async=True)
            targetX = targetX.cuda(async=True)

        mask = model(data)  # prediction
        loss = criterion(output_transform(mask, targetY), targetX)
        val_loss.append(loss.data[0])

    # assume loss is emperical mean of the batch and i.i.d
    loss, loss_std = np.mean(val_loss), np.std(val_loss) * batch_size**.5
    print("Avg. Loss: {:.4e} {:.4e}".format(loss, loss_std))
    print(loss, loss_std,
          sep=',', file=open(logpath, 'a'))
    model.train()


if __name__ == '__main__':
    print('Training ' + experiment_name)

    for epoch in range(1, p['epochs_max'] + 1):
        sample_indecies = next(indecies_genrator)
        train()
        validation(),
        save_checkpoint(
            save_str.format(epoch),
            {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'rng_state': torch.get_rng_state(),
             'p': p
             })
