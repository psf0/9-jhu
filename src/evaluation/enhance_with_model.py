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


def compute_mask_batch(model, data, cuda=False):
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
    save_audio_(y, save_path)


def clean_Datafolder(input_dir, output_dir, model, batch_size, cuda=False, samples=None):
    # Load dataset
    make_identical_dir_structure(input_dir, output_dir)
    test_set = Datafolder_soundfiles(y_paths=walk_dir(input_dir), transform=model.transform)
    output_paths = format_paths(test_set.y_paths, input_dir, output_dir, extention='.wav')

    sampler = None
    if samples is not None:
        if type(slice(0)) == type(samples):
            sampler = SubsetSampler(indices=range(test_set.length), slice_=samples)
        else:
            sampler = SubsetSampler(indices=samples)

    test_data_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2,
                                  pin_memory=cuda, sampler=sampler)

    if cuda:
        model.cuda()
    model.eval()

    pool = Pool(processes=2)
    jobs = []
    for (indexs, (data, Y_m, Y_a, length)) in test_data_loader:
        try:
            mask = compute_mask_batch(model, data, cuda=cuda)
        except:
            for index in indexs:
                print(index, output_paths[index], 'failed')
        else:
            mask, Y_m, Y_a, length = (k.numpy() for k in (mask, Y_m, Y_a, length))

            for job in jobs:
                job.wait()
            jobs = []
            for I, index in enumerate(indexs):
                # clean_sample_(model, mask[I], Y_m[I], Y_a[I], length[I], output_paths[index])
                jobs.append(pool.apply_async(
                    clean_sample_(model, mask[I], Y_m[I], Y_a[I], length[I], output_paths[index])))
    pool.close()
    pool.join()
    print('Finished clearning', input_dir)


def clean_lre17_dev(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'lre17_dev')
    output_dir = os.path.join('data', 'interim', 'lre17_dev', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=1, cuda=cuda, samples=samples)


def clean_lre17_eval(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'lre17_eval')
    output_dir = os.path.join('data', 'interim', 'lre17_eval', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=1, cuda=cuda, samples=samples)


def clean_lre17tel_dev(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'lre17tel_dev')
    output_dir = os.path.join('data', 'interim', 'lre17tel_dev', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=1, cuda=cuda, samples=samples)


def clean_lre17tel_eval(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'lre17tel_eval')
    output_dir = os.path.join('data', 'interim', 'lre17tel_eval', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=1, cuda=cuda, samples=samples)


def clean_lre17tel_train(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'raw', 'lre17_train')
    output_dir = os.path.join('data', 'interim', 'lre17_train', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=1, cuda=cuda, samples=samples)


def clean_dataset_2_eval(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'y')
    output_dir = os.path.join('data', 'interim', 'dataset_2_val', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=10, cuda=cuda, samples=samples)


def clean_dataset_3_eval(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'y')
    output_dir = os.path.join('data', 'interim', 'dataset_3_val', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=10, cuda=cuda, samples=samples)


def clean_dataset_4_eval(model, cuda=False, samples=None):
    input_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'y')
    output_dir = os.path.join('data', 'interim', 'dataset_4_val', model.experiment_name)
    clean_Datafolder(input_dir, output_dir, model,
                     batch_size=10, cuda=cuda, samples=samples)


def enhance_with_model(experiment_name, loadpath, cuda=False, samples=None, dev_only=False):
    checkpoint = load_to_cpu(loadpath)
    p = checkpoint['p']
    model = p['model_class'](**p['model_kwargs'])
    model.load_state_dict(checkpoint['state_dict'])

    model.transform = p['input_transform']
    model.transform.target_transform = None
    model.output_transform = p['output_transform']
    model.inverse_transform = istft
    model.experiment_name = experiment_name  # this is a hack

    clean_lre17_dev(model, cuda=cuda, samples=samples)
    clean_lre17tel_dev(model, cuda=cuda, samples=samples)
    if not dev_only:
        clean_lre17_eval(model, cuda=cuda, samples=samples)
        clean_dataset_4_eval(model, cuda=cuda, samples=samples)
        clean_lre17tel_eval(model, cuda=cuda, samples=samples)
    # clean_lre17tel_train(model, cuda=cuda, samples=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval model datasets.')
    parser.add_argument('experiment_name', type=str,
                        help='The experiment name')
    parser.add_argument('loadpath', type=str,
                        help='the loadpath')
    # parser.add_argument('--cuda', type=bool,
    #                     help='Use cuda')
    # parser.add_argument('--dev_only', type=bool,
    #                     help='Only enhance the devset')
    args = parser.parse_args()

    enhance_with_model(experiment_name=args.reference_dir,
                       loadpath=args.degraded_dir,
                       cuda=args.output_path,
                       # dev_only=False
                       )

    # experiment_name = 'BLSTM_A5_27'
    # loadpath = os.path.join('models', 'BLSTM_A5',
    #                         'BLSTM_A5_epoch_27.state')
    # enhance_with_model(experiment_name, loadpath)
