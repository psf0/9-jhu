#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:38:30 2017

@author: peter
"""

#==============================================================================
# eval files script
#==============================================================================

import sys
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.signal import stft, istft
from scipy import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data_functions import get_SNR_dB

plt.close('all')

#==============================================================================
# Model training and fitting analysis
#==============================================================================


def load_train_history(path, title):
    with open(path, 'r') as f:
        lines = f.readlines()

    train_loss, train_loss_std, t, val_loss, val_loss_std = np.ones((5, 1000)) * np.nan

    for line in lines:
        if "{" in line:
            line = line.replace("{'", "'").replace(": ", ":").replace("}", "").replace("'", "").replace(",", "")

        line = line.split()
        i = int(line[1])
        if line[0] == 'train':
            train_loss[i] = float(line[2].replace('loss:', ''))
            train_loss_std[i] = float(line[3].replace('loss_std:', ''))
            t[i] = float(line[4].replace('t:', ''))
        elif line[0] == 'val':
            val_loss[i] = float(line[2].replace('loss:', ''))
            val_loss_std[i] = float(line[3].replace('loss_std:', ''))

    train_loss, train_loss_std, t, val_loss, val_loss_std = (
        I[1:i + 1] for I in (train_loss, train_loss_std, t, val_loss, val_loss_std))
    epochs = np.arange(1, i + 1)
    return {'train_loss': train_loss, 'train_loss_std': train_loss_std, 't': t,
            'val_loss': val_loss, 'val_loss_std': val_loss_std,
            'epochs': epochs, 'title': title}


def load_training_history(path, title):
    df = pd.DataFrame.from_csv(path)
    dict_ = {key.strip(' '): df[key].values for key in df.keys()}
    dict_['title'] = title
    dict_['epochs'] = df.index.values
    return dict_


BLSTM_A0_log = load_train_history(os.path.join('models', 'BLSTM_A0', 'log.txt'), 'BLSTM_A0')
BLSTM_A2_log = load_train_history(os.path.join('models', 'BLSTM_A2', 'log.txt'), 'BLSTM_A2')
BLSTM_A3_log = load_train_history(os.path.join('models', 'BLSTM_A3', 'log.txt'), 'BLSTM_A3')
BLSTM_A4_log = load_training_history(os.path.join('models', 'BLSTM_A4', 'training_log.txt'), 'BLSTM_A4')
BLSTM_A5_log = load_training_history(os.path.join('models', 'BLSTM_A5', 'training_log.txt'), 'BLSTM_A5')
BLSTM_A6_log = load_training_history(os.path.join('models', 'BLSTM_A6', 'training_log.txt'), 'BLSTM_A6')
BLSTM_A7_log = load_training_history(os.path.join('models', 'BLSTM_A7', 'training_log.txt'), 'BLSTM_A7')
# BLSTM_A8_log = load_training_history(os.path.join('models', 'BLSTM_A8', 'training_log.txt'), 'BLSTM_A8')
BLSTM_A9_log = load_training_history(os.path.join('models', 'BLSTM_A9', 'training_log.txt'), 'BLSTM_A9')
BLSTM_A10_log = load_training_history(os.path.join('models', 'BLSTM_A10', 'training_log.txt'), 'BLSTM_A10')
BLSTM_A11_log = load_training_history(os.path.join('models', 'BLSTM_A11', 'training_log.txt'), 'BLSTM_A11')
BLSTM_A12_log = load_training_history(os.path.join('models', 'BLSTM_A12', 'training_log.txt'), 'BLSTM_A12')
BLSTM_A13_log = load_training_history(os.path.join('models', 'BLSTM_A13', 'training_log.txt'), 'BLSTM_A13')
BLSTM_A14_log = load_training_history(os.path.join('models', 'BLSTM_A14', 'training_log.txt'), 'BLSTM_A14')

#plt.plot(epochs, train_loss, label='_'.join((title,'train')))
#plt.plot(epochs, val_loss, label='_'.join((title,'val')))
# plt.plot(epochs[np.argmin(val_loss)],val_loss[np.argmin(val_loss)],'ro')
# plt.legend()
# print(title,epochs[np.argmin(val_loss)],val_loss[np.argmin(val_loss)])

# epochs = BLSTM_A5_log['epochs']
# val_loss = BLSTM_A5_log['val_loss']


def plot_history(epochs, train_loss, val_loss, title, **kwags):
    plt.plot(epochs, train_loss, label='_'.join((title, 'train')))
    plt.plot(epochs, val_loss, label='_'.join((title, 'val')))
    plt.plot(epochs[np.argmin(val_loss)], val_loss[np.argmin(val_loss)], 'ro')
    plt.legend()
    print(title, epochs[np.argmin(val_loss)], val_loss[np.argmin(val_loss)])


plot_history(**BLSTM_A0_log)
plot_history(**BLSTM_A2_log)
plot_history(**BLSTM_A3_log)
plot_history(**BLSTM_A4_log)
plot_history(**BLSTM_A5_log)
plot_history(**BLSTM_A6_log)
plot_history(**BLSTM_A7_log)
plot_history(**BLSTM_A8_log)
plot_history(**BLSTM_A9_log)
plot_history(**BLSTM_A10_log)
plot_history(**BLSTM_A11_log)
plot_history(**BLSTM_A12_log)
plot_history(**BLSTM_A13_log)
plot_history(**BLSTM_A14_log)
plt.xlim(0, 57)
plt.savefig('sdf.pdf')
plt.show()



#==============================================================================
# Overfit BLSTM_A5 SE experiment
#==============================================================================

plot_history(**BLSTM_A5_log)
plt.xlim(0,30)
plt.ylim(None,0.0004)
plt.show()


A = pd.DataFrame.from_dict(BLSTM_A5_log)
A = A.set_index(A.epochs)
A = A.drop('epochs', axis=1)





"""
averge error SNR epoch
vertical lines averge SNR
    of noisy
    of baselines (they are time domain?)
"""

""" Is the train error really only 2%?
np.mean((x-x*.98)**2)
"""


#==============================================================================
# Analysis
#==============================================================================
"""
overfitting train histrory?
score train/test,  listen dev/eval

SNR effect train with higher/lower SNR

move lre17 outside to src so I can use del command

use LRE train as clean,
(Now it learns to preserve english and remove pieaces af other languishes
(rangly labled as pure noise))

let aswin try his system (general/open dataset, reverberations)

reverberation (100 ms reverberation, what does it sound like?)
add lre_train to lre_dev

Is the noise windowing broken? listen to noise files
nosy speech synthisation, SNR, samples with no speech
try 30 sec samples
"""


"""
OBSERVATIONS
works 4, 5, 6, 11

7 has some speech in noise, migth have learned to remove it to
1 has no sound

when OM_LSA result here I can conclude that
the systems enhances in the domain, but not on the eval data
cant handle reverberations (in eval data)

poor labeling means that there is sometimes speech in the noise

TODO
meeting with shinji, najim, jesus
mail zheng-hua
Look at the spectrograms, mask
Is reverberationa problem?
make matlab into one script
is SNR the problem? (look at pesq for high/low snr)
is the system overfitting? (train/val)
is languich the problem? (try generating at multi langush dataset)
some samples dont have any speech on it do we have vast laying around?
"""


"""
how namy are nan? try them again?
"""


def load_data(path_model_PESQ, path_model_eval, model_name):
    keys = ['data_mos', 'data_mos_lqo', 'data_STOI', 'data_eSTOI', 'data_SDR', 'names']
    model_dict = {}

    model_PESQ = io.matlab.loadmat(path_model_PESQ)
    names_i = [int(i.split('_')[-1].split('.')[0]) for i in model_PESQ['names']]
    names_argsort = np.argsort(names_i)
    for key in model_PESQ:
        if key in keys:
            model_dict[key] = model_PESQ[key].flatten()[names_argsort]

    model_eval = io.matlab.loadmat(path_model_eval)
    filenames_i = [int(i[0].split('_')[-1].split('.')[0]) for i in model_eval['filenames'][0]]
    filenames_argsort = np.argsort(filenames_i)
    for key in model_eval:
        if key in keys:
            model_dict[key] = model_eval[key].flatten()[filenames_argsort]

    model_dict['SNR_dB'] = get_SNR_dB(np.arange(len(names_argsort)))
    model_dict['model_name'] = model_name
    return model_dict


"""
dict with dataset 2 results
dict with dataset 3 results
dict with dataset 4 results
"""


dataset_2 = load_data(os.path.join('models', 'dataset_2', 'PESQ_results.mat'),
                      os.path.join('models', 'dataset_2', 'eval_results.mat'),
                      'dataset_2')
OM_LSA = load_data(os.path.join('models', 'OM_LSA', 'PESQ_results.mat'),
                   os.path.join('models', 'OM_LSA', 'eval_results.mat'),
                   'OM_LSA')
BLSTM_A0 = load_data(os.path.join('models', 'BLSTM_A0', 'PESQ_results.mat'),
                     os.path.join('models', 'BLSTM_A0', 'eval_results.mat'),
                     'BLSTM_A0')
BLSTM_A2 = load_data(os.path.join('models', 'BLSTM_A2', 'PESQ_results.mat'),
                     os.path.join('models', 'BLSTM_A2', 'eval_results.mat'),
                     'BLSTM_A2')
BLSTM_A2_7 = load_data(os.path.join('models', 'BLSTM_A2_7', 'PESQ_results.mat'),
                       os.path.join('models', 'BLSTM_A2_7', 'eval_results.mat'),
                       'BLSTM_A2_7')
BLSTM_A2_17 = load_data(os.path.join('models', 'BLSTM_A2_17', 'PESQ_results.mat'),
                        os.path.join('models', 'BLSTM_A2_17', 'eval_results.mat'),
                        'BLSTM_A2_17')
BLSTM_A3_dataset_2 = load_data(os.path.join('models', 'BLSTM_A3', 'PESQ_results2.mat'),
                               os.path.join('models', 'BLSTM_A3', 'eval_results2.mat'),
                               'BLSTM_A3')
result_dataset_2 = [dataset_2, OM_LSA, BLSTM_A0, BLSTM_A2_7, BLSTM_A2_17, BLSTM_A2, BLSTM_A3_dataset_2]

dataset_3 = load_data(os.path.join('models', 'dataset_3', 'PESQ_results.mat'),
                      os.path.join('models', 'dataset_3', 'eval_results.mat'),
                      'dataset_2')
BLSTM_A3 = load_data(os.path.join('models', 'BLSTM_A3', 'PESQ_results.mat'),
                     os.path.join('models', 'BLSTM_A3', 'eval_results.mat'),
                     'BLSTM_A3')
BLSTM_A4_3 = load_data(os.path.join('models', 'BLSTM_A4', 'PESQ_results_3.mat'),
                       os.path.join('models', 'BLSTM_A4', 'eval_results_3.mat'),
                       'BLSTM_A4')
result_dataset_3 = [dataset_3, BLSTM_A3, BLSTM_A4_3]


dataset_4 = load_data(os.path.join('models', 'dataset_4', 'PESQ_results.mat'),
                      os.path.join('models', 'dataset_4', 'eval_results.mat'),
                      'dataset_4')
OM_LSA_4 = load_data(os.path.join('models', 'OM_LSA', 'PESQ_results_4.mat'),
                     os.path.join('models', 'OM_LSA', 'eval_results_4.mat'),
                     'OM_LSA')
BLSTM_A4 = load_data(os.path.join('models', 'BLSTM_A4', 'PESQ_results.mat'),
                     os.path.join('models', 'BLSTM_A4', 'eval_results.mat'),
                     'BLSTM_A4')
BLSTM_A5 = load_data(os.path.join('models', 'BLSTM_A5', 'PESQ_results.mat'),
                     os.path.join('models', 'BLSTM_A5', 'eval_results.mat'),
                     'BLSTM_A5')
BLSTM_A9 = load_data(os.path.join('models', 'BLSTM_A9', 'PESQ_results.mat'),
                     os.path.join('models', 'BLSTM_A9', 'eval_results.mat'),
                     'BLSTM_A9')
BLSTM_A5_27 = load_data(os.path.join('models', 'BLSTM_A5_27', 'PESQ_results.mat'),
                        os.path.join('models', 'BLSTM_A5_27', 'eval_results.mat'),
                        'BLSTM_A5_27')
BLSTM_A9_24 = load_data(os.path.join('models', 'BLSTM_A9_24', 'PESQ_results.mat'),
                        os.path.join('models', 'BLSTM_A9_24', 'eval_results.mat'),
                        'BLSTM_A9_24')
BLSTM_A10 = load_data(os.path.join('models', 'BLSTM_A10', 'PESQ_results.mat'),
                      os.path.join('models', 'BLSTM_A10', 'eval_results.mat'),
                      'BLSTM_A10')
BLSTM_A11 = load_data(os.path.join('models', 'BLSTM_A11', 'PESQ_results.mat'),
                      os.path.join('models', 'BLSTM_A11', 'eval_results.mat'),
                      'BLSTM_A11')
result_dataset_4 = [dataset_4, OM_LSA_4, BLSTM_A4, BLSTM_A5, BLSTM_A9, BLSTM_A5_27, BLSTM_A9_24, BLSTM_A10, BLSTM_A11]
result_dataset_4 = [dataset_4, OM_LSA_4, BLSTM_A5_27, BLSTM_A9_24]
result_A5_overfit = [dataset_4, OM_LSA_4, BLSTM_A5_27]
epoch_numbers = [8, 14, 15, 16, 18, 19, 20, 24, 26, 28, 49, 68] + [13, 12, 17]
for epoch in epoch_numbers:
    experiment_name = 'BLSTM_A5_{}'.format(str(epoch))
    result_A5_overfit.append(
        load_data(os.path.join('models', experiment_name, 'PESQ_results.mat'),
                  os.path.join('models', experiment_name, 'eval_results.mat'),
                  experiment_name))



"""
propor handeling of NAN
"""
A = pd.concat([pd.DataFrame.from_dict(r) for r in result_A5_overfit])
A.SNR_dB = A.SNR_dB.astype("category")
A.model_name = A.model_name.astype("category")
A.names = A.names.astype("category")
A.columns = [s.replace('data_', '') for s in A.columns]
# A.names[A.loc[:, 'STOI'].isnull()].unique()
# A.model_name.cat.categories

models = [['dataset_4', 'dataset-4'],
          ['OM_LSA', 'OM-LSA'],
          ['BLSTM_A5_8', 'BLSTM-A5-8'],
          ['BLSTM_A5_12', 'BLSTM-A5-12'],
          ['BLSTM_A5_13', 'BLSTM-A5-13'],
          ['BLSTM_A5_14', 'BLSTM-A5-14'],
          ['BLSTM_A5_15', 'BLSTM-A5-15'],
          ['BLSTM_A5_16', 'BLSTM-A5-16'],
          ['BLSTM_A5_18', 'BLSTM-A5-18'],
          ['BLSTM_A5_19', 'BLSTM-A5-19'],
          ['BLSTM_A5_20', 'BLSTM-A5-20'],
          ['BLSTM_A5_24', 'BLSTM-A5-24'],
          ['BLSTM_A5_26', 'BLSTM-A5-26'],
          ['BLSTM_A5_27', 'BLSTM-A5-27'],
          ['BLSTM_A5_28', 'BLSTM-A5-28'],
          ['BLSTM_A5_49', 'BLSTM-A5-49'],
          ['BLSTM_A5_68', 'BLSTM-A5-68'],
          ]

A.model_name.cat.set_categories([s[0] for s in models], inplace=True)
A.model_name.cat.categories = [s[1] for s in models]

AA = A.groupby(['model_name']).mean().T
# Paper table

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
I = [int(s[1].split('-')[-1]) for s in models[2:]]
for v, c in zip(AA.iloc[:, 2:].values, colors):
    plt.plot(I, v, '-o', color=c)
    plt.show()
AA


def analyse():
    """see improvemnt in score of the cleaned files compared to the noisy files"""

def plot_smooth_dist(kernal):
    pass


def plot_histogram():
    pass


def get_bins(name):
    if 'mos' in name:
        bins = np.linspace(-1, 5, 30)  # PESQ
    elif 'STOI' in name:
        bins = np.linspace(-0.1, 1.1, 30)  # STOI
    else:
        bins = np.linspace(-15, 40, 30)  # SDR
    return bins


for name in ['data_mos', 'data_mos_lqo', 'data_STOI', 'data_eSTOI', 'data_SDR']:
    bins = get_bins(name)

    print()
    print('{:12}   mean std'.format(name))
    format_ = '{:12} {:6.2f} ±{:<6.2f}'
    for flag, model_dict in enumerate(result_dataset_4):
        #    SNRs = np.unique(ref_dict['SNR_dB'])
        I = model_dict['SNR_dB'] == 15
        I = np.zeros_like(I) == 0

        data = model_dict[name].T
        data = data[I]
        print(sum(np.isnan(data)))
        data = data[~np.isnan(data)]

        print(format_.format(model_dict['model_name'], np.mean(data), np.std(data)))

        # dist
        fig = plt.figure(name)
        ax = plt.gca()
        plt.title(name)
#        pd.DataFrame(data,columns=[model_dict['model_name']]).plot(kind='density',ax=ax) # or pd.Series()
        sns.distplot(data, hist=False, label=model_dict['model_name'])
plt.show()

# pd.DataFrame(data).plot(kind='density') # or pd.Series()
#sns.distplot(data, hist=False)


#==============================================================================
#
#==============================================================================


result_ = result_dataset_4
for name in ['data_mos', 'data_mos_lqo', 'data_STOI', 'data_eSTOI', 'data_SDR']:
    bins = get_bins(name)

    print()
    print('{:12}  mean  std'.format(name))
    format_ = '{:12} {:6.3f} ±{:<6.3f}'

    I = result_[0]['SNR_dB'] == -3
    # I = result_[0]['SNR_dB'] == 15
    # I = result_[0]['SNR_dB'] == 3
    # I = np.zeros_like(I) == 0
    for model_dict in result_:
        data = model_dict[name].T
        I = np.logical_and(I, ~np.isnan(data))
    # print(sum(~I))  # total failed
    # print(np.bincount(result_[0]['SNR_dB'][I]+3))  # count remaining samples in each snr
    for model_dict in result_:
        data = model_dict[name].T
        data = data[I]
        print(format_.format(model_dict['model_name'], np.mean(data), np.std(data)))


#==============================================================================
#
#==============================================================================


"""
plot 2 values make point plot, connect the 2 corresponding dots
"""



#==============================================================================
# OM-LSA error files
#==============================================================================

#
# from src.features.features_functions import walk_dir, format_paths
#
# p = walk_dir('data/processed/lre17tel_eval')
# # p2 = walk_dir('data/interim/lre17tel_eval/BLSTM_A5_27')
# # p3 = walk_dir('data/interim/lre17tel_eval/OM_LSA')
#
# p2 = format_paths(p, 'data/processed/lre17tel_eval', 'data/interim/lre17tel_eval/BLSTM_A5_27')
# p3 = format_paths(p, 'data/processed/lre17tel_eval', 'data/interim/lre17tel_eval/OM_LSA')
#
# a = np.zeros((len(p), 3), dtype=np.int)
# for i, I in enumerate(zip(p, p2, p3)):
#     a[i] = [os.path.getsize(x) for x in I]
# plt.hist(np.diff(a))
# plt.show()
# np.min(np.diff(a),axis=0)
# (a[np.diff(a)[:,1]<-62]/8000).astype(np.int)
# [p[i] for i in np.nonzero(np.diff(a)[:,1]<-62)[0]]
#
#
# p_dev = ['data/processed/lre17tel_dev/lre17_ajughykp.wav',
#          'data/processed/lre17tel_dev/lre17_kogewabk.wav',
#          'data/processed/lre17tel_dev/lre17_oxzlkapx.wav',
#          'data/processed/lre17tel_dev/lre17_qomottan.wav',
#          'data/processed/lre17tel_dev/lre17_rewnsmxj.wav',
#          'data/processed/lre17tel_dev/lre17_wilkzdqw.wav',
#          'data/processed/lre17tel_dev/lre17_wwiieshx.wav',
#          'data/processed/lre17tel_dev/lre17_zolwxdhw.wav']
#
#
# p_eval = ['data/processed/lre17tel_eval/lre17_ardrvoke.wav',
#           'data/processed/lre17tel_eval/lre17_dpsgwyrl.wav',
#           'data/processed/lre17tel_eval/lre17_defeioyx.wav',
#           'data/processed/lre17tel_eval/lre17_ecuqdhew.wav',
#           'data/processed/lre17tel_eval/lre17_fmlbruzc.wav',
#           'data/processed/lre17tel_eval/lre17_gdjuhesj.wav',
#           'data/processed/lre17tel_eval/lre17_gqbayebv.wav',
#           'data/processed/lre17tel_eval/lre17_hobemxpu.wav',
#           'data/processed/lre17tel_eval/lre17_hzaprdgw.wav',
#           'data/processed/lre17tel_eval/lre17_iazoghaf.wav',
#           'data/processed/lre17tel_eval/lre17_npawaapg.wav',
#           'data/processed/lre17tel_eval/lre17_oqcuqhap.wav',
#           'data/processed/lre17tel_eval/lre17_pbecqmik.wav',
#           'data/processed/lre17tel_eval/lre17_pfvxszla.wav',
#           'data/processed/lre17tel_eval/lre17_rglozgix.wav',
#           'data/processed/lre17tel_eval/lre17_rlkgkqbk.wav',
#           'data/processed/lre17tel_eval/lre17_secmcaqe.wav',
#           'data/processed/lre17tel_eval/lre17_swjanhva.wav',
#           'data/processed/lre17tel_eval/lre17_sxtuxgov.wav',
#           'data/processed/lre17tel_eval/lre17_tacgkiia.wav',
#           'data/processed/lre17tel_eval/lre17_tjnuqvui.wav',
#           'data/processed/lre17tel_eval/lre17_tqclgkyw.wav',
#           'data/processed/lre17tel_eval/lre17_ttzhcakz.wav',
#           'data/processed/lre17tel_eval/lre17_ukrpjvwu.wav',
#           'data/processed/lre17tel_eval/lre17_uucfdofr.wav',
#           'data/processed/lre17tel_eval/lre17_vafeeths.wav',
#           'data/processed/lre17tel_eval/lre17_vroqwckl.wav',
#           'data/processed/lre17tel_eval/lre17_vsgjxudr.wav',
#           'data/processed/lre17tel_eval/lre17_wjfkjgue.wav',
#           'data/processed/lre17tel_eval/lre17_wnsejyfd.wav',
#           'data/processed/lre17tel_eval/lre17_wrtrqjhg.wav',
#           'data/processed/lre17tel_eval/lre17_ybzvygkq.wav']
# p3_dev = format_paths(p_dev, 'data/processed/lre17tel_dev', 'data/interim/lre17tel_dev/OM_LSA')
# p3_eval = format_paths(p_eval, 'data/processed/lre17tel_eval', 'data/interim/lre17tel_eval/OM_LSA')
#
# from shutil import copyfile
# for src, dst in zip(p_dev, p3_dev):
#     copyfile(src, dst)
# for src, dst in zip(p_eval, p3_eval):
#     copyfile(src, dst)


#==============================================================================
# LRE evaluation
#==============================================================================
# Collect data on server
def load_lre_results():
    s1 = '/export/b16/pfredericks/AAU/9-jhu/lre17/v1.d/exp'
    s2 = '/export/b16/pfredericks/AAU/9-jhu/lre17/v1.d2/exp'
    res = []
    for s in (s1, s2):
        for (dirpath, dirnames, filenames) in os.walk(s):
            break
        models = [s for s in dirnames if s.startswith('results')]
        for model in models:
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(s, model)):
                if 'c_primary.txt' not in filenames:
                    continue
                with open(os.path.join(dirpath, 'c_primary.txt'), 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    model = model.replace('results_', '').replace('results', 'baseline')
                    backend, mode = dirpath.split(os.path.sep)[-2:]
                    result_name, result_value = line.split()
                    result_value = float(result_value)
                    res.append([model, backend, mode, result_name, result_value])

    # Create and format Pandas table
    df = pd.DataFrame(res, columns=['model', 'backend', 'mode', 'result_name', 'result_value'])
    df = df.drop_duplicates()

    df.model = df.model.astype("category")
    df.backend = df.backend.astype("category")
    df['mode'] = df['mode'].astype("category")
    df.result_name = df.result_name.astype("category")

    backend_names = [
        ['lnorm_lgbe_lre17tr', ''],
        ['lnorm_lgbe_lre17tr_cal_c1_10fold', 'GBE Non-adapt + Cal-CI'],
        ['lnorm_lgbe_lre17tr_cal_c2_10fold', 'GBE Non-adapt + Cal-CD'],
        ['lnorm_lgbe_lre17tr_devmap_rmu64_rw128_10fold', ''],
        ['lnorm_lgbe_lre17tr_devmap_rmu64_rw128_10fold_cal_c1_10fold', 'GBE Adapt-CI + Cal-CI'],
        ['lnorm_lgbe_lre17tr_devmap_rmu64_rw128_10fold_cal_c2_10fold', 'GBE Adapt-CI + Cal-CD'],
        ['lnorm_lgbe_lre17tr_devmap_mrmu64_mrw128_vrmu16_vrw64_10fold', ''],
        ['lnorm_lgbe_lre17tr_devmap_mrmu64_mrw128_vrmu16_vrw64_10fold_cal_c1_10fold', 'GBE Adapt-CD + Cal-CI'],
        ['lnorm_lgbe_lre17tr_devmap_mrmu64_mrw128_vrmu16_vrw64_10fold_cal_c2_10fold', 'GBE Adapt-CD + Cal-CD']]

    for key, name in backend_names:
        if not name:
            df = df.drop(df[df.backend == key].index)

    df.backend.cat.set_categories([s[0] for s in backend_names if s[1]], inplace=True)
    df.backend.cat.categories = [s[1] for s in backend_names if s[1]]
    return df


df = load_lre_results()
print(df.drop(['model'], axis=1).duplicated().sum())
df.model.cat.categories

# Paper table
models = [['baseline', 'Baseline'],
          ['OM_LSA', 'OM-LSA'],
          ['BLSMT_A5_8', 'BLSTM-A5-8'],
          ['BLSMT_A5_12', 'BLSTM-A5-12'],
          ['BLSMT_A5_13', 'BLSTM-A5-13'],
          ['BLSMT_A5_14', 'BLSTM-A5-14'],
          ['BLSMT_A5_15', 'BLSTM-A5-15'],
          ['BLSMT_A5_16', 'BLSTM-A5-16'],
          ['BLSMT_A5_18', 'BLSTM-A5-18'],
          ['BLSMT_A5_19', 'BLSTM-A5-19'],
          ['BLSMT_A5_20', 'BLSTM-A5-20'],
          ['BLSMT_A5_24', 'BLSTM-A5-24'],
          ['BLSMT_A5_26', 'BLSTM-A5-26'],
          ['BLSMT_A5_27', 'BLSTM-A5-27'],
          ['BLSMT_A5_28', 'BLSTM-A5-28'],
          ['BLSMT_A5_49', 'BLSTM-A5-49'],
          ['BLSMT_A5_68', 'BLSTM-A5-68'],
          ['BLSMT_A5', 'BLSTM-A5-97'],
          ]


def table(df, models):
    df = df.drop('mode', axis=1)
    df = df.drop('result_name', axis=1)
    df.model.cat.set_categories([s[0] for s in models], inplace=True)
    df.model.cat.categories = [s[1] for s in models]
    df = df.pivot(index='backend', columns='model', values='result_value')
    return df


a = df.query("(model in {}) & result_name == 'VAST_ACT_COST' & mode == 'eval'".format([s[0] for s in models]))
a = table(a, models)
a.style.format(lambda s: '{:.3f}'.format(s))
((1 - a.T / a.iloc[:, 0].values).T * 100).style.format(lambda s: '{:.1f}'.format(s))

m = df.query("(model in {}) & result_name == 'MLS14_ACT_COST' & mode == 'eval'".format([s[0] for s in models]))
m = table(m, models)
m.style.format(lambda s: '{:.3f}'.format(s))
(1 - m[-1:].values / m[-1:].values[0][0]) * 100


u = df.query("(model in {}) & result_name == 'EQ_ACT_COST' & mode == 'eval'".format([s[0] for s in models]))
u = table(u, models)
u.style.format(lambda s: '{:.3f}'.format(s))
(1 - u[-1:].values / u[-1:].values[0][0]) * 100


# print(a.to_latex())

# df.loc[:, 'model':'mode']
# df[(df['mode'] == 'eval') &
#    (df['result_name'] == 'VAST_ACT_COST') &
#    (df['model'].isin(['baseline', 'OM_LSA', 'BLSTM_A5']))]


#==============================================================================
# LRE evaluation plot VAST
#==============================================================================

d = df.query("(model in {}) & result_name == 'VAST_ACT_COST' & mode == 'dev'".format([s[0] for s in models]))
d = table(d, models)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
I = [s[1].split('-')[-1] for s in models[2:]]
for v, c in zip(a.iloc[2:, 2:].values, colors):
    plt.plot(I, v, '-o', color=c)
for v, c in zip(d.iloc[2:, 2:].values, colors):
    plt.plot(I, v, '--o', color=c)
plt.plot(BLSTM_A5_log['epochs'], BLSTM_A5_log['val_loss'] * 800, 'k-')
plt.plot(BLSTM_A5_log['epochs'], BLSTM_A5_log['train_loss'] * 800, 'k-')
plt.xlim(6, 30)
plt.ylim(.19, .31)
# plt.savefig('eval_over_epochs.pdf')
# plt.savefig('dev_over_epochs.pdf')
# plt.savefig('eval_dev_over_epochs.pdf')
# plt.ylim(.169, .20)
plt.show()


#==============================================================================
# LRE evaluation plot MLS14
#==============================================================================

d = df.query("(model in {}) & result_name == 'MLS14_ACT_COST' & mode == 'dev'".format([s[0] for s in models]))
d = table(d, models)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
I = [s[1].split('-')[-1] for s in models[2:]]
for v, c in zip(m.iloc[2:, 2:].values, colors):
    plt.plot(I, v, '-o', color=c)
for v, c in zip(d.iloc[2:, 2:].values, colors):
    plt.plot(I, v, '--o', color=c)
plt.plot(BLSTM_A5_log['epochs'], BLSTM_A5_log['val_loss'] * 550, 'k-')
plt.plot(BLSTM_A5_log['epochs'], BLSTM_A5_log['train_loss'] * 550, 'k-')
plt.xlim(6, 30)
plt.ylim(.15, .18)
# plt.savefig('eval_over_epochs.pdf')
# plt.savefig('dev_over_epochs.pdf')
# plt.savefig('eval_dev_over_epochs.pdf')
# plt.ylim(.169, .20)
plt.show()


#==============================================================================
# LRE evaluation plot UNEQ
#==============================================================================

d = df.query("(model in {}) & result_name == 'EQ_ACT_COST' & mode == 'dev'".format([s[0] for s in models]))
d = table(d, models)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
I = [s[1].split('-')[-1] for s in models[2:]]
for v, c in zip(a.iloc[2:, 2:].values, colors):
    plt.plot(I, v, '-o', color=c)
for v, c in zip(d.iloc[2:, 2:].values, colors):
    plt.plot(I, v, '--o', color=c)
plt.plot(BLSTM_A5_log['epochs'], BLSTM_A5_log['val_loss'] * 700, 'k-')
plt.plot(BLSTM_A5_log['epochs'], BLSTM_A5_log['train_loss'] * 700, 'k-')
plt.xlim(6, 30)
plt.ylim(.16, .27)
# plt.savefig('eval_over_epochs.pdf')
# plt.savefig('dev_over_epochs.pdf')
# plt.savefig('eval_dev_over_epochs.pdf')
# plt.ylim(.169, .20)
plt.show()
