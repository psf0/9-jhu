#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:38:30 2017

@author: peter
"""

import numpy as np
import os
from scipy import io
import subprocess

from src.features.features_functions import makedir


# def makedir(dirpath):
#     if not os.path.exists(dirpath):
#         os.makedirs(dirpath)


def eval_PESQ(reference_dir, degraded_dir, output_path):
    makedir(os.path.dirname(output_path))
    print('Warning: PESQ will save a file in the current working directory')
    fs = "+8000"
    PESQ_path = os.path.join('src', 'external', 'PESQ')

    for (dirpath, dirnames, reference_files) in os.walk(reference_dir):
        break
    for (dirpath, dirnames, degraded_files) in os.walk(degraded_dir):
        break
    assert len(reference_files) == len(degraded_files)

    def try_to_run_PESQ(filename):
        reference = os.path.join(reference_dir, filename)
        degraded = os.path.join(degraded_dir, filename)

        PESQ = [PESQ_path, fs, reference, degraded]

        try:
            c = subprocess.run(PESQ, stdout=subprocess.PIPE)
            mos, mos_lqo = c.stdout[-100:].decode().split('\n')[-2].split('=')[-1].split('\t')
            mos, mos_lqo = float(mos), float(mos_lqo)
            return mos, mos_lqo
        except:
            return None, None

    mos_list, mos_lqo_list = np.zeros((2, len(reference_files)))

#    #single process
#    res = list(map(try_to_run_PESQ, reference_files))
#    mos_list, mos_lqo_list = list(zip(*res))

    from multiprocessing.dummy import Pool  # threading
    p = Pool(4)
    res = p.map(try_to_run_PESQ, reference_files)
    p.close()
    mos_list, mos_lqo_list = list(zip(*res))

    io.savemat(output_path,
               {'data_mos': np.array(mos_list, dtype=np.float),
                'data_mos_lqo': np.array(mos_lqo_list, dtype=np.float),
                'names': reference_files}
               )


def PESQ_dataset_1():
    reference_dir = os.path.join('data', 'processed', 'dataset_1', 'val', 'x')
    degraded_dir = os.path.join('data', 'processed', 'dataset_1', 'val', 'y')
    output_path = os.path.join('models', 'dataset_1', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'y')
    output_path = os.path.join('models', 'dataset_2', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_OM_LSA_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_2_val', 'OM_LSA')
    output_path = os.path.join('models', 'OM_LSA', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A0_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_2_val', 'BLSTM_A0')
    output_path = os.path.join('models', 'BLSTM_A0', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A2_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_2_val', 'BLSTM_A2')
    output_path = os.path.join('models', 'BLSTM_A2', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A2_7_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_2_val', 'BLSTM_A2_7')
    output_path = os.path.join('models', 'BLSTM_A2_7', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A2_17_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_2_val', 'BLSTM_A2_17')
    output_path = os.path.join('models', 'BLSTM_A2_17', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_dataset_3():
    reference_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'x')
    degraded_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'y')
    output_path = os.path.join('models', 'dataset_3', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A3_dataset_2():
    reference_dir = os.path.join('data', 'processed', 'dataset_2', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_2_val', 'BLSTM_A3')
    output_path = os.path.join('models', 'BLSTM_A3', 'PESQ_results2.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A3_dataset_3():
    reference_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_3_val', 'BLSTM_A3')
    output_path = os.path.join('models', 'BLSTM_A3', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A4_dataset_3():
    reference_dir = os.path.join('data', 'processed', 'dataset_3', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_3_val', 'BLSTM_A4')
    output_path = os.path.join('models', 'BLSTM_A4', 'PESQ_results_3.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'y')
    output_path = os.path.join('models', 'dataset_4', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_OM_LSA_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'OM_LSA')
    output_path = os.path.join('models', 'OM_LSA', 'PESQ_results_4.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A4_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A4')
    output_path = os.path.join('models', 'BLSTM_A4', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A5_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A5')
    output_path = os.path.join('models', 'BLSTM_A5', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A9_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A9')
    output_path = os.path.join('models', 'BLSTM_A9', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A5_27_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A5_27')
    output_path = os.path.join('models', 'BLSTM_A5_27', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A9_24_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A9_24')
    output_path = os.path.join('models', 'BLSTM_A9_24', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A10_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A10')
    output_path = os.path.join('models', 'BLSTM_A10', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_BLSTM_A11_dataset_4():
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', 'BLSTM_A11')
    output_path = os.path.join('models', 'BLSTM_A11', 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


def PESQ_model_dataset_4(experiment_name):
    reference_dir = os.path.join('data', 'processed', 'dataset_4', 'val', 'x')
    degraded_dir = os.path.join('data', 'interim', 'dataset_4_val', experiment_name)
    output_path = os.path.join('models', experiment_name, 'PESQ_results.mat')

    eval_PESQ(reference_dir=reference_dir,
              degraded_dir=degraded_dir,
              output_path=output_path
              )


if __name__ == '__main__':
    pass

    # PESQ_dataset_1()
    # PESQ_dataset_2()
    # PESQ_OM_LSA_dataset_2()
    # PESQ_BLSTM_A0_dataset_2()
    # PESQ_BLSTM_A2_dataset_2()
    # PESQ_BLSTM_A2_7_dataset_2()
    # PESQ_BLSTM_A2_17_dataset_2()

    # PESQ_dataset_3()
    # PESQ_BLSTM_A3_dataset_2()
    # PESQ_BLSTM_A3_dataset_3()
    # PESQ_BLSTM_A4_dataset_3()

    # PESQ_dataset_4()
    # PESQ_model_dataset_4('OM_LSA')
    # PESQ_model_dataset_4('BLSTM_A4')
    # PESQ_model_dataset_4('BLSTM_A5')
    # PESQ_model_dataset_4('BLSTM_A9')
    #
    # PESQ_model_dataset_4('BLSTM_A5_27')
    # PESQ_model_dataset_4('BLSTM_A9_24')
    # PESQ_model_dataset_4('BLSTM_A10')
    # PESQ_model_dataset_4('BLSTM_A11')
    #
    # epoch_numbers = [8, 14, 15, 16, 18, 19, 20, 24, 26, 28, 49, 68] + [13, 12, 17]
    # for epoch in epoch_numbers:
    #     PESQ_model_dataset_4('BLSTM_A5_{}'.format(str(epoch)))

#==============================================================================
#
#==============================================================================

    # multiple processes many processes
#    from joblib import Parallel, delayed
#    gen = (delayed(try_to_run_PESQ)(name) for name in reference_files)
#    res = Parallel(n_jobs=3)(gen)
#
#    from multiprocessing import Pool
