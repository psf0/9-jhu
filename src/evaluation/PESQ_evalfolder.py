#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:38:30 2017

@author: peter
"""

import argparse
import numpy as np
import os
import subprocess
from multiprocessing.dummy import Pool  # threading
# import pandas as pd
from scipy import io
from functools import partial

from src.features.features_functions import makedir, walk_dir, format_paths

# 'NSLOTS'
THREADS = 4
PESQ_path = os.path.join('src', 'external', 'PESQ')


def PESQ(reference_path, degraded_path, fs):
    """Compute the PESQ score of a degraded signal with respect to a refrence..

    Parameters
    ----------
    reference_path : str
        The path to the refrence wavfile.
    degraded_path : str
        The path to the degraded wavfile.
    fs : int
        The sample frequency should be 8000 or 16000.

    Returns
    -------
    mos : float
        The mean opinion score. Returns None if PESQ failed.
    mos_lqo : float
        The mean opinion score rescaled with lqo. Returns None if PESQ failed.

    """
    if fs == 8000:
        PESQ_input = [PESQ_path, "+8000", reference_path, degraded_path]
    elif fs == 16000:
        PESQ_input = [PESQ_path, "+16000", '+wb', reference_path, degraded_path]
    else:
        raise ValueError('fs must be 8000 or 16000 Hz')

    try:
        c = subprocess.run(PESQ_input, stdout=subprocess.PIPE)
        mos, mos_lqo = c.stdout[-100:].decode().split('\n')[-2].split('=')[-1].split('\t')
        mos, mos_lqo = float(mos), float(mos_lqo)
        return mos, mos_lqo
    except:
        return None, None


def PESQ_evalpaths(reference_paths, degraded_paths, fs):
    """Compute the PESQ scores for all wavefiles in a list.

    Walks though a list of degraded wavefiles paths and computes all the scores
    with refrence wavefiles paths in the refrence list.

    Parameters
    ----------
    reference_path : str
        The path to the refrence wavfile.
    degraded_path : str
        The path to the degraded wavfile.
    fs : int
        The sample frequency should be 8000 or 16000.
    reference_dir : str
        Removes the refrence directory from the path to only save the filenames.

    Returns
    -------
    mos : 1-d float array
        The mean opinion score. Returns NAN if PESQ failed.
    mos_lqo : 1-d float array
        The mean opinion score rescaled with lqo. Returns NAN if PESQ failed.

    """
    PESQ_fs = partial(PESQ, fs=fs)

    p = Pool(THREADS)
    res = p.starmap(PESQ_fs, zip(reference_paths, degraded_paths))
    p.close()

    mos_list, mos_lqo_list = list(zip(*res))
    mos_list = np.array(mos_list, dtype=np.float)
    mos_lqo_list = np.array(mos_lqo_list, dtype=np.float)
    return mos_list, mos_lqo_list


def PESQ_evalfolder(reference_dir, degraded_dir, output_path, fs):
    """Compute the PESQ scores for all wavefiles in a folder.

    Walks though a directory of degraded wavefiles and computes all the scores
    with similar named refrence wavefiles in the refrence directory.
    Saves a csv file with the filenames, mos, and mos_lqo scores.

    Parameters
    ----------
    reference_dir : str
        The path to the refrence wavfile.
    degraded_dir : str
        The path to the degraded wavfile.
    fs : int
        The sample frequency should be 8000 or 16000.

    """
    reference_paths = walk_dir(reference_dir)
    degraded_paths = format_paths(reference_paths, reference_dir, degraded_dir)
    assert set(degraded_paths) == set(walk_dir(degraded_dir))
    print('Warning: PESQ will save a file in the current working directory')
    mos_list, mos_lqo_list = PESQ_evalpaths(reference_paths, degraded_paths,
                                            fs=fs)

    print('PESQ failed for {} samples'.format(np.isnan(mos_list).sum()))

    makedir(os.path.dirname(output_path))
    io.savemat(output_path,
               {'data_mos': mos_list,
                'data_mos_lqo': mos_lqo_list,
                'names': list(os.path.relpath(s, reference_dir) for s in reference_paths),
                'reference_dir': reference_dir,
                'degraded_dir': degraded_dir})

    # df = pd.DataFrame.from_dict(
    #     {'data_mos': mos_list,
    #      'data_mos_lqo': mos_lqo_list,
    #      'names': list(os.path.relpath(s, reference_dir) for s in reference_paths)})
    # df.index = df.loc[:, 'names']
    # df = df.drop('names', axis=1)
    # df.to_csv(output_path)


# ipython src/evaluation/PESQ_evalfolder.py 8000 data/processed/dataset_4/val/x data/processed/dataset_4/val/y models/dataset_4/PESQ_results.csv
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval PESQ on folder.')
    parser.add_argument('fs', type=int,
                        help='sample rate valid is 8000 or 16000)')
    parser.add_argument('reference_dir', type=str,
                        help='path to the reference directory')
    parser.add_argument('degraded_dir', type=str,
                        help='path to the degraded directory')
    parser.add_argument('output_path', type=str,
                        help='path to save the results(mat file)')

    args = parser.parse_args()

    PESQ_evalfolder(reference_dir=args.reference_dir,
                    degraded_dir=args.degraded_dir,
                    output_path=args.output_path,
                    fs=args.fs,
                    )

# reference_dir = os.path.join(*['data', 'processed', 'dataset_4', 'val', 'x'])
# degraded_dir = os.path.join(*['data', 'processed', 'dataset_4', 'val', 'y'])
# output_path = os.path.join(*['models', 'dataset_4', 'PESQ_results.mat'])
#
# PESQ_evalfolder(reference_dir=reference_dir,
#                 degraded_dir=degraded_dir,
#                 output_path=output_path
#                 )
