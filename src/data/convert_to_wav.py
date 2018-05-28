#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:53:54 2017

@author: peter
"""

import os
from subprocess import call
from multiprocessing.dummy import Pool #threading
from functools import partial


#==============================================================================
# functions

def call_sox(input_output_path):
    input_path, output_path = input_output_path
    call(["sox", input_path, '-t', 'wav', '-r',
      '8000', '-b', '16', '-e',
      'signed-integer', output_path])

def soundfiles_to_wav(input_dir, output_dir, input_paths=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_paths is None:
        input_paths = []
        for (dirpath, dirnames, filenames) in os.walk(input_dir):
            input_paths.extend([os.path.join(dirpath, s) for s in filenames])

#    input_paths = (os.path.join(*[input_dir, input_name]) for input_name in input_names)
    output_names = (os.path.basename(s).rsplit('.')[0]+'.wav' for s in input_paths)
    output_paths = (os.path.join(*[output_dir, s]) for s in output_names)


#    call_sox_ = partial(call_sox, output_dir=output_dir)

    p = Pool(4)
    _ = p.map(call_sox, zip(input_paths,output_paths))
    p.close()

    # tjeck output
    for (dirpath, dirnames, output_names) in os.walk(output_dir):
        break
    print("input files# {} output files# {}".format(len(input_paths), len(output_names)))



#sox lre17_htwpxcbo.flac -t wav -r 8000 -b 16 -e signed-integer lre17_htwpxcbo.wav

#==============================================================================
# main

if __name__ == '__main__':
   # VAST data in lre17_dev
   input_dir = os.path.join(*['data','raw','lre17_dev'])
   output_dir = os.path.join(*['data','processed', 'lre17_dev'])
   for (dirpath, dirnames, filenames) in os.walk(input_dir):
           break
   input_names = [os.path.join(dirpath, s) for s in filenames if s.endswith('.flac')]
   soundfiles_to_wav(input_dir, output_dir, input_names)

   # VAST data in lre17_eval
   input_dir = os.path.join(*['data','raw','lre17_eval'])
   output_dir = os.path.join(*['data','processed', 'lre17_eval'])
   for (dirpath, dirnames, filenames) in os.walk(input_dir):
           break
   input_names = [os.path.join(dirpath, s) for s in filenames if s.endswith('.flac')]
   soundfiles_to_wav(input_dir, output_dir, input_names)

#    # Telephone data in lre17_dev
#    input_dir = os.path.join(*['data','raw','lre17_dev'])
#    output_dir = os.path.join(*['data','processed', 'lre17tel_dev'])
#    for (dirpath, dirnames, filenames) in os.walk(input_dir):
#            break
#    input_names = [os.path.join(dirpath, s) for s in filenames if s.endswith('.sph')]
#    soundfiles_to_wav(input_dir, output_dir, input_names)
#
#    # Telephone data in lre17_eval
#    input_dir = os.path.join(*['data','raw','lre17_eval'])
#    output_dir = os.path.join(*['data','processed', 'lre17tel_eval'])
#    for (dirpath, dirnames, filenames) in os.walk(input_dir):
#            break
#    input_names = [os.path.join(dirpath, s) for s in filenames if s.endswith('.sph')]
#    soundfiles_to_wav(input_dir, output_dir, input_names)




#==============================================================================
# old
#==============================================================================

##    # Telephone data in lre17_train
##    input_dir = os.path.join(*['data','raw','lre17_train'])
##    output_dir = os.path.join(*['data','processed', 'lre17_train'])
##    soundfiles_to_wav(input_dir, output_dir)
