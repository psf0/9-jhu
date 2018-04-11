#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:39:11 2017

@author: peter
"""

"""
learn initialZATION h0,c0 layers?
EXperiment use augmentstyle data is it better?
ONline
test that auto tunes batch size?
test speed test compare to data generated gpu?
how much memory
dataset repr fixit to beautiful

CAUTION WORK: QUALITY over QUANTETY
"""

"""
tecnically the correct structure is to save x,n,y with their correct powers, but if it dosent matter
"""

#from pathlib import Path
#p = Path('.')
#[x for x in p.iterdir() if x.is_dir()]
# list(p.glob('**/*.py'))
#a = Path('.') / 'data' / 'processed' / 'dataset_1'
#os.path.join(*['data','processed', 'dataset_1'])
# a.exists()
# a.mkdir()
# a.anchor
#b = Path('.') / 'data' / 'processed' / 'dataset_1/../'
#c = b.resolve()
# c.parent,c.name
# Path.cwd()
# print("critical error", file=open('a.txt','a'))  # Python 3
# print(f'{batch:3} {epoch:3} / {total_epochs:3}  accuracy: {numpy.mean(accuracies):0.4f}±{numpy.std(accuracies):0.4f} time: {time / len(data_batch):3.2f}')


"""The volume does not matter nor quantization error"""
#from scipy.io import wavfile
#fs, x = wavfile.read('notebooks/lre17_aqgfiyro.wav')
#y = x//1.1
# wavfile.write('notebooks/lre17_aqgfiyro_2.wav',fs,y)

#==============================================================================
#
#==============================================================================

#    def _initialize_weights(self):
#        import torch.nn.init as init
#        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
#        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
#        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
#        init.orthogonal(self.conv4.weight)
"""
initial orthgonal
initial fc as inv mel, how to about the forward and backw seqence just add?
"""

"""
experiment sgd tuned vs adam(easy but sub optimal?)
batch size learnning rate...
"""

"""
pre emphasis and mel , log
tjeck it out in simlulate what is the differnce in the dataset
"""


#import soundfile as sf
#sound, fs = sf.read('/home/peter/AAU/9-jhu/data/raw/switchboard/sw_43471.sph',dtype=np.int16)


#@profile
# def train_online():
#    """
#    Online training
#    """
#    model.train()
#    epoch, epoch_loss = 1, 0
#    for i_batch, (data, targetY, targetX) in enumerate(training_data_loader,1):
#        data, targetY, targetX = Variable(data), Variable(targetY), Variable(targetX)
#        if cuda:
#            data = data.cuda(async=True)
#            targetY = targetY.cuda(async=True)
#            targetX = targetX.cuda(async=True)
#
#        if i_batch>6: #debug
#            print('debug')
#            break
#
#        optimizer.zero_grad()
#        mask = model(data) #prediction
#        loss = criterion(targetY*mask, targetX)
#        epoch_loss += loss.data[0]
#        loss.backward()
#        optimizer.step()
#        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, (i_batch-1)%epoch_length+1, epoch_length, loss.data[0]))
#
#        if (i_batch)//epoch_length + 1> epoch: # end of epoch
#            print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / epoch_length))
#            validation()
#            checkpoint_model_(epoch)
#
#
#            if epoch == epochs_n:
#                break
#            epoch, epoch_loss = epoch+1, 0

# train_online()


#==============================================================================
# Tests
#==============================================================================


# def unit_test_model(model,optimizer):
#    model = type(model)()
#    optimizer = type(optimizer)(model.parameters())
#    before = [p.sum() for p in model.parameters()]
#    for i_batch, (data, targetY, targetX) in enumerate(training_data_loader,1):
#        data, targetY, targetX = Variable(data), Variable(targetY), Variable(targetX)
#        break
#    data = data[0][None,:]
#    optimizer.zero_grad()
#    mask = model(data) #prediction
#    loss = criterion(mask, Variable(torch.randn(mask.size())))
#    loss.backward()
#    optimizer.step()
#    after = [p.sum() for p in model.parameters()]
#
#    # Make sure parameters changed
#    for b, a in zip(before, after):
#        assert b != a
#    assert loss.data[0] != 0
#    print('Tests Passed')
# unit_test_model(model,optimizer)

#==============================================================================
# train history
#==============================================================================


"""
averge error SNR epoch
vertical lines averge SNR
    of noisy
    of baselines (they are time domain?)
"""

""" Is the train error really only 2%?
np.mean((x-x*.98)**2)
"""
