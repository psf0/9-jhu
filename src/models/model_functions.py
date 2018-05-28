#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:39:11 2017

@author: peter
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os

# checkpoint = load_to_cpu(os.path.join('models', 'BLSTM_A0', 'BLSTM_A0_epoch_100.state'))
# model.load_state_dict(checkpoint['state_dict'])

# save_checkpoint({
#     'epoch': epoch + 1,
#     'arch': model.__class__.__name__,
#     # 'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
# }, False,filename='asda')
#
# os.path.join('models', 'BLSTM_A0',
#                         'BLSTM_A0_epoch_100.state')


"""
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
With

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
Loading/Resuming from the dictionary is there

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            """


# Loading/Resuming from the dictionary is there
#
# if args.resume:
# if os.path.isfile(args.resume):
#     print("=> loading checkpoint '{}'".format(args.resume))
#     checkpoint = torch.load(args.resume)
#     args.start_epoch = checkpoint['epoch']
#     best_prec1 = checkpoint['best_prec1']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("=> loaded checkpoint '{}' (epoch {})"
#           .format(args.resume, checkpoint['epoch']))
# else:
#     print("=> no checkpoint found at '{}'".format(args.resume))


# torch.load(loadpath).keys()
# torch.load(os.path.join('models', 'BLSTM_A0','BLSTM_A0_epoch_35.state'),
#            map_location=lambda storage, loc: storage).keys()
# torch.get_num_threads()


#==============================================================================
# Support functions
#==============================================================================

"""
Change log just juse print append to files, one for training one for validation
change epoch, loop, define epoch as a number of samples,
    that can be consistent with previus data sizes
checkpoint_model_, save everything size?

https://github.com/pytorch/tnt might replace log or perhaps tensorboard
"""
"""mask functions"""


def log(logpath, mode, index, **kwargs):
    s = ' '.join([':'.join((str(i), str(v))) for i, v in kwargs.items()])
    with open(logpath, 'a') as file:
        string = ' '.join((str(mode), str(index), s, '\n'))
        file.write(string)


def save_checkpoint(filename, state):
    # 'checkpoint.pth.tar'
    torch.save(state, filename)
    # if is_best:
    # shutil.copyfile(filename, 'model_best.pth.tar')


def load_to_cpu(filename):
    return torch.load(filename, map_location=lambda storage, loc: storage)


def checkpoint_model_(savepath, model, save_as_cpu=True, is_best=False):
    is_cuda = next(model.parameters()).is_cuda

    if save_as_cpu and is_cuda:
        model.cpu()

    torch.save(model.state_dict(), savepath)

    if is_best:
        is_best_path = os.path.join(os.path.dirname(savepath), model.__class__.__name__ + "_best.state")
        torch.save(model.state_dict(), is_best_path)

    if save_as_cpu and is_cuda:
        model.cuda()

    print("Checkpoint saved to {}".format(savepath))


def apply_mask(mask, targetY):
    return targetY * mask

#==============================================================================
# Model Arhitectues
#==============================================================================


class BLSTM_A(nn.Module):
    """
    First Model
    """
    def __init__(self, input_size=100, output_size=129, hidden_size=384, LSTM_layers=2, bidirectional=True, output_act_f='Sigmoid'):
        super(type(self), self).__init__()
        self.hidden_size = hidden_size

        self.hidden_vectors = LSTM_layers * (2 if bidirectional else 1)
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=self.hidden_size,
                           num_layers=LSTM_layers,
                           bidirectional=bidirectional,
                           )
        self.fc1 = nn.Linear(self.hidden_size * (2 if bidirectional else 1), output_size)
        if output_act_f == 'Sigmoid':
            self.output_act_f = nn.Sigmoid()
        elif output_act_f == 'ReLU':
            self.output_act_f = nn.ReLU()
        else:
            1 / 0

    def init_hidden(self, x, batch_size):
        """
        Varible with the same datatype, cuda sensitive
        """
        shape = (self.hidden_vectors, batch_size, self.hidden_size)
        h0 = Variable(x.data.new(*shape).zero_())
        c0 = Variable(x.data.new(*shape).zero_())
        return h0, c0

    def forward(self, x):
        b, f, s = x.size()  # (batch, feature, seq)

        # BLSTM layers
        h0, c0 = self.init_hidden(x, b)

        x = x.permute(2, 0, 1)  # (seq, batch, feature)
        x, (h0, c0) = self.rnn(x, (h0, c0))  # (seq, batch, feature)

        # Sigmoid layer appied to the feature dimension over the sequence
        s, b, f = x.size()
        x = self.output_act_f(self.fc1(x.view((-1, f))))  # (batch*seq, feature)
        x = x.view(s, b, -1)  # (seq, batch, feature)

        x = x.permute(1, 2, 0)  # (batch, feature, seq)
        return x
#        print(6,x.size(),x.is_contiguous(),x.stride())
