#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:39:11 2017

@author: peter
"""

import numpy as np

#==============================================================================
# Mask functions
#==============================================================================


def ideal_binary_mask(X, Y):
    N = Y - X
    X_m, N_m = np.abs(X), np.abs(N)
    return X_m > N_m


def ideal_binary_target(X, Y):
    N = Y - X
    X_m, N_m, Y_m = np.abs(X), np.abs(N), np.abs(Y)
    Y_m[X_m < N_m] = 0
    return Y_m


def ideal_ratio_mask(X, Y):
    N = Y - X
    X_m, N_m = np.abs(X), np.abs(N)
    return X_m / (X_m + N_m)


def ideal_ratio_target(X, Y):
    N = Y - X
    X_m, N_m, Y_m = np.abs(X), np.abs(N), np.abs(Y)
    return X_m / (X_m + N_m) * Y_m


def ideal_Wiener_mask(X, Y):
    N = Y - X
    X_m, N_m = np.abs(X), np.abs(N)
    X_m2 = X_m**2
    return X_m2 / (X_m2 + N_m**2)


def ideal_Wiener_target(X, Y):
    N = Y - X
    X_m, N_m, Y_m = np.abs(X), np.abs(N), np.abs(Y)
    X_m2 = X_m**2
    return X_m2 / (X_m2 + N_m**2) * Y_m


def ideal_amplitude_mask(X, Y):
    X_m, Y_m = np.abs(X), np.abs(Y)
    return X_m / Y_m


def ideal_amplitude_target(X, Y):
    X_m = np.abs(X)
    return X_m


def ideal_phase_sensitive_mask(X, Y):
    X_m, Y_m = np.abs(X), np.abs(Y)
    c = np.cos(np.angle(X) - np.angle(Y))
    return X_m / Y_m * c


def ideal_phase_sensitive_target(X, Y):
    X_m = np.abs(X)
    c = np.cos(np.angle(X) - np.angle(Y))
    return X_m * c
