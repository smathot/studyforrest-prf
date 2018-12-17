#!/usr/bin/env python3
# coding=utf-8

import os
import numpy as np
import scipy.ndimage.filters as fi
from nipy.modalities.fmri.hrf import spmt
from forrestprf import stimulus


PREDICTION_CACHE = 'inputs/bold-predictions/{w}-{h}/{x}-{y}-{sd}.npy'
HRF = spmt(np.linspace(0, 18, 10))
_prediction_cache = {}


def prf(x, y, sd, shape):

    X0 = np.arange(0, shape[2])
    Y0 = np.arange(0, shape[1])
    Xm, Ym = np.meshgrid(X0, Y0)
    gauss = np.exp(-(((Ym - y) ** 2) + ((Xm - x) ** 2)) / (2 * sd) ** 2)
    inp = np.zeros(shape)
    inp[:] = gauss
    return inp / inp.max()


def bold_prediction(x, y, sd, stim, hrf=True, norm=True):

    path = PREDICTION_CACHE.format(
        x=x,
        y=y,
        sd=sd,
        h=stim.shape[1],
        w=stim.shape[2]
    )
    if hrf and norm:
        if path in _prediction_cache:
            return _prediction_cache[path]
        if os.path.exists(path):
            pred = np.load(path)
            _prediction_cache[path] = pred
            return pred
    pred = (prf(x, y, sd, stim.shape) * stim).sum(axis=(1, 2))
    if hrf:
        pred = np.convolve(pred, HRF)[:pred.shape[0]]
    if norm:
        pred = pred - pred.min()
        pred /= pred.max()
    if hrf and norm:
        _prediction_cache[path] = pred
        np.save(path, pred)
    return pred


if __name__ == '__main__':

    from datamatrix import functional as fnc
    with fnc.profile():
        stim = stimulus.retinotopic_mapping_stim(4)
        prf(2, 2, 2, stim.shape)
        # bold_prediction(2, 2, 2, stim, hrf=False)
