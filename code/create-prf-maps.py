# coding=utf-8

from datamatrix import (
    functional as fnc,
    DataMatrix,
    operations as ops,
    io
)
from forrestprf import stimulus, data, fitting, plotting
import multiprocessing
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pickle


SUBJECTS = 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16 ,17, 18, 19, 20
N_PROCESS = 4
DST = 'outputs/prf-matrix.pkl'


def prf_map(sub_roi):

    sub, roi = sub_roi
    prf_map = fitting.prf_map(
        stimulus.retinotopic_mapping_stim(4),
        data.subject_data(sub),
        data.juelich_mask(data.ROI_JUELICH[roi]),
        npass=3
    )
    prf_map.sub = sub
    prf_map.roi = roi
    print(prf_map)
    return prf_map


def flatten(n, minval=-np.inf, maxval=np.inf):
    
    a = n.get_data().flatten()
    return a[~np.isnan(a) & (a >= minval) & (a <= maxval)]


if __name__ == '__main__':
    
    with multiprocessing.Pool(N_PROCESS) as pool:
        maps = pool.map(prf_map, itertools.product(SUBJECTS, data.ROI_JUELICH))
    dm = DataMatrix()
    for sdm in maps:
        dm <<= sdm
    dm.rf_size = dm.prf_sd @ (
        lambda img: np.nanmean(
            flatten(img, minval=1, maxval=60)
        )
    )
    io.writepickle(dm, DST)