# coding=utf-8

import nibabel as nib
from nilearn import image
import numpy as np
from datamatrix import io, series as srs
from scipy import signal
from scipy.stats import linregress


NIFTI_SRC = 'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-avmovie_run-{run}_bold.nii.gz'
PUPIL_SRC = '/home/sebastiaan/git/coeruleus/sub-{sub:02}/merged_timeseries_run-{run}.csv'


def avmovie_data(sub, run):

    return image.clean_img(NIFTI_SRC.format(sub=sub, run=run))


def pupil_data(sub, run):

    return signal.detrend(
        srs._interpolate(
            io.readtxt(
                PUPIL_SRC.format(sub=sub, run=run)
            ).pupil_size
        )
    )


def lc_data(sub, run):

    dm = io.readtxt(PUPIL_SRC.format(sub=sub, run=run))
    return signal.detrend(
        srs._interpolate(dm.LC_l_nonconf + dm.LC_r_nonconf)
    )


def corr_img(img, pupil, xyz, dt=0):

    pcor = np.empty(img.shape[:-1])
    pcor[:] = np.nan
    imgdat = img.get_data()
    for x, y, z in zip(*xyz):
        bold = imgdat[x, y, z, :len(pupil)]
        if dt > 0:
            s, i, r, p, se = linregress(bold[dt:], pupil[:-dt])
        elif dt < 0:
            s, i, r, p, se = linregress(bold[:dt], pupil[-dt:])
        else:
            s, i, r, p, se = linregress(bold, pupil)
        pcor[x, y, z] = r
    return nib.Nifti2Image(pcor, img.affine)
