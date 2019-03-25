# coding=utf-8

import multiprocessing
import itertools
import numpy as np
import os
import sys
from datamatrix import series as srs, io, DataMatrix

SRC_EYEGAZE = 'inputs/studyforrest-data-phase2/sub-{subject:0>2}/ses-movie/func/sub-{subject:0>2}_ses-movie_task-movie_run-{run}_recording-eyegaze_physio.tsv.gz'
SRC_LUMINANCE = 'inputs/luminance-traces/sub-{subject:0>2}/run-{run}.csv'
DST = 'inputs/pupil-traces/sub-{subject:0>2}/run-{run}.csv'
DOWNSAMPLE = 2000
SUBJECTS = 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20
RUNS = 1, 2, 3, 4, 5, 6, 7, 8
N_PROCESSES = 4


def luminance_timeseries(subject, run, frames):

    dm = io.readtxt(SRC_LUMINANCE.format(subject=subject, run=run))
    dm = dm.sub == subject
    a = np.empty(frames.shape)
    a[:] = np.nan
    for i, row in enumerate(dm):
        a[frames == row.frame] = row.luminance
    return a


def merge_pupil(subject_run):

    subject, run = subject_run
    print(SRC_EYEGAZE.format(subject=subject, run=run))
    if '--clean' not in sys.argv and os.path.exists(
        DST.format(subject=subject, run=run)
    ):
        print('already done ...')
        return
    print('\treading ...')
    a = np.genfromtxt(
        SRC_EYEGAZE.format(subject=subject, run=run),
        delimiter='\t'
    )
    n_eyegaze = a.shape[0] // DOWNSAMPLE
    dm = DataMatrix(length=n_eyegaze)
    gazex = a[:, 0]
    gazey = a[:, 1]
    pupil = a[:, 2]
    print('\treconstructing blinks ...')
    pupil = srs.blinkreconstruct(pupil)
    pupil[pupil == 0] = np.nan
    print('\tgetting average luminance ...')
    luminance = luminance_timeseries(subject=subject, run=run, frames=a[:, 3])
    print('\tdownsampling ...')
    frame = a[:, 3]
    dm.pupil_size = srs.downsample(pupil, DOWNSAMPLE, fnc=np.nanmedian)
    dm.luminance = srs.downsample(luminance, DOWNSAMPLE, fnc=np.nanmedian)
    dm.sdgazex = srs.downsample(gazey, DOWNSAMPLE, fnc=np.nanstd)
    dm.sdgazey = srs.downsample(gazex, DOWNSAMPLE, fnc=np.nanstd)
    dm.start_frame = srs.downsample(frame, DOWNSAMPLE, fnc=np.nanmin)
    dm.end_frame = srs.downsample(frame, DOWNSAMPLE, fnc=np.nanmax)
    print('\twriting {}'.format(DST.format(subject=subject, run=run)))
    io.writetxt(dm, DST.format(subject=subject, run=run))


if __name__ == '__main__':

    with multiprocessing.Pool(N_PROCESSES) as p:
        p.map(merge_pupil, itertools.product(SUBJECTS, RUNS))
