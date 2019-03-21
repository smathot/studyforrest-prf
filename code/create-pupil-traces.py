# coding=utf-8

import itertools as it
import cv2
import sys
import numpy as np
from scipy import signal
from datamatrix import (
    functional as fnc,
    operations as ops,
    DataMatrix, FloatColumn, IntColumn, io
)


SRC_VIDEO = 'inputs/videos/fg_av_ger_seg{}.mkv'
SRC_EYEGAZE = 'inputs/studyforrest-data-phase2/sub-{subject:0>2}/ses-movie/func/sub-{subject:0>2}_ses-movie_task-movie_run-{run}_recording-eyegaze_physio.tsv.gz'
SUBJECTS = 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20
MAX_FRAME = np.inf


@fnc.memoize(persistent=True)
def get_subject_data(subject, run):

    src_eyegaze = SRC_EYEGAZE.format(subject=subject, run=run)
    print('Reading {}'.format(src_eyegaze))
    a = np.genfromtxt(src_eyegaze, delimiter='\t')
    dm = DataMatrix(length=np.max(a[:, 3]))
    dm.x = FloatColumn
    dm.y = FloatColumn
    dm.pupil = FloatColumn
    dm.frame = IntColumn
    for row, frame in zip(dm, np.unique(a[:, 3])):
        if frame >= MAX_FRAME:
            break
        i = np.where(a[:, 3] == frame)
        row.x = np.nanmedian(a[i, 0])
        row.y = np.nanmedian(a[i, 1])
        row.pupil = np.nanmedian(a[i, 2])
        row.frame = frame
    return dm


@fnc.memoize
def _get_subject_data(subject, run):

    return get_subject_data(subject, run)


def process_frame(run, frame, lm):

    dm = DataMatrix(length=len(SUBJECTS))
    dm.frame = frame
    dm.subject = IntColumn
    dm.x = FloatColumn
    dm.y = FloatColumn
    dm.pupil = FloatColumn
    dm.luminance = FloatColumn
    print('Run {}, frame {}'.format(run, frame))
    for row, sub in zip(dm, SUBJECTS):
        _dm = _get_subject_data(sub, run)
        _dm.pupil = ops.z(_dm.pupil)
        try:
            _row = (_dm.frame == frame)[0]
        except IndexError:
            continue
        row.x = min(1279, max(0, _row.x))
        row.y = min(546, max(0, _row.y))
        row.pupil = _row.pupil
        row.sub = sub
        row.pupil = _row.pupil
        row.luminance = lm[int(row.y), int(row.x)]
    return dm


def smoothing_kernel(size=30, px_per_deg=7):

    X0 = np.arange(-size//2, size//2)
    Y0 = np.arange(-size//2, size//2)
    Xm, Ym = np.meshgrid(X0, Y0)
    k = 33.2 + 10.6 * np.exp(
        -11.2 * (
            np.maximum(
                px_per_deg,
                ((Ym ** 2) + (Xm ** 2)) ** .5
            ) / px_per_deg
        )
    )
    inp = np.zeros((size, size), dtype=float)
    inp[:] = k
    return inp


def luminance_map(im, kernel):

    return signal.convolve2d(im.mean(axis=2), kernel)


def process_video(run, start_frame=1):

    kernel = smoothing_kernel()
    cap = cv2.VideoCapture(SRC_VIDEO.format(run - 1))
    dm = DataMatrix()
    for frame in it.count(start=start_frame):
        ret, im = cap.read()
        if not ret or frame >= MAX_FRAME:
            print('Done!')
            break
        dm <<= process_frame(run, frame, luminance_map(im, kernel))
    for sub, sdm in ops.split(dm.sub):
        io.writetxt(
            sdm,
            'outputs/traces/sub-{subject:0>2}/run-{}.csv'.format(sub, run)
        )


process_video(int(sys.argv[-1]))
