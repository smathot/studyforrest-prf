# coding=utf-8

import itertools as it
import multiprocessing
import cv2
import numpy as np
from scipy import signal
from datamatrix import (
    functional as fnc,
    operations as ops,
    DataMatrix, FloatColumn, IntColumn, io
)


SRC_VIDEO = 'inputs/videos/fg_av_ger_seg{}.mkv'
SRC_EYEGAZE = 'inputs/studyforrest-data-phase2/sub-{subject:0>2}/ses-movie/func/sub-{subject:0>2}_ses-movie_task-movie_run-{run}_recording-eyegaze_physio.tsv.gz'
DST = 'inputs/luminance-traces/sub-{subject:0>2}/run-{run}.csv'
SUBJECTS = 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20
RUNS = 1, 2, 3, 4, 5, 6, 7, 8
MAX_FRAME = np.inf
N_PROCESSES = 4


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


def process_frame(run, frame, lm, cm):

    dm = DataMatrix(length=len(SUBJECTS))
    dm.frame = frame
    dm.sub = IntColumn
    dm.x = FloatColumn
    dm.y = FloatColumn
    dm.pupil = FloatColumn
    dm.luminance = FloatColumn
    dm.change = FloatColumn
    print('Run {}, frame {}'.format(run, frame))
    for row, sub in zip(dm, SUBJECTS):
        _dm = _get_subject_data(sub, run)
        _dm.pupil = ops.z(_dm.pupil)
        try:
            _row = (_dm.frame == frame)[0]
        except IndexError:
            continue
        row.sub = sub
        x = min(1279, max(0, _row.x))
        y = min(546, max(0, _row.y))
        if not x and not y:
            row.x = np.nan
            row.y = np.nan
            row.pupil = np.nan
            row.luminance = np.nan
            row.change = np.nan
        else:
            row.x = x
            row.y = y
            row.pupil = _row.pupil
            row.luminance = lm[int(y), int(x)]
            row.change = cm[int(y), int(x)]
    return dm


def smoothing_kernel(size=30, px_per_deg=7):

    """Based on Mathôt et al. (2015 JEP:Gen)"""

    X0 = np.arange(-size // 2, size // 2)
    Y0 = np.arange(-size // 2, size // 2)
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


def change_map(im1, im2, kernel):

    if im2 is None:
        return np.zeros(im1.shape[:-1])
    im1 = np.array(im1, dtype=np.int32)
    im2 = np.array(im2, dtype=np.int32)
    im_diff = np.abs(im1 - im2)
    cm = (
        signal.convolve2d(im_diff[:, :, 0], kernel) +
        signal.convolve2d(im_diff[:, :, 1], kernel) +
        signal.convolve2d(im_diff[:, :, 2], kernel)
    )
    return cm


def process_video(run, start_frame=1):

    kernel = smoothing_kernel()
    cap = cv2.VideoCapture(SRC_VIDEO.format(run - 1))
    dm = DataMatrix()
    im_prev = None
    for frame in it.count(start=start_frame):
        ret, im = cap.read()
        if not ret or frame >= MAX_FRAME:
            print('Done!')
            break
        dm <<= process_frame(
            run,
            frame,
            luminance_map(im, kernel),
            change_map(im, im_prev, kernel)
        )
        im_prev = im
    for sub, sdm in ops.split(dm.sub):
        io.writetxt(sdm, DST.format(subject=sub, run=run))


if __name__ == '__main__':

    process_video(1)
    with multiprocessing.Pool(N_PROCESSES) as p:
        p.map(process_video, RUNS)
