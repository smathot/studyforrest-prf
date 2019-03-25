# coding=utf-8

import itertools as it
import multiprocessing
import cv2
import sys
import numpy as np
from scipy import signal, misc
from datamatrix import (
    functional as fnc,
    operations as ops,
    DataMatrix, FloatColumn, IntColumn, io
)


SRC_VIDEO = 'inputs/videos/fg_av_ger_seg{}.mkv'
DST = 'outputs/mean-luminance/run-{}.png'
RUNS = 1, 2, 3, 4, 5, 6, 7, 8
MAX_FRAME = np.inf
DOWNSAMPLE = 8
N_PROCESSES = 4


def process_video(run, start_frame=1):

    cap = cv2.VideoCapture(SRC_VIDEO.format(run - 1))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = np.empty((
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // DOWNSAMPLE,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // DOWNSAMPLE
    ))
    for frame in it.count(start=start_frame):
        print(frame)
        ret, im = cap.read()
        if not ret or frame >= MAX_FRAME:
            print('Done!')
            break
        im = im.mean(axis=2)
        a[frame] = im[::DOWNSAMPLE, ::DOWNSAMPLE]
    misc.imsave(
        DST.format(run),
        a.mean(axis=0)
    )


if __name__ == '__main__':
    
    with multiprocessing.Pool(N_PROCESSES) as p:
        p.map(process_video, RUNS)
