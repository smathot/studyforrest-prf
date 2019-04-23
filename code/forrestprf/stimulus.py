#!/usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np
import os
from datamatrix import functional as fnc


SAMPLES_PER_CYCLE = 16
CYCLES = 5
FRAMES_PER_SAMPLE = 50
STIM_WIDTH = 1280
STIM_HEIGHT = 1024
STIM_BG = 47
VIDEO_SRC = 'inputs/studyforrest-data-phase2/stimuli/retinotopic_mapping/{}.mkv'
# In case `code` is the working directory
if not os.path.isdir(os.path.dirname(VIDEO_SRC)):
    VIDEO_SRC = os.path.join('..', VIDEO_SRC)


def video_to_stim(src, downsample):

    print('Processing {}'.format(src))
    cap = cv2.VideoCapture(src)
    for sample in range(CYCLES * SAMPLES_PER_CYCLE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample * FRAMES_PER_SAMPLE)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[::downsample, ::downsample]
        frame[frame != STIM_BG] = 1
        frame[frame == STIM_BG] = 0
        if not sample:
            stim = np.empty((CYCLES * SAMPLES_PER_CYCLE, *frame.shape))
        stim[sample] = frame
    return stim


def blank_stim(samples, downsample):

    return np.zeros((
        samples,
        STIM_HEIGHT // downsample,
        STIM_WIDTH // downsample
    ))


@fnc.memoize(persistent=True)
def retinotopic_mapping_stim(downsample):

    return np.concatenate((
        blank_stim(2, downsample),
        video_to_stim(VIDEO_SRC.format('wedge_clock'), downsample),
        blank_stim(10, downsample),
        video_to_stim(VIDEO_SRC.format('wedge_counter'), downsample),
        blank_stim(10, downsample),
        video_to_stim(VIDEO_SRC.format('ring_contract'), downsample),
        blank_stim(10, downsample),
        video_to_stim(VIDEO_SRC.format('ring_expand'), downsample),
        blank_stim(8, downsample)
    ))


if __name__ == '__main__':

    for downsample in (64, 16, 4):
        print('Downsampling {}×'.format(downsample))
        retinotopic_mapping_stim(downsample)
