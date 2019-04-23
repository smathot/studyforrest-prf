#!/usr/bin/env python3
# coding=utf-8

import os
import numpy as np
import nibabel as nib
from nilearn import image


MNI_ATLAS = 'inputs/mni-structural-atlas/MNI/MNI-maxprob-thr50-2mm.nii.gz'
JUELICH_ATLAS = 'inputs/juelich-histological-atlas/Juelich/Juelich-maxprob-thr50-2mm.nii.gz'
FORREST_BRAIN = 'inputs/studyforrest-data-templatetransforms/templates/grpbold3Tp2/brain.nii.gz'
NIFTI_SRC = [
    'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-retmapclw_run-1_bold.nii.gz',
    'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-retmapccw_run-1_bold.nii.gz',
    'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-retmapcon_run-1_bold.nii.gz',
    'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-retmapexp_run-1_bold.nii.gz',
]
SMOOTHING = 6
ROI_OCCIPITAL = 5
ROI_JUELICH = {
    'V1': (81, 82),
    'V2': (83, 84),
    'V3': (85, 86),
    'V4': (87, 88),
    'LGN': (103, 104)
}
# In case `code` is the working directory
if not os.path.isdir(os.path.dirname(MNI_ATLAS)):
    MNI_ATLAS = os.path.join('..', MNI_ATLAS)
    JUELICH_ATLAS = os.path.join('..', JUELICH_ATLAS)
    FORREST_BRAIN = os.path.join('..', FORREST_BRAIN)
    NIFTI_SRC = [
        os.path.join('..', path)
        for path in NIFTI_SRC
    ]


def subject_data(sub):

    sessions = np.zeros(360)
    sessions[:90] = 1
    sessions[90:180] = 2
    sessions[180:270] = 3
    sessions[270:] = 4
    return image.smooth_img(
        image.clean_img(
            image.concat_imgs(src.format(sub=sub) for src in NIFTI_SRC),
            sessions=sessions
        ),
        SMOOTHING
    )


def mni_atlas(roi):

    atlas = nib.load(MNI_ATLAS)
    a = atlas.get_data()
    a[a != roi] = 0
    mask = image.resample_to_img(
        atlas,
        nib.load(FORREST_BRAIN),
        interpolation='nearest'
    )
    return mask


def juelich_mask(roi):

    atlas = nib.load(JUELICH_ATLAS)
    a = atlas.get_data()
    a[(a != roi[0]) & (a != roi[1])] = 0
    mask = image.resample_to_img(
        atlas,
        nib.load(FORREST_BRAIN),
        interpolation='nearest'
    )
    return mask


if __name__ == '__main__':
    print('Reading subject 1')
    print(subject_data(1))
    print('Reading MNI atlas')
    print(mni_atlas(ROI_OCCIPITAL))
