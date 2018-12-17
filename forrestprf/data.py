#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import nibabel as nib
from nilearn import image
from datamatrix import functional as fnc


MNI_ATLAS = 'inputs/mni-structural-atlas/MNI/MNI-maxprob-thr50-2mm.nii.gz'
FORREST_BRAIN = 'inputs/studyforrest-data-templatetransforms/templates/grpbold3Tp2/brain.nii.gz'
NIFTI_SRC = [
    'inputs/studyforrest-data-mni/sub-{sub:02}_task-retmapclw_run-1_bold.nii.gz',
    'inputs/studyforrest-data-mni/sub-{sub:02}_task-retmapccw_run-1_bold.nii.gz',
    'inputs/studyforrest-data-mni/sub-{sub:02}_task-retmapcon_run-1_bold.nii.gz',
    'inputs/studyforrest-data-mni/sub-{sub:02}_task-retmapexp_run-1_bold.nii.gz',
]
ROI_OCCIPITAL = 5


@fnc.memoize(persistent=True)
def subject_data(sub):

    sessions = np.zeros(360)
    sessions[:90] = 1
    sessions[90:180] = 2
    sessions[180:270] = 3
    sessions[270:] = 4
    data = image.concat_imgs(src.format(sub=sub) for src in NIFTI_SRC)
    return image.clean_img(data, sessions=sessions)


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


if __name__ == '__main__':
    print('Reading subject 1')
    print(subject_data(1))
    print('Reading MNI atlas')
    print(mni_atlas(ROI_OCCIPITAL))