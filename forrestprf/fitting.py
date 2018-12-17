#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import itertools
from forrestprf import predictions
import nibabel as nib


def fit_prf(stim, bold, search_space):

    min_err = np.inf
    for x, y, sd in itertools.product(
        search_space['x'],
        search_space['y'],
        search_space['sd']
    ):
        p = predictions.bold_prediction(x, y, sd, stim)
        err = ((p - bold) ** 2).sum()
        if err <= min_err:
            min_err = err
            best_params = x, y, sd, err
    return best_params


def prf_map(stim, data, mask):

    vox = data.get_data().copy()
    xyz = np.where(mask.get_data() != 0)
    print('pass1')
    pass1 = _prf_map(stim[:, ::16, ::16], vox, xyz, downsample=True)
    print('pass2')
    pass2 = _prf_map(stim[:, ::4, ::4], vox, xyz, downsample=True, est=pass1)
    print('pass3')
    pass3 = _prf_map(stim, vox, xyz, downsample=False, est=pass2)
    return pass3


def _prf_map(stim, vox, xyz, downsample=False, est=None):

    prf_map = np.empty(vox.shape[:-1] + (4,))
    prf_map[:] = np.nan
    for y, x, z in zip(*xyz):
        if y % 2 or x % 2 or z % 2:
            continue
        if downsample:
            bold = np.nanmean(vox[y:y + 2, x:x + 2, z:z + 2], axis=(0, 1, 2))
        else:
            bold = vox[y, x, z]
        bold -= bold.min()
        bold /= bold.max()
        if est is not None:
            # If an estimate is provided, then this comes from a PRF mapping
            # that was downsample 4 times relative to the current pass. We use
            # these estimates as a starting point, and limit our search to a
            # range of 4.
            prfx, prfy, prfsd, prferr = est[y, x, z]
            search_space = {
                'y': range(int(prfy * 4) - 2, int(prfy * 4) + 2),
                'x': range(int(prfx * 4) - 2, int(prfx * 4) + 2),
                'sd': range(max(1, int(prfsd) * 4 - 2), int(prfsd) * 4 + 2),
            }
        else:
            # If no estimate is provided, we search the entire stimulus space.
            search_space = {
                'y': range(stim.shape[1]),
                'x': range(stim.shape[2]),
                'sd': range(1, stim.shape[1] // 4),
            }
        params = fit_prf(stim, bold, search_space)
        if downsample:
            prf_map[y:y + 2, x:x + 2, z:z + 2] = params
        else:
            prf_map[y, x, z] = params
    return prf_map


if __name__ == '__main__':

    from forrestprf import data, stimulus
    print('Load stimulus4')
    stim4 = stimulus.retinotopic_mapping_stim(4)
    print('Load subject data')
    bold = data.subject_data(1)
    print('Load subject mask')
    mask = data.mni_atlas(roi=data.ROI_OCCIPITAL)
    print('Map PRF')
    print(prf_map(stim4, bold, mask))
