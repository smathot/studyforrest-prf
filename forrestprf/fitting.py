#!/usr/bin/env python3
# coding=utf-8

import random
import itertools
import numpy as np
from forrestprf import predictions
from datamatrix import DataMatrix, NiftiColumn
import nibabel as nib


def prf_map(stim, data, mask, npass=4):

    if npass not in (1, 2, 3, 4):
        raise ValueError('npass should be 1, 2, 3, or 4')
    vox = data.get_data().copy()
    xyz = np.where(mask.get_data() != 0)
    print('PRF mapping (pass 1)')
    pass1 = _prf_map(
        stim[:, ::16, ::16],
        vox,
        xyz,
        downsample=2
    )
    if npass == 1:
        return _prf_dm(pass1, data)
    print('PRF mapping (pass 2)')
    pass2 = _prf_map(
        stim[:, ::4, ::4],
        vox, xyz,
        downsample=2,
        scale=4,
        est=pass1
    )
    if npass == 2:
        return _prf_dm(pass2, data)
    print('PRF mapping (pass 3)')
    xyz = np.where(~np.isnan(pass2[:, :, :, 0]))
    pass3 = _prf_map(
        stim[:, ::2, ::2],
        vox,
        xyz,
        est=pass2,
        scale=2
    )
    if npass == 3:
        return _prf_dm(pass3, data)
    print('PRF mapping (pass 4)')
    xyz = np.where(~np.isnan(pass3[:, :, :, 0]))
    pass4 = _prf_map(
        stim,
        vox,
        xyz,
        scale=2,
        est=pass3
    )
    return prf_dm


def _get_bold(vox, x, y, z, downsample):

    if downsample:
        if x % downsample or y % downsample or z % downsample:
            return None
        bold = np.nanmean(
            vox[
                x:x + downsample,
                y:y + downsample,
                z:z + downsample
            ],
            axis=(0, 1, 2)
        )
    else:
        bold = vox[x, y, z]
    bold -= bold.min()
    bold /= bold.max()
    return bold


def _scaled_search_space(est, x, y, z, scale):

    prfx, prfy, prfsd, prferr = est[x, y, z]
    return {
        'x': range(
            int(prfx * scale) - scale + 1,
            int(prfx * scale) + scale
        ),
        'y': range(
            int(prfy * scale) - scale + 1,
            int(prfy * scale) + scale
        ),
        'sd': range(
            max(1, int(prfsd * scale) - scale + 1),
            int(prfsd * scale) + scale
        ),
    }


def _full_search_space(stim):

    return {
        'y': range(stim.shape[1]),
        'x': range(stim.shape[2]),
        'sd': range(1, stim.shape[1] // 2),
    }


def _prf_map(stim, vox, xyz, downsample=None, scale=2, est=None):

    prf_map = np.empty(vox.shape[:-1] + (4,))
    prf_map[:] = np.nan
    for x, y, z in zip(*xyz):
        bold = _get_bold(vox, x, y, z, downsample)
        if bold is None:
            continue
        search_space = (
            _full_search_space(stim)
            if est is None
            else _scaled_search_space(est, x, y, z, scale)
        )
        params = _fit_prf(stim, bold, search_space)
        if downsample:
            prf_map[
                x:x + downsample,
                y:y + downsample,
                z:z + downsample
            ] = params
        else:
            prf_map[x, y, z] = params
    return prf_map


def _fit_prf(stim, bold, search_space):

    params = list(itertools.product(
        search_space['x'],
        search_space['y'],
        search_space['sd']
    ))
    random.shuffle(params)
    min_err = np.inf
    best_params = np.nan, np.nan, np.nan, np.inf
    for x, y, sd in params:
        err = ((predictions.bold_prediction(x, y, sd, stim) - bold) ** 2).sum()
        if err <= min_err:
            min_err = err
            best_params = x, y, sd, err
    return best_params


def _prf_dm(passx, data):

    prf_dm = DataMatrix(length=1)
    prf_dm.prf_x = NiftiColumn
    prf_dm.prf_y = NiftiColumn
    prf_dm.prf_sd = NiftiColumn
    prf_dm.prf_err = NiftiColumn
    prf_dm.prf_x = nib.Nifti2Image(passx[:, :, :, 0], affine=data.affine)
    prf_dm.prf_y = nib.Nifti2Image(passx[:, :, :, 1], affine=data.affine)
    prf_dm.prf_sd = nib.Nifti2Image(passx[:, :, :, 2], affine=data.affine)
    prf_dm.prf_err = nib.Nifti2Image(passx[:, :, :, 3], affine=data.affine)
    return prf_dm


if __name__ == '__main__':

    from forrestprf import data, stimulus
    print('Load stimulus4')
    stim4 = stimulus.retinotopic_mapping_stim(4)
    print(stim4.shape)
    print('Load subject data')
    bold = data.subject_data(1)
    print('Load subject mask')
    mask = data.mni_atlas(roi=data.ROI_OCCIPITAL)
    #mask.get_data()[:] = 0
    #mask.get_data()[34, 12, 22] = 1
    print('Map PRF')
    prf_dm = prf_map(stim4, bold, mask, npass=1)
    print(prf_dm)
