# coding=utf-8

import numpy as np
from datamatrix import DataMatrix, io, NiftiColumn, FloatColumn, series as srs
from scipy import stats
import nibabel as nib
from nilearn import image
from scipy import signal
import warnings
import multiprocessing
from forrestprf import data


DT = 2
RUNS = 1, 2, 3, 4, 5, 6, 7, 8
PRF_XC = 80
PRF_YC = 64
CLEAN_IMG = True
N_PROCESS = 6
NIFTI_SRC = 'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-avmovie_run-{run}_bold.nii.gz'
TRACE_SRC = 'inputs/pupil-traces/sub-{sub:02}/run-{run}.csv'


def nanregress(x, y):
    
    """Performs linear regression after removing nan data"""
    
    i = np.where(~np.isnan(x) & ~np.isnan(y))
    return stats.linregress(x[i], y[i])


def get_avmovie_data(sub, run):

    """Get a clean nifti image for one subject and one avmovie run"""

    if CLEAN_IMG:
        return image.smooth_img(
            image.clean_img(NIFTI_SRC.format(sub=sub, run=run)),
            data.SMOOTHING
        )
    return image.load_img(NIFTI_SRC.format(sub=sub, run=run))


def get_pupil_data(sub, run):

    """Get a detrended pupil trace for one subject and one avmovie run """

    return signal.detrend(
        srs._interpolate(
            io.readtxt(
                TRACE_SRC.format(sub=sub, run=run)
            ).pupil_size
        )
    )


def get_luminance_data(sub, run):

    """Get a detrended luminance trace for one subject and one avmovie run """

    return signal.detrend(
        srs._interpolate(
            io.readtxt(
                TRACE_SRC.format(sub=sub, run=run)
            ).luminance
        )
    )


def corr_img(img, pupil, xyz, dt=0):

    """Get a per-voxel correlation image between a nifti image and a trace"""

    pcor = np.empty(img.shape[:-1])
    pcor[:] = np.nan
    imgdat = img.get_data()
    for x, y, z in zip(*xyz):
        bold = imgdat[x, y, z, :len(pupil)]
        if dt > 0:
            s, i, r, p, se = nanregress(bold[dt:], pupil[:-dt])
        elif dt < 0:
            s, i, r, p, se = nanregress(bold[:dt], pupil[-dt:])
        else:
            s, i, r, p, se = nanregress(bold, pupil)
        pcor[x, y, z] = r
    return nib.Nifti2Image(pcor, img.affine)


def trim(a, minval=-np.inf, maxval=np.inf):

    """Set out of range values to nan"""

    a[(a < minval) | (a > maxval)] = np.nan


def flatten(nft, mask=None):

    """Get a 1D array of non-nan values"""

    if nft is None:
        warnings.warn('cannot flatten an image that doesn\'t exist')
        return np.nan
    a = nft.get_data().flatten()
    if mask is None:
        return a[~np.isnan(a)]
    mask = mask.get_data().flatten()
    return a[~np.isnan(mask)]


def do_subject(args):

    """Determine per-voxel and average correlations between:
    - VC <-> pupil size, luminance
    """

    sub, roi, xyz = args
    rdm = DataMatrix(length=len(RUNS))
    rdm.r_vc_pupil = NiftiColumn
    rdm.r_vc_lum = NiftiColumn
    rdm.r_lum_pupil = FloatColumn
    rdm.r_vcavg_pupil = FloatColumn
    rdm.r_vcavg_lum = FloatColumn
    rdm.sub = sub
    rdm.roi = roi
    for row, run in zip(rdm, RUNS):
        print(
            'Starting sub: {}, roi: {}, run: {}, voxels: {}'.format(
                sub,
                roi,
                run,
                len(xyz[0])
            )
        )
        img = get_avmovie_data(sub, run)
        pupil_trace = get_pupil_data(sub, run)
        lum_trace = get_luminance_data(sub, run)
        # Per-voxel correlations
        row.r_vc_pupil = corr_img(img, pupil_trace, xyz, dt=DT)
        row.r_vc_lum = corr_img(img, lum_trace, xyz, dt=DT)
        # Per-ROI correlations
        voxels = img.get_data()[xyz[0], xyz[1], xyz[2], :]
        avg = np.nanmean(voxels, axis=0)[:len(pupil_trace)]
        s, i, r, p, se = nanregress(avg[DT:], pupil_trace[:-DT])
        row['r_vcavg_pupil'] = r
        s, i, r, p, se = nanregress(avg, lum_trace)
        row['r_vcavg_lum'] = r
        # Pupil-size luminance correlation
        s, i, r, p, se = nanregress(lum_trace, pupil_trace)
        row.r_lum_pupil = r
    print('Done with sub: {}, roi: {}'.format(sub, roi))
    return rdm


if __name__ == '__main__':

    # First read the data with PRF maps
    dm = io.readpickle('outputs/prf-matrix.pkl')
    # Use multiple processes for determining the correlations for performance
    args = [
        (row.sub, row.roi, np.where(~np.isnan(row.prf_x.get_data())))
        for row in dm
    ]
    if N_PROCESS == 1:
        results = map(do_subject, args)
    else:
        with multiprocessing.Pool(N_PROCESS) as pool:
            results = pool.map(do_subject, args)
    # Merge the correlation matrix and save it
    dm.r_vc_pupil = NiftiColumn
    dm.r_vc_lum = NiftiColumn
    dm.r_lum_pupil = FloatColumn
    dm.r_vcavg_pupil = FloatColumn
    dm.r_vcavg_lum = FloatColumn
    for rdm in results:
        i = (dm.roi == rdm.roi[0]) & (dm.sub == rdm.sub[0])
        dm.r_vc_pupil[i] = rdm.r_vc_pupil.mean
        dm.r_vc_lum[i] = rdm.r_vc_lum.mean
        dm.r_lum_pupil[i] = rdm.r_lum_pupil.mean
        dm.r_vcavg_pupil[i] = rdm.r_vcavg_pupil.mean
        dm.r_vcavg_lum[i] = rdm.r_vcavg_lum.mean
    csvdm = dm[:]
    del csvdm.prf_err
    del csvdm.prf_sd
    del csvdm.prf_x
    del csvdm.prf_y
    del csvdm.r_vc_pupil
    del csvdm.r_vc_lum
    io.writetxt(csvdm, 'outputs/correlation-matrix.csv')
    io.writepickle(dm, 'outputs/correlation-matrix.pkl')
    # Convert the correlation matrix to a longish format with one voxel per row
    # and save it.
    ldm = DataMatrix()
    for row in dm:
        x = flatten(row.prf_x)
        y = flatten(row.prf_y)
        sd = flatten(row.prf_sd)
        err = flatten(row.prf_err, mask=row.prf_x)
        r_vc_pupil = flatten(row.r_vc_pupil)
        r_vc_lum = flatten(row.r_vc_lum)
        sdm = DataMatrix(len(x))
        sdm.sub = row.sub
        sdm.roi = row.roi
        sdm.prf_x = FloatColumn
        sdm.prf_y = FloatColumn
        sdm.prf_sd = FloatColumn
        sdm.prf_err = FloatColumn
        sdm.r_vc_pupil = r_vc_pupil
        sdm.r_vc_lum = r_vc_lum
        sdm.prf_x = x
        sdm.prf_y = y
        sdm.prf_sd = sd
        sdm.prf_err = err
        ldm <<= sdm
    ldm.ecc = ((ldm.prf_x - PRF_XC) ** 2 + (ldm.prf_y - PRF_YC) ** 2) ** .5
    io.writetxt(ldm, 'outputs/longish-correlation-matrix.csv')
