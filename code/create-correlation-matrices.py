# coding=utf-8

import numpy as np
from datamatrix import DataMatrix, io, NiftiColumn, FloatColumn, series as srs
from scipy import stats
import nibabel as nib
from nilearn import image
from scipy import signal
from scipy.stats import linregress
import multiprocessing


DT = 2
RUNS = 1, 2, 3, 4, 5, 6, 7, 8
PRF_XC = 80
PRF_YC = 64
PRF_X_RANGE = 4, 156
PRF_Y_RANGE = 4, 124
PRF_SD_RANGE = 1, 60
N_PROCESS = 4
NIFTI_SRC = 'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-avmovie_run-{run}_bold.nii.gz'
TRACE_SRC = '/home/sebastiaan/git/coeruleus/sub-{sub:02}/merged_timeseries_run-{run}.csv'


def get_avmovie_data(sub, run):

    return image.clean_img(NIFTI_SRC.format(sub=sub, run=run))


def get_pupil_data(sub, run):

    return signal.detrend(
        srs._interpolate(
            io.readtxt(
                TRACE_SRC.format(sub=sub, run=run)
            ).pupil_size
        )
    )


def get_lc_data(sub, run):

    dm = io.readtxt(TRACE_SRC.format(sub=sub, run=run))
    return signal.detrend(
        srs._interpolate(dm.LC_l_nonconf + dm.LC_r_nonconf)
    )


def corr_img(img, pupil, xyz, dt=0):

    pcor = np.empty(img.shape[:-1])
    pcor[:] = np.nan
    imgdat = img.get_data()
    for x, y, z in zip(*xyz):
        bold = imgdat[x, y, z, :len(pupil)]
        if dt > 0:
            s, i, r, p, se = linregress(bold[dt:], pupil[:-dt])
        elif dt < 0:
            s, i, r, p, se = linregress(bold[:dt], pupil[-dt:])
        else:
            s, i, r, p, se = linregress(bold, pupil)
        pcor[x, y, z] = r
    return nib.Nifti2Image(pcor, img.affine)


def trim(a, minval=-np.inf, maxval=np.inf):

    """Set out of range values to nan"""

    a[(a < minval) | (a > maxval)] = np.nan


def flatten(nft):

    """Get a 1D array of non-nan values"""

    a = nft.get_data().flatten()
    return a[~np.isnan(a)]


def do_subject(args):

    """Determine per-voxel and average correlations between:
    - VC <-> pupil size, LC
    """

    sub, roi, xyz = args
    rdm = DataMatrix(length=len(RUNS))
    rdm.r_vc_pupil = NiftiColumn
    rdm.r_vc_lc = NiftiColumn
    rdm.r_lc_pupil = FloatColumn
    rdm.r_vcavg_pupil = FloatColumn
    rdm.r_vcavg_lc = FloatColumn
    rdm.sub = sub
    rdm.roi = roi
    for row, run in zip(rdm, RUNS):
        print('Starting sub: {}, roi: {}, run: {}'.format(sub, roi, run))
        img = get_avmovie_data(sub, run)
        pupil_trace = get_pupil_data(sub, run)
        lc_trace = get_lc_data(sub, run)
        # Per-voxel correlations
        row.r_vc_pupil = corr_img(img, pupil_trace, xyz, dt=DT)
        row.r_vc_lc = corr_img(img, lc_trace, xyz, dt=0)
        # Per-ROI correlations
        avg = np.nanmean(img.get_data(), axis=(0, 1, 2))
        s, i, r, p, se = stats.linregress(avg[DT:], pupil_trace[:-DT])
        row['r_vcavg_pupil'] = r
        s, i, r, p, se = stats.linregress(avg, lc_trace)
        row['r_vcavg_lc'] = r
        # Pupil-size LC correlation
        s, i, r, p, se = stats.linregress(lc_trace[DT:], pupil_trace[:-DT])
        row.r_lc_pupil = r
    print('Done with sub: {}, roi: {}'.format(sub, roi))
    return rdm


if __name__ == '__main__':

    # First read the data with PRF maps, and trim out voxels with poorly fitted
    # RFs.
    dm = io.readpickle('outputs/prf-matrix.pkl')
    for row in dm:
        trim(
            row.prf_x.get_data(),
            minval=PRF_X_RANGE[0],
            maxval=PRF_X_RANGE[1]
        )
        trim(
            row.prf_y.get_data(),
            minval=PRF_Y_RANGE[0],
            maxval=PRF_Y_RANGE[1]
        )
        trim(
            row.prf_sd.get_data(),
            minval=PRF_SD_RANGE[0],
            maxval=PRF_SD_RANGE[1]
        )
    # Use multiple processes for determining the correlations for performance
    with multiprocessing.Pool(N_PROCESS) as pool:
        results = pool.map(
            do_subject,
            [
                (row.sub, row.roi, np.where(~np.isnan(row.prf_x.get_data())))
                for row in dm
            ]
        )
    # Merge the correlation matrix and save it
    dm.r_vc_pupil = NiftiColumn
    dm.r_vc_lc = NiftiColumn
    dm.r_lc_pupil = FloatColumn
    for rdm in results:
        i = (dm.roi == rdm.roi[0]) & (dm.sub == rdm.sub[0])
        dm.r_vc_pupil[i] = rdm.r_vc_pupil.mean
        dm.r_vc_lc[i] = rdm.r_vc_lc.mean
        dm.r_lc_pupil[i] = rdm.r_lc_pupil.mean
    csvdm = dm[:]
    del csvdm.prf_err
    del csvdm.prf_sd
    del csvdm.prf_x
    del csvdm.prf_y
    del csvdm.r_vc_pupil
    del csvdm.r_vc_lc
    io.writetxt(csvdm, 'outputs/correlation-matrix.csv')
    io.writepickle(dm, 'outputs/correlation-matrix.pkl')
    # Convert the correlation matrix to a longish format with one voxel per row
    # and save it.
    ldm = DataMatrix()
    for row in dm:
        x = flatten(row.prf_x)
        y = flatten(row.prf_y)
        sd = flatten(row.prf_sd)
        err = flatten(row.prf_err)
        r = flatten(row.r_pupil)
        sdm = DataMatrix(len(x))
        sdm.sub = row.sub
        sdm.roi = row.roi
        sdm.r_pupil = FloatColumn
        sdm.prf_x = FloatColumn
        sdm.prf_y = FloatColumn
        sdm.prf_sd = FloatColumn
        sdm.r_pupil = r
        sdm.prf_x = x
        sdm.prf_y = y
        sdm.prf_sd = sd
        ldm <<= sdm
    ldm.ecc = ((ldm.prf_x - PRF_XC) ** 2 + (ldm.prf_y - PRF_YC)) ** .5
    io.writetxt(csvdm, 'outputs/longish-correlation-matrix.csv')
