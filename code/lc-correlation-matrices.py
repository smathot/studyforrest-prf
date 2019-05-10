# coding=utf-8

import sys
import numpy as np
from datamatrix import (
    DataMatrix,
    io,
    NiftiColumn,
    FloatColumn,
    SeriesColumn,
    series as srs,
    operations as ops,
    convert as cnv,
    functional as fnc,
)
from nipy.modalities.fmri.hrf import spmt
from scipy import stats
import nibabel as nib
from nilearn import image
from scipy import signal
import warnings
import multiprocessing
from forrestprf import data
import statsmodels.formula.api as smf


def prf(t):
    
    t = t * 1000
    return t ** 10.1 * np.exp(-10.1 * t / 930)


RUNS = 1, 2, 3, 4, 5, 6, 7, 8
SUBJECTS = 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16 ,17, 18, 19, 20
PRF_XC = 80
PRF_YC = 64
CLEAN_IMG = True
N_PROCESS = 1 if '--single-process' in sys.argv else 6
NIFTI_SRC = 'inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-avmovie_run-{run}_bold.nii.gz'
MCPARAMS = 'inputs/studyforrest-data-aligned/sub-{sub:02}/in_bold3Tp2/sub-{sub:02}_task-avmovie_run-{run}_bold_mcparams.txt'
TRACE_SRC = 'inputs/pupil-traces/sub-{sub:02}/run-{run}.csv'
LC_ATLAS = 'inputs/mni-lc-atlas/lc-atlas-12.5.nii.gz'
FORNIX_ROI = 100, 100
LGN_ROI = 103, 104
BROCA_ROI = 13, 14
CONTROL_ROI = BROCA_ROI
SKIP_FIRST = 3  # First samples of the bold signal to ignore
X = np.linspace(0, 20, 10)
HRF = spmt(X)  # Hemodynamic response function
PRF = prf(X)  # Pupil response function
MULTIREGRESS = True
FORMULA_BOLD_PUPIL = 'bold ~ pupil + rot_1 + rot_2 + rot_3 + trans_x + trans_y + trans_z'
FORMULA_BOLD_AROUSAL = 'bold ~ arousal + rot_1 + rot_2 + rot_3 + trans_x + trans_y + trans_z'
FORMULA_PUPIL_AROUSAL = 'pupil ~ arousal'
FORMULA_TRACE = 'bold ~ trace + rot_1 + rot_2 + rot_3 + trans_x + trans_y + trans_z'
CONTROL = '--control' in sys.argv  # If set to True
SRC_EMOTION = 'inputs/ioats_2s_av_allchar.csv'
EMOTION_SEGMENTATION = [
    (0, 902),
    (886, 1768),
    (1752, 2628),
    (2612, 3588),
    (3572, 4496),
    (4480, 5358),
    (5342, 6426),
    (6410, 7086)
]


def flt(s):
    
    """Basic filtering that is applied to all signals"""
    
    return srs.filter_highpass(
        signal.detrend(s),
        len(s) / 64
    )


def deconv_hrf(s):
    
    """Deconvolve a signal with the canonical HRF"""
    
    return np.convolve(s, HRF)[:(len(s))]


def deconv_prf(s):
    
    """Deconvolve a signal with the pupil-response function from
    Hoeks and Levelt (1993)
    """

    return np.convolve(s, PRF)[:(len(s))]


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


def get_mcparams(sub, run):
    
    """Get motion parameters for one subject and one avmovie run"""
    
    a = np.loadtxt(MCPARAMS.format(sub=sub, run=run))
    dm = DataMatrix(length=a.shape[0], default_col_type=FloatColumn)
    dm.rot_1 = a[:, 0]
    dm.rot_2 = a[:, 1]
    dm.rot_3 = a[:, 2]
    dm.trans_x = a[:, 3]
    dm.trans_y = a[:, 4]
    dm.trans_z = a[:, 5]
    return dm


@fnc.memoize
def get_arousal_data(run):
    
    """Get arousal data for one avmovie run"""

    dm = io.readtxt(SRC_EMOTION)
    t0, t1 = EMOTION_SEGMENTATION[run - 1]
    a = list(dm.arousal[t0 // 2:t1 // 2])  # Align with 2s fMRI data
    if run == 8:  # One sample missing from the emotion timeseries in run 8!
        a.append(a[-1])
    return deconv_hrf(a)


@fnc.memoize
def get_pupil_data(sub, run):

    """Get a preprocessed pupil trace for one subject and one avmovie run """

    dm = io.readtxt(TRACE_SRC.format(sub=sub, run=run))
    dm = ops.auto_type(dm)
    dm.pupil_size @= lambda i: np.nan if i == 0 else i    
    dm.luminance @= lambda i: np.nan if i == 0 else i    
    dm.pupil_size = srs._interpolate(ops.z(dm.pupil_size))
    dm.luminance = srs._interpolate(ops.z(dm.luminance))
    i, s1, s2 = np.polyfit(dm.pupil_size, dm.luminance, deg=2)
    dm.pupil_size -= i + s1 * dm.luminance + s2 * dm.luminance ** 2
    return deconv_hrf(flt(dm.pupil_size))


def corr_img(img, trace, mcparams, xyz, deconv_bold):

    """Get a per-voxel correlation image between a nifti image and a trace"""

    pcor = np.empty(img.shape[:-1])
    pcor[:] = np.nan
    imgdat = img.get_data()
    for x, y, z in zip(*xyz):
        bold = flt(imgdat[x, y, z])
        if deconv_bold:
            bold = deconv_prf(bold)
        if MULTIREGRESS:
            dm = mcparams[SKIP_FIRST:len(trace)]
            dm.bold = FloatColumn
            dm.bold = bold[SKIP_FIRST:len(trace)]
            dm.trace = FloatColumn
            dm.trace = trace[SKIP_FIRST:]
            results = smf.ols(FORMULA_TRACE, dm).fit()
            pcor[x, y, z] = results.tvalues[1]
        else:
            s, i, r, p, se = nanregress(bold, pupil)
            pcor[x, y, z] = r
    return nib.Nifti2Image(pcor, img.affine)


def do_subject(sub):

    """Determine per-voxel and average correlations between:
    - VC <-> pupil size, luminance
    """

    if CONTROL:
        mask = data.juelich_mask(CONTROL_ROI)
    else:
        mask = image.load_img(LC_ATLAS)    
    xyz = np.where(mask.get_data() != 0)    
    rdm = DataMatrix(length=len(RUNS))
    rdm.t_vc_pupil = NiftiColumn
    rdm.t_vcavg_pupil = FloatColumn
    rdm.t_vc_arousal = NiftiColumn
    rdm.t_vcavg_arousal = FloatColumn
    rdm.t_pupil_arousal = FloatColumn
    rdm.sub = sub
    rdm.run = -1
    for row, run in zip(rdm, RUNS):
        print(
            'Starting sub: {}, run: {}, voxels: {}'.format(
                sub,
                run,
                len(xyz[0])
            )
        )
        img = get_avmovie_data(sub, run)        
        pupil_trace = get_pupil_data(sub, run)
        arousal_trace = get_arousal_data(run)
        mcparams = get_mcparams(sub, run)
        # Per-voxel correlations
        row.t_vc_pupil = corr_img(img, pupil_trace, mcparams, xyz, deconv_bold=True)
        row.t_vc_arousal = corr_img(img, arousal_trace, mcparams, xyz, deconv_bold=False)
        # Per-ROI correlations
        voxels = img.get_data()[xyz[0], xyz[1], xyz[2], :]
        avg = np.nanmean(voxels, axis=0)
        if MULTIREGRESS:            
            dm = mcparams[SKIP_FIRST:len(pupil_trace)]
            dm.bold = FloatColumn
            dm.bold = avg[SKIP_FIRST:len(pupil_trace)]
            dm.pupil = FloatColumn
            dm.pupil = pupil_trace[SKIP_FIRST:]
            dm.arousal = FloatColumn
            dm.arousal = arousal_trace[SKIP_FIRST:len(pupil_trace)]
            # Bold ~ pupil
            results = smf.ols(FORMULA_BOLD_PUPIL, data=dm).fit()     
            t = results.tvalues[1]
            print('t(bold ~ pupil) = {:.4f}'.format(t))
            row.t_vcavg_pupil = t
            # Bold ~ arousal
            results = smf.ols(FORMULA_BOLD_AROUSAL, data=dm).fit()     
            t = results.tvalues[1]
            print('t(bold ~ arousal) = {:.4f}'.format(t))
            row.t_vcavg_arousal = t
            # Pupil ~ arousal
            results = smf.ols(FORMULA_PUPIL_AROUSAL, data=dm).fit()     
            t = results.tvalues[1]
            print('t(pupil ~ arousal) = {:.4f}'.format(t))
            row.t_pupil_arousal = t
        else:
            s, i, r, p, se = nanregress(deconv_prf(flt(avg)), pupil_trace)
            row.t_vcavg_pupil = r
        row.run = run
    print('Done with sub: {}'.format(sub))
    return rdm


if __name__ == '__main__':

    if N_PROCESS == 1:
        results = map(do_subject, SUBJECTS)
    else:
        with multiprocessing.Pool(N_PROCESS) as pool:
            results = pool.map(do_subject, SUBJECTS)
    dm = DataMatrix()
    for rdm in results:
        dm <<= rdm
    if CONTROL:
        io.writepickle(dm, 'outputs/lc-correlation-matrix-control.pkl')
    else:
        io.writepickle(dm, 'outputs/lc-correlation-matrix.pkl')
