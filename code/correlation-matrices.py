# coding=utf-8

try:
    # Fix compatibility error
    from scipy.misc import factorial
except ImportError:
    from scipy import misc, special
    misc.factorial = special.factorial
else:
    from scipy import misc

import itertools
import numpy as np
from datamatrix import (
    DataMatrix,
    io,
    NiftiColumn,
    FloatColumn,
    series as srs,
    operations as ops,
    convert as cnv,
    functional as fnc,
)
from nipy.modalities.fmri.hrf import spmt
from scipy import stats
import nibabel as nib
from nilearn import image
import multiprocessing
from forrestprf import data
import statsmodels.formula.api as smf
import argparse


def prf(t):

    t = t * 1000
    return t ** 10.1 * np.exp(-10.1 * t / 930)


# Constants
ALL_RUNS = 1, 2, 3, 4, 5, 6, 7, 8
ALL_SUBJECTS = 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20
PRF_XC = 80
PRF_YC = 64
NIFTI_SRC = '../inputs/studyforrest-data-mni/sub-{sub:02}/sub-{sub:02}_task-avmovie_run-{run}_bold.nii.gz'
MCPARAMS = '../inputs/studyforrest-data-aligned/sub-{sub:02}/in_bold3Tp2/sub-{sub:02}_task-avmovie_run-{run}_bold_mcparams.txt'
TRACE_SRC = '../inputs/pupil-traces/sub-{sub:02}/run-{run}.csv'
LC_ATLAS = '../inputs/mni-lc-atlas/lc-atlas-12.5.nii.gz'
FULL_BRAIN = '../inputs/full-brain-mask.nii.gz'
DST = '../outputs/correlation-matrices/sub-{sub:02}_run-{run}.pkl'
FORNIX_ROI = 100, 100
LGN_ROI = 103, 104
BROCA_ROI = 13, 14
SKIP_FIRST = 3  # First samples of the bold signal to ignore
X = np.linspace(0, 20, 10)
HRF = spmt(X)  # Hemodynamic response function
PRF = prf(X)  # Pupil response function
MULTIREGRESS = True
FORMULA_BOLD_TRACE = 'bold ~ {} + rot_1 + rot_2 + rot_3 + trans_x + trans_y + trans_z'
FORMULA_TRACE_TRACE = '{} ~ {}'
FORMULA_VOXEL_TRACE = 'bold ~ trace + rot_1 + rot_2 + rot_3 + trans_x + trans_y + trans_z'
SRC_EMOTION = '../inputs/ioats_2s_av_allchar.csv'
TRACES = 'pupil', 'arousal', 'luminance', 'change'  # All the timeseries
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

    return srs.filter_highpass(s, len(s) / 64)


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

    src = NIFTI_SRC.format(sub=sub, run=run)
    print('Reading {}'.format(src))
    nft = image.smooth_img(image.clean_img(src), data.SMOOTHING)
    print('Done')
    return nft


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


def get_trace_data(sub, run, tracename):

    """A generic function to get a simple trace."""

    dm = io.readtxt(TRACE_SRC.format(sub=sub, run=run))
    dm = ops.auto_type(dm)
    dm[tracename] @= lambda i: np.nan if i == 0 else i
    dm[tracename] = srs._interpolate(ops.z(dm[tracename]))
    return deconv_hrf(flt(dm[tracename]))


def get_luminance_data(sub, run):

    """Gets a luminance trace for a given run and subject."""

    return get_trace_data(sub, run, 'luminance')


def get_change_data(sub, run):

    """Gets a visual-change trace for a given run and subject."""

    return get_trace_data(sub, run, 'change')


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
    if REMOVE_LUMINANCE_FROM_PUPIL:
        i, s1, s2 = np.polyfit(dm.pupil_size, dm.luminance, deg=2)
        dm.pupil_size -= i + s1 * dm.luminance + s2 * dm.luminance ** 2
    return deconv_hrf(flt(dm.pupil_size))


def corr_img(img, trace, mcparams, xyz, deconv_bold):

    """Get a per-voxel correlation image between a nifti image and a trace"""

    pcor = np.empty(img.shape[:-1])
    pcor[:] = np.nan
    imgdat = img.get_data()
    dm = mcparams[SKIP_FIRST:len(trace)]
    dm.bold = FloatColumn
    dm.trace = FloatColumn
    dm.trace = trace[SKIP_FIRST:]
    df = cnv.to_pandas(dm)
    for i, (x, y, z) in enumerate(zip(*xyz)):
        if DOWNSAMPLE and (x % 2 or y % 2 or z % 2):
            continue
        df.bold = flt(imgdat[x, y, z])[SKIP_FIRST:len(trace)]
        if deconv_bold:
            df.bold = deconv_prf(df.bold)
        results = smf.ols(FORMULA_VOXEL_TRACE, df).fit()
        pcor[x, y, z] = results.tvalues[1]
        if DOWNSAMPLE:
            pcor[x + 1, y + 1, z + 1] = results.tvalues[1]
    return nib.Nifti2Image(pcor, img.affine)


def do_run(sub, run, row, xyz):

    print(
        'Starting sub: {}, run: {}, voxels: {}'.format(
            sub,
            run,
            len(xyz[0])
        )
    )
    img = get_avmovie_data(sub, run)
    traces = {
        'pupil': get_pupil_data(sub, run),
        'arousal': get_arousal_data(run),  # doesn't depend on subject
        'luminance': get_luminance_data(sub, run),
        'change': get_change_data(sub, run)
    }
    mcparams = get_mcparams(sub, run)
    # Per-voxel correlationsx, y, z
    for tracename, trace in traces.items():
        row['t_bold_{}'.format(tracename)] = corr_img(
            img,
            trace,
            mcparams,
            xyz,
            deconv_bold=tracename == 'pupil'  # PRF deconvolution
        )
    # Per-ROI correlations
    voxels = img.get_data()[xyz[0], xyz[1], xyz[2], :]
    avg = np.nanmean(voxels, axis=0)
    # Create a DataFrame with equally long traces, also including the
    # average bold value
    trace_len = min(len(trace) for trace in traces.values())
    print('trace length: {}'.format(trace_len))
    dm = mcparams[SKIP_FIRST:trace_len]
    dm.bold = FloatColumn
    dm.bold = avg[SKIP_FIRST:trace_len]
    for tracename, trace in traces.items():
        dm[tracename] = FloatColumn
        dm[tracename] = trace[SKIP_FIRST:trace_len]
    df = cnv.to_pandas(dm)
    # Correlate each trace with the average bold signal
    for tracename in traces:
        results = smf.ols(
            FORMULA_BOLD_TRACE.format(tracename),
            data=df
        ).fit()
        t = results.tvalues[1]
        print('t(boldavg ~ {}) = {:.4f}'.format(tracename, t))
        row['t_boldavg_{}'.format(tracename)] = t
    # Correlate each trace with each other trace
    for tracename1, tracename2 in itertools.product(TRACES, TRACES):
        if tracename1 <= tracename2:
            continue
        results = smf.ols(
            FORMULA_TRACE_TRACE.format(tracename1, tracename2),
            data=df
        ).fit()
        t = results.tvalues[1]
        print('t({} ~ {}) = {:.4f}'.format(tracename1, tracename2, t))
        row['t_{}_{}'.format(tracename1, tracename2)] = t
    row.run = run


def do_subject(sub_runs):

    """Determine per-voxel and average correlations between:
    - VC <-> pupil size, luminance
    """

    sub, runs = sub_runs
    if isinstance(runs, int):
        runs = [runs]
    if VISUAL_CORTEX:
        mask = data.juelich_mask(data.ROI_JUELICH['VISUAL_CORTEX'])
    elif FULLBRAIN:
        mask = image.load_img(FULL_BRAIN)
    else:
        mask = image.load_img(LC_ATLAS)
    xyz = np.where(mask.get_data() != 0)
    rdm = DataMatrix(length=len(RUNS))
    # First create empty columns to be filled later.
    # Each series will be correlated with the bold signal in two ways:
    # - for individual voxels
    # - for the average bold activity
    for trace in TRACES:
        rdm['t_bold_{}'.format(trace)] = NiftiColumn
        rdm['t_boldavg_{}'.format(trace)] = FloatColumn
    # Each trace is also compared to each other trace (but not itself)
    for tracename1, tracename2 in itertools.product(TRACES, TRACES):
        if tracename1 <= tracename2:
            continue
        rdm['t_{}_{}'.format(tracename1, tracename2)] = FloatColumn
    rdm.sub = sub
    rdm.run = -1
    for row, run in zip(rdm, runs):
        do_run(sub, run, row, xyz)
        io.writepickle(rdm, DST.format(sub=sub, run=run))
        print('Done with sub: {}, run: {}'.format(sub, run))
    print('Done with sub: {}, runs: {}'.format(sub, runs))
    io.writepickle(rdm, DST.format(sub=sub, run=runs))
    return rdm


def parse_cmdargs():

    global RUNS
    global SUBJECTS
    global N_PROCESS
    global REMOVE_LUMINANCE_FROM_PUPIL
    global VISUAL_CORTEX
    global FULLBRAIN
    global DOWNSAMPLE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--roi',
        default='fullbrain',
        help='The region of interest to analyze',
    )
    parser.add_argument(
        '--n-process',
        default=1,
        type=int,
        help="The number of parallel processes to use",
    )
    parser.add_argument(
        '--remove-luminance-from-pupil',
        action='store_true',
        default=False,
        help="Determines whether fixation luminance should be regressed out "
             "of the pupil-size trace"
    )
    parser.add_argument(
        '--downsample',
        action='store_true',
        default=False,
        help="Determines whether the voxels should be downsampled, such that"
             "only 1 voxel in each 2 x 2 x 2 space is sampled"
    )
    parser.add_argument(
        '--runs',
        default='all',
        help="The video runs to process. Should be a list, e.g. [1, 2]",
    )
    parser.add_argument(
        '--subjects',
        default='all',
        help="The subjects to process. Should be a list, e.g. [1, 2]",
    )
    args = parser.parse_args()

    RUNS = eval(args.runs) if args.runs != 'all' else ALL_RUNS
    SUBJECTS = (
        eval(args.subjects) if args.subjects != 'all' else ALL_SUBJECTS
    )
    N_PROCESS = args.n_process
    REMOVE_LUMINANCE_FROM_PUPIL = args.remove_luminance_from_pupil
    VISUAL_CORTEX = args.roi == 'visual-cortex'
    FULLBRAIN = args.roi == 'fullbrain'
    DOWNSAMPLE = args.downsample


if __name__ == '__main__':

    parse_cmdargs()
    if VISUAL_CORTEX:
        print('Analyzing visual cortex')
    elif FULLBRAIN:
        print('Analyzing full brain')
    else:
        print('Analyzing LC')
    if N_PROCESS == 1:
        print('Using single process')
        results = map(do_subject, itertools.product(SUBJECTS, RUNS))
    else:
        print('Using {} processes'.format(N_PROCESS))
        with multiprocessing.Pool(N_PROCESS) as pool:
            results = pool.map(do_subject, itertools.product(SUBJECTS, RUNS))
    list(results)  # Consume the map
