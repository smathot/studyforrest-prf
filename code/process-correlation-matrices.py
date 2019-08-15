# coding=utf-8

# <markdowncell>
"""
# The relationship between pupil size and activity in visual cortex

## Imports and constants
"""
# </markdowncell>


# <codecell>
import os
import numpy as np
import seaborn as sns
import itertools
from matplotlib import pyplot as plt
from nilearn import plotting as nip
from datamatrix import (
    NAN,
    io,
    DataMatrix,
    FloatColumn,
    IntColumn,
    MixedColumn,
    plot,
    operations as ops
)
from datamatrix.rbridge import lme4
from forrestprf import data
# </codecell>


# </markdowncell>

COR_SRC = '../outputs/visual-cortex-correlation-matrices/'
PRF_SRC = '../outputs/prf-matrix.pkl'
PRF_COLS = 'prf_err', 'prf_sd', 'prf_x', 'prf_y'
NSUB = 15
TRACES = 'change', 'luminance', 'pupil'
ROIS = 'V1', 'V2', 'V3', 'V4'
XCEN, YCEN = 80, 64
COLORS = {
    'V1': '#4DB6AC',
    'V2': '#009688',
    'V3': '#00796B',
    'V4': '#004D40',
}
# </codecell>


# <markdowncell>
"""
## Data parsing

The data is stored in separate DataMatrix objects, one for each participant,
where each row corresponds to one video fragment. We merge these objects such
that we get a big DataMatrix where each row corresponds to a single
participant, and the cells are averaged across video fragments.
"""
# </markdowncell>


# <codecell>
dm = DataMatrix(length=NSUB)
for row, basename in zip(dm, os.listdir(COR_SRC)):
    path = os.path.join(COR_SRC, basename)
    sdm = io.readpickle(path)
    for colname, col in sdm.columns:
        if colname not in dm:
            dm[colname] = type(col)
        row[colname] = col.mean
# </codecell>


# <markdowncell>
"""
For some analyses it's more convient to have the data in long format such that
each row corresponds to a single voxel. That's what we do here. We also merge
the PRF data into this long format, such that we know the PRF properties for
each voxel.
"""
# </markdowncell>


# <codecell>
prf_dm = io.readpickle(PRF_SRC)
ldm = DataMatrix()
ldm.sub = IntColumn
ldm.roi = MixedColumn
for prf_col in PRF_COLS:
    ldm[prf_col] = FloatColumn
for trace in TRACES:
    ldm['t_bold_{}'.format(trace)] = FloatColumn
for row in dm:
    for roi in ROIS:
        mask = data.juelich_mask(data.ROI_JUELICH[roi])
        xyz = mask.get_data() != 0
        for j, trace in enumerate(TRACES):
            a = row['t_bold_{}'.format(trace)].get_data()
            a = a[xyz]
            if not j:  # Only the first trace of each roi
                ldm.length += len(a)
                ldm.roi[-len(a):] = roi
                ldm.sub[-len(a):] = row.sub
            ldm['t_bold_{}'.format(trace)][-len(a):] = a
        # Merge prf data
        prf_row = ((prf_dm.sub == row.sub) & (prf_dm.roi == roi))[0]
        for prf_col in PRF_COLS:
            ldm[prf_col][-len(a):] = prf_row[prf_col].get_data()[xyz]
ldm = ldm.prf_err != NAN
ldm.prf_ecc = ((ldm.prf_x - XCEN) ** 2 + (ldm.prf_y - YCEN) ** 2) ** .5
# </codecell>


# <markdowncell>
"""
## Correlates in visual cortex of pupil size, luminance, and visual change

__Conclusions:

- Pupil size correlates negatively with VC activity
- Visual change correlates negatively with VC activity
- Luminance does not notably correlate with VC activity
"""
# </markdowncell>


# <codecell>
def roi_hist(dm, col):

    for roi, rdm in ops.split(dm.roi):
        sns.distplot(
            rdm[col], label='{} (N={})'.format(roi, len(rdm)),
            hist=True,
            kde=False,
            bins=50,
            hist_kws={
                "alpha": .8,
                "color": COLORS[roi]
            }
        )
    plt.xlabel('t value')
    plt.ylabel('Density')
    plt.axvline(0, color='black', linestyle=':')
    plt.legend()


plot.new(size=(16, 5))
plt.subplot(1, 3, 1)
plt.xlim(-5, 5)
plt.title('a) VC ~ pupil size')
roi_hist(ldm, 't_bold_pupil')
plt.subplot(1, 3, 2)
plt.xlim(-5, 5)
plt.title('b) VC ~ luminance')
roi_hist(ldm, 't_bold_luminance')
plt.subplot(1, 3, 3)
plt.xlim(-10, 10)
plt.title('c) VC ~ visual change')
roi_hist(ldm, 't_bold_change')
plot.save('histograms', show=True)
# </codecell>


# <markdowncell>
"""
## Correlations between pupil size, luminance, and visual change

__Conclusion:__

- Pupil size and luminance are strongly negatively correlated
- Pupil size and visual change are negatively correlated
- Visual change and luminance are positively correlated
"""
# </markdowncell>


# <codecell>
def corr_plot(y):

    plt.ylim(-15, 15)
    plt.plot(sorted(y), 'o')
    plt.xticks([])
    plt.ylabel('t value')
    plt.axhline(0, color='black', linestyle=':')


plot.new(size=(16, 5))
plt.subplot(1, 3, 1)
plt.title('a) Pupil size ~ luminance')
corr_plot(dm.t_pupil_luminance)
plt.subplot(1, 3, 2)
plt.ylim(-15, 15)
plt.title('b) Pupil size ~ visual change')
corr_plot(dm.t_pupil_change)
plt.subplot(1, 3, 3)
plt.ylim(-15, 15)
plt.title('c) Visual change ~ luminance')
corr_plot(dm.t_luminance_change)
plot.save('correlations', show=True)
# </codecell>


# <markdowncell>
"""
## PRF properties

The PRF size and eccentricity as a function of ROI.

__Conclusion:__ PRFs become larger and more eccentric from V1 to V4.
"""
# </markdowncell>


# <codecell>
plot.new(size=(12, 5))
plt.subplot(1, 2, 1)
plt.title('a) PRF size ~ roi')
sns.barplot(y='prf_sd', x='roi', data=ldm)
plt.ylabel('Standard deviation (px)')
plt.subplot(1, 2, 2)
plt.title('b) PRF eccentricity ~ roi')
plt.ylabel('Eccentricity (px)')
sns.barplot(y='prf_ecc', x='roi', data=ldm)
plot.save('prf-size-ecc', show=True)
# </codecell>


# <markdowncell>
"""
The relationship between PRF size and eccentricity and the bold response to
trace signals.

__Conclusion:__ PRF properties do not predict the correlation between the bold
response and trace signals very well.
"""
# </markdowncell>


# <codecell>
for trace, prf_col in itertools.product(TRACES, ('prf_sd', 'prf_ecc')):
    print('(bold ~ {}) ~ {}'.format(trace, prf_col))
    print(lme4.lmer(
        ldm,
        't_bold_{trace} ~ {prf_col} + (1+{prf_col}|sub)'.format(
            trace=trace,
            prf_col=prf_col
        )
    ))
# </codecell>

# <codecell>
for trace in TRACES:
    print('(bold ~ {}) ~ roi'.format(trace))
    print(lme4.lmer(
        ldm,
        't_bold_{trace} ~ roi + (1+roi|sub)'.format(trace=trace)
    ))
# </codecell>
