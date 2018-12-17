# StudyForrest population-receptive-field mapping

Copyright 2018 Sebastiaan Mathôt (<https://www.cogsci.nl/smathot>)


## About

This repository contains code to perform population-receptive-field (PRF) mapping based on the retinotopic-mapping data from the [StudyForrest](http://studyforrest.org/) project. PRF mapping is a technique to estimate the spatial coordinates and standard deviation of receptive fields of voxels in fMRI data.


## Step 1: Get input data files from StudyForrest

~~~
datalad get inputs/studyforrest-data-aligned/sub-*/in_bold3Tp2/sub-*_task-retmap*_run-1_bold.nii.gz
datalad get inputs/studyforrest-data-templatetransforms/templates/grpbold3Tp2/brain.nii.gz
datalad get inputs/studyforrest-data-templatetransforms/sub-*/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz
datalad get inputs/studyforrest-data-phase2/stimuli/retinotopic_mapping/*.mkv
~~~


## Step 2: Transform fMRI data to MNI space

The data from StudyForrest is in a participant-specific space. `warp.sh` converts this data to MNI space, which is used by the rest of the scripts. This requires the `applywarp` command from FSL 5.0.

~~~
./code/warp.sh
~~~


## Step 3: PRF mapping

See `code/prf-mapping.ipynb`.


## License

This code is distributed under the terms of the GNU General Public License 3. The full license should be included in the file COPYING, or can be obtained from:

- http://www.gnu.org/licenses/gpl.txt
