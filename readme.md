# The correlation between visual-cortex activity and pupil size

Copyright 2018 - 2019 Sebastiaan Mathôt (<https://www.cogsci.nl/smathot>)


## About

This repository contains code to perform population-receptive-field (PRF) mapping based on the retinotopic-mapping data from the [StudyForrest](http://studyforrest.org/) project. PRF mapping is a technique to estimate the spatial coordinates and standard deviation of receptive fields of voxels in fMRI data.


## Dependencies

This code has been developed on Ubuntu 18.04 with FSL 5.0 installed through NeuroDebian. However, all code should run with minor adjustments on any system. The following packages are required:

- Python libraries for numeric computing and neuroimaging:
	- nilearn
	- sklearn
	- nibabel
	- nipy
	- datamatrix
- FSL 5.0


## Step 1: Get input data files from StudyForrest

~~~
datalad get inputs/studyforrest-data-aligned/sub-*/in_bold3Tp2/sub-*_task-retmap*_run-1_bold.nii.gz
datalad get inputs/studyforrest-data-templatetransforms/templates/grpbold3Tp2/brain.nii.gz
datalad get inputs/studyforrest-data-templatetransforms/sub-*/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz
datalad get inputs/studyforrest-data-phase2/stimuli/retinotopic_mapping/*.mkv
datalad get inputs/studyforrest-data-phase2/sub-*/ses-movie/func/sub-*_ses-movie_task-movie_run-*_recording-eyegaze_physio.tsv.gz
~~~


## Step 2: Transform fMRI data to MNI space

The data from StudyForrest is in a participant-specific space. `warp.sh` converts this data to MNI space, which is used by the rest of the scripts.

~~~
./code/warp.sh
~~~


## Step 3: PRF mapping

~~~
python3 code/create-prf-maps.py
~~~


## Step 4: Correlation matrices

~~~
python3 code/create-correlation-matrices.py
~~~


### Step 5: Inspect the results

Open `code\pupil-correlates.ipynb` in JupyterLab/ Jupyter Notebook.


## License

This code is distributed under the terms of the GNU General Public License 3. The full license should be included in the file COPYING, or can be obtained from:

- http://www.gnu.org/licenses/gpl.txt
