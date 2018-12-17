# StudyForrest population-receptive-field mapping

Copyright 2018 Sebastiaan Mathôt (<https://www.cogsci.nl/smathot>)


## About

This repository contains code to perform population-receptive-field (PRF) mapping based on the retinotopic-mapping data from the [StudyForrest](http://studyforrest.org/) project. PRF mapping is a technique to estimate the spatial coordinates and standard deviation of receptive fields of voxels in fMRI data.


## Transforming fMRI data to MNI space

The data from StudyForrest is in a participant-specific space. We want to transform the data to standard MNI space. To do so, we first need to get the data from the `studyforrest-data-aligned` dataset, and then execute `code/warp.sh`. This requires the `applywarp` command from FSL 5.0.

~~~
datalad get inputs/studyforrest-data-aligned/sub-*/in_bold3Tp2/sub-*_task-retmap*_run-1_bold.nii.gz
code/warp.sh
~~~


## PRF mapping

TODO


## License

This code is distributed under the terms of the GNU General Public License 3. The full license should be included in the file COPYING, or can be obtained from:

- http://www.gnu.org/licenses/gpl.txt
