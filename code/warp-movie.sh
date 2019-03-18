#!/bin/bash

ALIGNED="inputs/studyforrest-data-aligned/sub-%s/in_bold3Tp2/sub-%s_task-avmovie_run-%s_bold.nii.gz"
BRAIN="inputs/studyforrest-data-templatetransforms/templates/grpbold3Tp2/brain.nii.gz"
TEMPLATE="inputs/studyforrest-data-templatetransforms/sub-%s/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz"
WARPED="inputs/studyforrest-data-mni/sub-%s/sub-%s_task-avmovie_run-%s_bold.nii.gz"

for SUB in "19" "20" # "01" "02" "03" "04" "05" "06" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20"
do
   mkdir "inputs/studyforrest-data-mni/sub-$SUB"
    for RUN in "1" "2" "3" "4" "5" "6" "7" "8"
    do
        datalad get "inputs/studyforrest-data-aligned/sub-$SUB/in_bold3Tp2/sub-"$SUB"_task-avmovie_run-"$RUN"_bold.nii.gz"
        fsl5.0-applywarp \
            -i $(printf $ALIGNED $SUB $SUB $RUN) \
            -r "$BRAIN" \
            -w $(printf $TEMPLATE $SUB) \
            -o $(printf $WARPED $SUB $SUB $RUN)
        datalad drop "inputs/studyforrest-data-aligned/sub-$SUB/in_bold3Tp2/sub-"$SUB"_task-avmovie_run-"$RUN"_bold.nii.gz"
    done
done
