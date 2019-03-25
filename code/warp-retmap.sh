#!/bin/bash

ALIGNED="inputs/studyforrest-data-aligned/sub-%s/in_bold3Tp2/sub-%s_task-retmap%s_run-1_bold.nii.gz"
BRAIN="inputs/studyforrest-data-templatetransforms/templates/grpbold3Tp2/brain.nii.gz"
TEMPLATE="inputs/studyforrest-data-templatetransforms/sub-%s/bold3Tp2/in_grpbold3Tp2/subj2tmpl_warp.nii.gz"
WARPED="inputs/studyforrest-data-mni/sub-%s_task-retmap%s_run-1_bold.nii.gz"

for SUB in "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20"
do
    mkdir "inputs/studyforrest-data-mni/sub-$SUB"
    for CON in "con" "exp" "clw" "ccw"
    do
        fsl5.0-applywarp \
            -i $(printf $ALIGNED $SUB $SUB $CON) \
            -r "$BRAIN" \
            -w $(printf $TEMPLATE $SUB) \
            -o $(printf $WARPED $SUB $CON)
    done
done
