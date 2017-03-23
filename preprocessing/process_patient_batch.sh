#! /bin/bash

#Path to the preprocessing python script
#SCRIPT_PATH="/home/jannengm/CIS467/Project/preprocessing/preprocessing.py"

#Path to preprocessed files
#STAGE_1_PREPROCESSED_TEMP="/home/jannengm/CIS467/stage1/preprocessed"

#Path to temporary local storage for stage 1 source files
#STAGE_1_TEMP_SOURCE="/home/jannengm/CIS467/stage1/stage1"

#STAGE_1_PREPROCESSED="/run/media/jannengm/6dfd86ff-f041-4ee4-94c9-6c8c5f1c3ace/stage1/preprocessed"

# First parameter should be host to run script on
HOST="$1"

# Second parameter should be path to patient directory
PATIENT="$2"

# Third parateter should be target directory for preprocessing output
TARGET_DIR="$3"

# Forth parameter should be path to preprocessing script
SCRIPT_PATH="$4"

# Fifth parameter should be path to temporary patient storage
TEMP_PATIENT="$5"

# Sixth paramter should be path to temporary preprocessing storage
TEMP_PREPROCESSING="$6"

local_patient="$TEMP_PATIENT/$(basename $PATIENT)"
outfile="$TEMP_PREPROCESSING/$(basename $PATIENT).dat"

#echo local_patient- $local_patient
#echo outfile- $outfile

#num_local_patients="$(ls $STAGE_1_TEMP_SOURCE | wc -l)"

#while(( "$num_local_patients" >= 50 )); do
#	sleep 5
#	num_local_patients="$(ls $STAGE_1_TEMP_SOURCE | wc -l)"

#echo Copying $(basename $PATIENT) to $local_patient
cp -r -f $PATIENT $TEMP_PATIENT

echo processing $(basename $local_patient) on $HOST

#ssh $2 "python3 /home/jannengm/CIS467/Project/preprocessing/preprocessing.py \"$1\""
#ssh $2 "echo hello from \"$2\""
ssh $HOST "cd \"$TEMP_PREPROCESSING\"; python3 \"$SCRIPT_PATH\" \"$local_patient\" > /dev/null 2>&1"
#ssh $2 "cd \"$STAGE_1_PREPROCESSED_TEMP\"; python3 \"$SCRIPT_PATH\" \"$local_patient\" "

mv $outfile $TARGET_DIR
rm -rf $local_patient

echo $(basename $local_patient) completed on $HOST

