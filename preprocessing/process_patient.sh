#! /bin/bash

#Path to the preprocessing python script
SCRIPT_PATH="/home/jannengm/CIS467/Project/preprocessing/preprocessing.py"

#Path to preprocessed files
STAGE_1_PREPROCESSED_TEMP="/home/jannengm/CIS467/stage1/preprocessed"

#Path to temporary local storage for stage 1 source files
STAGE_1_TEMP_SOURCE="/home/jannengm/CIS467/stage1/stage1"

STAGE_1_PREPROCESSED="/run/media/jannengm/6dfd86ff-f041-4ee4-94c9-6c8c5f1c3ace/stage1/preprocessed"

local_patient="$STAGE_1_TEMP_SOURCE/$(basename $1)"
outfile="$STAGE_1_PREPROCESSED_TEMP/$(basename $1).dat"

num_local_patients="$(ls $STAGE_1_TEMP_SOURCE | wc -l)"

while(( "$num_local_patients" >= 50 )); do
	sleep 5
	num_local_patients="$(ls $STAGE_1_TEMP_SOURCE | wc -l)"
done

#echo Copying $(basename $1) to $local_patient
cp -r -f $1 $STAGE_1_TEMP_SOURCE

echo processing $(basename $local_patient) on $2

#ssh $2 "python3 /home/jannengm/CIS467/Project/preprocessing/preprocessing.py \"$1\""
#ssh $2 "echo hello from \"$2\""
ssh $2 "cd \"$STAGE_1_PREPROCESSED_TEMP\"; python3 \"$SCRIPT_PATH\" \"$local_patient\" > /dev/null 2>&1"
#ssh $2 "cd \"$STAGE_1_PREPROCESSED_TEMP\"; python3 \"$SCRIPT_PATH\" \"$local_patient\" "

mv $outfile $STAGE_1_PREPROCESSED
rm -rf $local_patient

echo $(basename $local_patient) completed on $2

