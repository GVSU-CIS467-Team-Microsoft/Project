#! /bin/bash

#Path to the preprocessing python script
SCRIPT_PATH="/home/jannengm/workspace/CIS467/Project/preprocessing/preprocessing.py"

#Path to stage 1 source files
STAGE_1_SOURCE="/media/jannengm/MyPassport/stage1"

#Path to preprocessed files
STAGE_1_PREPROCESSED="/home/jannengm/workspace/CIS467/stage1/preprocessed"

#Maximum number of concurrent threads
MAX_THREADS=4

#Initialize loop variables
testsRun=0

patient_list=()
# printf "\rRunning Tests...%d%%" "$((100*$testsRun/($numBots*$numTests)))"
for PATIENT in $STAGE_1_SOURCE/*; do
  patient_list+=("$PATIENT")
done

num_patients="${#patient_list[@]}"
active_threads=0
patient_index=0

printf "\rProcessing patients...%d%%" "$(((100*$patient_index)/$num_patients))"
while (( "$patient_index" < "$num_patients")); do
  while (( "$active_threads" < "$MAX_THREADS" )); do
    python3 $SCRIPT_PATH ${patient_list[$patient_index]} > /dev/null 2>&1 &
    ((patient_index+=1))
    ((active_threads+=1))
    if (("$patient_index" >= "$num_patients")); then
      break
    fi
  done
  wait
  ((active_threads=0))
  printf "\rProcessing patients...%d%%" "$(((100*$patient_index)/$num_patients))"
done
