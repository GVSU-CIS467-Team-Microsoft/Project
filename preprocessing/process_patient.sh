#! /bin/bash

# First parameter should be host to run script on
HOST="$1"

# Second parameter should be path to patient directory
PATIENT="$2"

# Third parateter should be target directory for preprocessing output
TARGET_DIR="$3"

# Forth parameter should be path to preprocessing script
SCRIPT_PATH="$4"

outfile="$TARGET_DIR/$(basename $PATIENT).dat"

echo processing $(basename $PATIENT) on $HOST

ssh $HOST "cd \"$TARGET_DIR\"; python3 \"$SCRIPT_PATH\" \"$PATIENT\" > /dev/null 2>&1"

echo $(basename $PATIENT) completed on $HOST
