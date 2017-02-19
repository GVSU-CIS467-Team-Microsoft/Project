#! /bin/bash

################################################################################
# CIS 467 - Capstone
# @author Mark Jannenga
#
# This script splits the stage1 Kaggle dataset into a training set and a test
# set based on the information in stage1_sample_submission.csv. It creates soft
# links to the original data, and does not actualy move or copy any of the
# original data.
################################################################################

# Use the variables below to specify the paths to the ucompressed source data,
# the desired destination directory for the training data, the destination for
# the testing data, and stage1_sample_submission.csv, respectively
stage1_path="/home/jannengm/workspace/CIS467/stage1/stage1"

stage1_training_path="/home/jannengm/workspace/CIS467/training"

stage1_testing_path="/home/jannengm/workspace/CIS467/testing"

stage1_sample_submission_path="/home/jannengm/workspace/CIS467/stage1_sample_submission.csv"

# Check for existence of specified stage1 and stage1_sample submission paths
if [ ! -d $stage1_path ]; then
	echo "$stage1_path not found. Aborting..."
	exit 1
fi

if [ ! -r $stage1_sample_submission_path ]; then
	echo "$stage1_sample_submission_path not found. Aborting..."
	exit 1
fi


# Check for existence of specified training and test directories
if [ -d $stage1_training_path ]; then
	while true; do
		read -p "$stage1_training_path already exists. Overwrite? [y/n]: " yn
		case $yn in
			[Yy]* ) rm $stage1_training_path/*; break;;
			[Nn]* ) exit 1;;
			* ) echo "Please anser yes or no";;
		esac
	done
else
	mkdir $stage1_training_path
fi

if [ -d $stage1_testing_path ]; then
	while true; do
		read -p "$stage1_testing_path already exists. Overwrite? [y/n]: " yn
		case $yn in
			[Yy]* ) rm $stage1_testing_path/*; break;;
			[Nn]* ) exit 1;;
			* ) echo "Please anser yes or no";;
		esac
	done
else
	mkdir $stage1_testing_path
fi


# Loop through the stage1 data set and create soft links in the testing and
# training directories. No need to move or copy the files, as there may be a need
# for the data in is original format later.
for patient in $stage1_path/*
do
	# echo $(basename $patient)
	if grep -q "$(basename $patient),[0,1]" $stage1_sample_submission_path; then
		# echo "$(basename $patient) found"
		ln -s $patient $stage1_testing_path/$(basename $patient)
	else
		# echo "$(basename $patient) not found"
		ln -s $patient $stage1_training_path/$(basename $patient)
	fi
done
