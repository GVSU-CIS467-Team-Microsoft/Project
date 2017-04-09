#!/bin/bash

#Path to the preprocessing python script
SCRIPT_PATH="/home/jannengm/CIS467/Project/preprocessing/preprocessing.py"

#Path to stage 1 source files
STAGE_1_SOURCE="/home/jannengm/CIS467/stage1/stage1"

#Path to preprocessed files
STAGE_1_PREPROCESSED="/home/jannengm/CIS467/stage1/preprocessed"

MAX_THREADS=8

#List of all hosts to attempt
# ALL_HOSTS=(eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc15 dc16 dc17 dc18 dc19 dc20 dc21 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)
ALL_HOSTS=(eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10)

#Checks if a host is available. Returns 0 if no users are logged in
function check_host()
{
		busy=false
		self=$(whoami)
		user_list=$(ssh -o LogLevel=QUIET $1 -t 'users')
		if [ -z "$user_list" ]; then
			# printf "could not connect to $1\n"
			return 2
		fi

		users=("${user_list[@]}")
		for user in $users; do
			if [ "${user::-1}" != $self ]; then
				if [ "$user" != $self ]; then
					# printf "$1 is busy with user $user\n"
					busy=true
					return 1
				fi
			fi
		done

		if [ "$busy" = false ]; then
			# printf "$1 is available\n"
			return 0
		fi
}

#Get list of available machines
TARGETS=("${ALL_HOSTS[@]}")

host_num=0
for PATIENT in $STAGE_1_SOURCE/*; do
	attempts=0
	#Try to find a host to run preprocessing on
	while (( "$attempts" < "${#TARGETS[@]}" )); do

		#If available host is found, run preprocessing and reset attempt count, and advance to next host
		if check_host "${TARGETS[$host_num]}"; then
				((attempts=0))
				num_threads=$(ssh ${TARGETS[$host_num]} "ps -u $(whoami) | grep python | wc -l")
				if [ "$num_threads" -lt "$MAX_THREADS" ]; then
					./process_patient.sh ${TARGETS[$host_num]} $PATIENT $STAGE_1_PREPROCESSED $SCRIPT_PATH &
					# echo Go on "${TARGETS[$host_num]}"
					((host_num+=1))
					if (( "$host_num" >= "${#TARGETS[@]}")); then
						((host_num=0))
					fi
					break
				fi

				((host_num+=1))
				if (( "$host_num" >= "${#TARGETS[@]}")); then
					((host_num=0))
				fi


		#If host is unavailable, advance to next host and increment attempt count
		else
			((host_num+=1))
			((attempts+=1))
			if(( "$host_num" >= "${#TARGETS[@]}")); then
				((host_num=0))
			fi
		fi
	done

	#If loop exited with attempts >= the number of available hosts, then there are no avaialbe hosts
	if (( "$attempts" >= "${#TARGETS[@]}" )); then
		echo No available hosts
		exit 1
	fi

done
wait
