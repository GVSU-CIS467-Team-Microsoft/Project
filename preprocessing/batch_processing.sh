#!/bin/bash

#Path to the preprocessing python script
SCRIPT_PATH="/home/jannengm/CIS467/Project/preprocessing/preprocessing.py"

#Path to stage 1 source files
#STAGE_1_SOURCE="/home/jannengm/CIS467/stage1/stage1"
STAGE_1_SOURCE="/run/media/jannengm/6dfd86ff-f041-4ee4-94c9-6c8c5f1c3ace/stage1/stage1"

#Path to preprocessed files
STAGE_1_PREPROCESSED="/home/jannengm/CIS467/stage1/preprocessed"

ALL_HOSTS=(eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc15 dc16 dc17 dc18 dc19 dc20 dc21 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)

TARGETS=("${ALL_HOSTS[@]}")
self="jannengm"
available=()
for HOST in "${TARGETS[@]}"; do
	busy=false
	user_list=$(ssh -o LogLevel=QUIET $HOST -t 'users')
	if [ -z "$user_list" ]; then
#		printf "could not connect to $HOST\n"
		continue
	fi
	
	users=("${user_list[@]}")
	for user in $users; do
		if [ "${user::-1}" != $self ]; then
			if [ "$user" != $self ]; then
#				printf "$HOST is busy with user $user\n"
				busy=true
				break
			fi
		fi
	done
	
	if [ "$busy" = false ]; then
#		printf "$HOST is available\n"
		available+=("$HOST")
	fi
done

printf "Available hosts (%d):\n" "${#available[@]}"
echo ${available[@]}

host_num=0
for PATIENT in $STAGE_1_SOURCE/*; do
	#echo "Running " $PATIENT " on " ${available[$host_num]}
	#ssh -f ${available[$host_num]} "cd $STAGE_1_PREPROCESSED; python3 $SCRIPT_PATH $PATIENT > /dev/null; echo $PATIENT completed on ${available[$host_num]}; exit"
	#ssh ${available[$host_num]} "cd \"$STAGE_1_PREPROCESSED\"; python3 \"$SCRIPT_PATH\" \"$PATIENT\" > /dev/null 2>&1; echo \"$PATIENT\" completed on \"${available[$host_num]}\"" &
	./process_patient.sh $PATIENT ${available[$host_num]} &
	sleep 1
	((host_num+=1))
	if(( "$host_num" >= "${#available[@]}")); then
		((host_num=0))
	fi
done
wait
