#!/bin/bash

#Path to the preprocessing python script
SCRIPT_PATH="/home/jannengm/CIS467/Project/preprocessing/preprocessing.py"

#Path to stage 1 source files
#STAGE_1_SOURCE="/home/jannengm/CIS467/stage1/stage1"
STAGE_1_SOURCE="/run/media/jannengm/6dfd86ff-f041-4ee4-94c9-6c8c5f1c3ace/stage1/stage1"

#Path to preprocessed files
STAGE_1_PREPROCESSED="/run/media/jannengm/6dfd86ff-f041-4ee4-94c9-6c8c5f1c3ace/stage1/preprocessed"

#Path to temporary storage for source files
TEMP_SOURCE="/home/jannengm/CIS467/stage1/stage1"

#Path to temporary storage for preprocessing output
TEMP_PREPROCESSING="/home/jannengm/CIS467/stage1/preprocessed"

MAX_PATIENTS=40

MAX_THREADS=8

#ALL_HOSTS=(eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc15 dc16 dc17 dc18 dc19 dc20 dc21 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)
ALL_HOSTS=(eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc16 dc17 dc18 dc19 dc20 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)

TARGETS=("${ALL_HOSTS[@]}")

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

host_num=0
#for PATIENT in $STAGE_1_SOURCE/*; do
#
#	num_local_patients="$(ls $TEMP_SOURCE | wc -l)"
#	while(( "$num_local_patients" >= $MAX_PATIENTS )); do
#		sleep 5
#		num_local_patients="$(ls $TEMP_SOURCE | wc -l)"
#	done
#	
#	./process_patient_batch.sh $PATIENT ${available[$host_num]} &
#	sleep 1
#	((host_num+=1))
#	if(( "$host_num" >= "${#available[@]}")); then
#		((host_num=0))
#	fi
#done
for PATIENT in $STAGE_1_SOURCE/*; do

    #Sleep until space is available
    num_local_patients="$(ls $TEMP_SOURCE | wc -l)"
    while(( "$num_local_patients" >= $MAX_PATIENTS )); do
        sleep 5
        num_local_patients="$(ls $TEMP_SOURCE | wc -l)"
    done

    #Try to find a host to run preprocessing on
    attempts=0
    while (( "$attempts" < "${#TARGETS[@]}" )); do

        #If available host is found, run preprocessing and reset attempt count, and advance to next host
        if check_host "${TARGETS[$host_num]}"; then
            ((attempts=0))
            num_threads=$(ssh ${TARGETS[$host_num]} "ps -u $(whoami) | grep python | wc -l")
            if [ "$num_threads" -lt "$MAX_THREADS" ]; then
                ./process_patient_batch.sh ${TARGETS[$host_num]} $PATIENT $STAGE_1_PREPROCESSED $SCRIPT_PATH $TEMP_SOURCE $TEMP_PREPROCESSING &
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
