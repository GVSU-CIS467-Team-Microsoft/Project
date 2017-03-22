#!/bin/bash

function check_host()
{
		busy=false
		self=$(whoami)
		user_list=$(ssh -o LogLevel=QUIET $1 -t 'users')
		if [ -z "$user_list" ]; then
			printf "could not connect to $HOST\n"
			return 2
		fi

		users=("${user_list[@]}")
		for user in $users; do
			if [ "${user::-1}" != $self ]; then
				if [ "$user" != $self ]; then
					printf "$HOST is busy with user $user\n"
					busy=true
					return 1
				fi
			fi
		done

		if [ "$busy" = false ]; then
			printf "$HOST is available\n"
			return 0
		fi
}

ALL_HOSTS=(eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc15 dc16 dc17 dc18 dc19 dc20 dc21 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)

TARGETS=("${ALL_HOSTS[@]}")
self=$(whoami)
available=()
# for HOST in "${TARGETS[@]}"; do
# 	busy=false
# 	user_list=$(ssh -o LogLevel=QUIET $HOST -t 'users')
# 	if [ -z "$user_list" ]; then
# 		printf "could not connect to $HOST\n"
# 		continue
# 	fi
#
# 	users=("${user_list[@]}")
# 	for user in $users; do
# 		if [ "${user::-1}" != $self ]; then
# 			if [ "$user" != $self ]; then
# 				printf "$HOST is busy with user $user\n"
# 				busy=true
# 				break
# 			fi
# 		fi
# 	done
#
# 	if [ "$busy" = false ]; then
# 		printf "$HOST is available\n"
# 		available+=("$HOST")
# 	fi
# done

for HOST in "${TARGETS[@]}"; do
	# if [[ "$(check_host "$HOST")" == "0" ]]; then
	if check_host "$HOST"; then
		available+=("$HOST")
	fi
done

printf "Available hosts (%d):\n" "${#available[@]}"
echo ${available[@]}
