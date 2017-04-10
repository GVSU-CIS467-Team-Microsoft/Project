#!/bin/bash

#ALL_HOSTS=(eos01 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc03 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc16 dc17 dc18 dc19 dc20 dc21 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)
#list is minus dc06, dc15, and eos02
#ALL_HOSTS=(eos01 eos03 eos04 eos05)
#$(echo './putMPIv2.sh')
#./putMPIv2.sh

#ALL_HOSTS=(gv2017zvg000000 gv2017zvg000001 gv2017zvg00000A gv2017zvg00000C gv2017zvg000003 gv2017zvg000004 gv2017zvg000005 gv2017zvg000007 gv2017zvg000008 gv2017zvg000009)
ALL_HOSTS=(azure1 azure2 azure3 azure4 azure5 azure6 azure7 azure8 azure9 azure10)

TARGETS=("${ALL_HOSTS[@]}")
self="gvuser"
available=()
once=false
count=0
max=10
for HOST in "${TARGETS[@]}"; do

	if [ "$count" -gt "$max" ]; then
		continue
	fi
	
	busy=false
	user_list=$(ssh -o LogLevel=QUIET $HOST -t 'users')
	if [ -z "$user_list" ]; then
		continue
	fi
	
	users=("${user_list[@]}")
	for user in $users; do
		if [ "${user::-1}" != $self ]; then
			if [ "$user" != $self ]; then
				busy=true
				break
			fi
		fi
	done
	
	if [ "$busy" = false ]; then
		count=$((count+1))
		NOSPACE=$(echo "$HOST" | tr -s " ")
		if [ "$once" = false ]; then
			available+=($NOSPACE)
			once=true
		else
			available+=($NOSPACE)
		fi
	fi
done

printf "Available hosts (%d):\n" "${#available[@]}"
howMany=${#available[@]}
available=$( echo ${available[@]} | sed 's/[[:space:]]/,/g')
echo mpirun --mca pml ob1 -np $howMany -host $available MPIv3 $1 $2 $3 $4 $5
mpirun --mca pml ob1 -np $howMany -host $available MPIv3 $1 $2 $3 $4 $5
#mpirun -np $howMany -host $available MPIv2 $1 $2 $3 $4 $5

#ALL_HOSTS=(eos01 eos08 eos32 dc01 dc02 dc03 arch02)

#mpirun -np 25 -host eos01,eos02,eos04,eos05,eos06,eos07,eos08,eos09,eos10,eos11,eos13,eos14,eos15,eos16,eos17,eos18,eos19,eos21,eos22,eos23.eos24,eos25,eos26,eos27,eos28,eos29 MPIv2 layers=196 showInterval=500 setSize=60000 learningRate=2
#mpirun -np 10 -host eos01,eos03,eos06,eos13,eos14,eos15,eos17,eos19,eos23,eos25 MPIv2 layers=128,128 learningRate=0.01 showInterval=1000 setSize=60000 batchSize=5
#mpirun -np 2 -host eos01,eos03 MPIv2 layers=10 learningRate=0.01 showInterval=10 setSize=2 batchSize=2
#mpirun --mca pml ob1 -np 4 -host eos01,eos03,eos04,eos05 MPIv2 $1 $2 $3 $4 $5
#mpirun -np 40 -host eos01,dc01,dc02,dc04,dc05,dc07,eos02,eos03,eos04,eos05,eos06,eos07,eos09,eos13,eos14,eos15,eos16,eos17,eos18,eos19,eos20,eos21,eos22,eos23,eos24,eos25,eos26,eos27,eos28,eos29,eos30,eos31,eos32,arch01,arch02,arch03,arch04,arch05,arch06,arch07,arch08,arch09,arch10 MPIv2 setSize=60000 $1 $2 $3
#dc06,dc03,eos08,eos10,eos11,eos12,dc08,dc09,dc10,dc11,dc12,dc13,dc14,dc15,dc16,dc17,dc18,dc19,dc20,dc21,dc22,dc23,dc24,dc25
