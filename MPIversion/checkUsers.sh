#!/bin/bash
ALL_HOSTS=(quattro eos01 eos02 eos03 eos04 eos05 eos06 eos07 eos08 eos09 eos10 eos11 eos12 eos13 eos14 eos15 eos16 eos17 eos18 eos19 eos20 eos21 eos22 eos23 eos24 eos25 eos26 eos27 eos28 eos29 eos30 eos31 eos32 dc01 dc02 dc04 dc05 dc07 dc08 dc09 dc10 dc11 dc12 dc13 dc14 dc15 dc16 dc17 dc18 dc19 dc20 dc21 dc22 dc23 dc24 dc25 arch01 arch02 arch03 arch04 arch05 arch06 arch07 arch08 arch09 arch10)
#dc03,dc06
TARGETS=("${ALL_HOSTS[@]}")
me="patricro "
for HOST in "${TARGETS[@]}"; do
	who=$(ssh -o LogLevel=QUIET $HOST -t 'users')
	whos=("${who[@]}")
	for who2 in $whos; do
    	if [ "${who2::-1}" != "patricro" ]; then
    		if [ "$who2" != "patricro" ]; then
  				printf "$HOST is busy with user $who2\n"
  			fi
		fi
    done
done
