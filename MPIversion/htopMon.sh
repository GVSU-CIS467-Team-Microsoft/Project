#!/bin/bash
ALL_HOSTS=(eos01 eos02 eos04)

if [ $# == 0 ]; then
    TARGETS=("${ALL_HOSTS[@]}")
else
    echo '$0:' $0
    TARGETS=("$@")
fi

for HOST in "${TARGETS[@]}"; do
    terminal -t $HOST -e "ssh $HOST -t htop" &
done