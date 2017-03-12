#!/bin/bash

#mpirun -np 25 -host eos01,eos02,eos04,eos05,eos06,eos07,eos08,eos09,eos10,eos11,eos13,eos14,eos15,eos16,eos17,eos18,eos19,eos21,eos22,eos23.eos24,eos25,eos26,eos27,eos28,eos29 MPIv2 layers=196 showInterval=500 setSize=60000 learningRate=2
mpirun -np 6 -host eos01,eos02,eos03,eos04,eos05,eos06 MPIv2 layers=128,64,32 learningRate=0.999 showInterval=100 setSize=60000
#mpirun -np 40 -host eos01,dc01,dc02,dc04,dc05,dc07,eos02,eos03,eos04,eos05,eos06,eos07,eos09,eos13,eos14,eos15,eos16,eos17,eos18,eos19,eos20,eos21,eos22,eos23,eos24,eos25,eos26,eos27,eos28,eos29,eos30,eos31,eos32,arch01,arch02,arch03,arch04,arch05,arch06,arch07,arch08,arch09,arch10 MPIv2 setSize=60000 $1 $2 $3
#dc06,dc03,eos08,eos10,eos11,eos12,dc08,dc09,dc10,dc11,dc12,dc13,dc14,dc15,dc16,dc17,dc18,dc19,dc20,dc21,dc22,dc23,dc24,dc25