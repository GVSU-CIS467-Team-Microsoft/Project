#!/bin/bash

rm -f pConcept
cp pConcept.cpp pConcept.cu
nvcc -O2 -I/usr/lib/openmpi/include -L/usr/lib/openmpi/lib/openmpi -lmpi pConcept.cu -o pConcept -std=c++11 -Wno-deprecated-gpu-targets