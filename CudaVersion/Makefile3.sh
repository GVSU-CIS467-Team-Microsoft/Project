#!/bin/bash
cp mnistCUDNN.cpp mnistCUDNN.cu
nvcc -O3 -ImnistCUDNN -L/home/patricro/Documents/Capstone/MPIversion/cuda -lcudart -lcublas mnistCUDNN.cu -o cudnnTest -std=c++11 -lcudnn -lfreeimage
