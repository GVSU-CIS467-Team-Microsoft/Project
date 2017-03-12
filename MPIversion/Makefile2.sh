#!/bin/bash
mpic++ -Ofast MPIv2.cpp -I/home/patricro/Documents/Capstone/MPIversion/library/include -I/usr/include/openmpi-x86_64 -L/usr/lib64/openmpi/lib -L/home/patricro/Documents/Capstone/MPIversion -L/home/patricro/Documents/Capstone/MPIversion/artifacts -o MPIv2 -lmpi -lopenblas -lpthread -std=c++11
