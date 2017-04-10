#!/bin/bash
echo g++ -Ofast MPIv2.cpp -L/home/gvuser/MPIversion -o MPIv3 -lopenblas -std=c++11
g++ -Ofast MPIv2.cpp -L/home/gvuser/MPIversion -o MPIv3 -lopenblas -std=c++11
