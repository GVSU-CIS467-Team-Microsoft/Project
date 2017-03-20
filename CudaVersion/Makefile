CC = nvcc
override CFLAGS += -O3

.PHONY: clean

#$(CC) $(CFLAGS) -I/usr/include/openmpi-x86_64/ -L/usr/lib/openmpi/lib/openmpi -lmpi -lcudnn pConcept.cu -o pConcept -std=c++11 -Wno-deprecated-gpu-targets
#$(CC) $(CFLAGS) -I/usr/include/openmpi-x86_64/,/home/patricro/Documents/Capstone/MPIversion/library/include -L/usr/lib64/openmpi/lib,/home/patricro/Documents/Capstone/MPIversion/artifacts -lcublas -lmpi -limebra -lcurand MPIv.cu -o MPIv -std=c++11 -Wno-deprecated-gpu-targets
#$(CC) $(CFLAGS) -I/usr/include/openmpi-x86_64/ -L/home/patricro/Documents/Capstone/MPIversion/cuda -L/usr/lib64/openmpi/lib -lcudnn -lcublas -lmpi -lcurand MPIv.cu -o MPIv -std=c++11 -Wno-deprecated-gpu-targets
HW: clean copy
	$(CC) $(CFLAGS) -lcublas -lcurand MPIv.cu -o MPIv -std=c++11 -Wno-deprecated-gpu-targets

clean:
	rm -f MPIv

copy:
	cp MPIv.cpp MPIv.cu
