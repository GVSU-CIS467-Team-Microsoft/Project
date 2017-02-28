CC = nvcc
override CFLAGS += -O2

.PHONY: clean

#$(CC) $(CFLAGS) -I/usr/lib/openmpi/include -L/usr/lib/openmpi/lib/openmpi -lmpi -lcudnn pConcept.cu -o pConcept -std=c++11 -Wno-deprecated-gpu-targets
HW: clean copy
	$(CC) $(CFLAGS) -I/usr/lib/openmpi/include,/home/ron-patrick/Documents/Capstone/pConcept/library/include -L/usr/lib/openmpi/lib/openmpi,/home/ron-patrick/Documents/Capstone/pConcept/artifacts -lcublas -lmpi -limebra -lcurand olga.cu -o olga -std=c++11 -Wno-deprecated-gpu-targets

clean:
	rm -f olga

copy:
	cp olga.cpp olga.cu
