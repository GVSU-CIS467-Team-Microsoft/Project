CC = nvcc
override CFLAGS += -O2

.PHONY: clean

#$(CC) $(CFLAGS) -I/usr/lib/openmpi/include -L/usr/lib/openmpi/lib/openmpi -lmpi -lcudnn pConcept.cu -o pConcept -std=c++11 -Wno-deprecated-gpu-targets
HW: clean copy
	$(CC) $(CFLAGS) -I/usr/lib/openmpi/include -L/usr/lib/openmpi/lib/openmpi -lmpi pConcept.cu -o pConcept -std=c++11 -Wno-deprecated-gpu-targets

clean:
	rm -f pConcept

copy:
	cp pConcept.cpp pConcept.cu
