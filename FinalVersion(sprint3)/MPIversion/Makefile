CC = mpic++
#override CFLAGS += -Ofast
override CFLAGS += -Ofast

.PHONY: clean

#$(CC) $(CFLAGS) -I/home/gvuser/MPIversion/library/include -I/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/include/ -L/opt/intel/mkl/lib/intel64_lin -L/home/gvuser/MPIversion/artifacts MPIv2.cpp -DMKL_ILP64 -m64 -o MPIv2 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lmpi -fopenmp -std=c++14
#$(CC) $(CFLAGS) MPIv2.cpp -I/home/patricro/Documents/Capstone/MPIversion/library/include -I/usr/include/openmpi-x86_64 -L/usr/lib64/openmpi/lib -L/home/patricro/Documents/Capstone/MPIversion -L/home/patricro/Documents/Capstone/MPIversion/artifacts -o MPIv2 -lmpi -lopenblas -lpthread -std=c++11
#$(CC) $(CFLAGS) MPIv2.cpp -I/research/patricro/Kaggle/MPIversion/library/include -L/research/patricro/Kaggle/MPIversion -L/research/patricro/Kaggle/MPIversion/artifacts -o MPIv2 -lmpi -lopenblas -lpthread -std=c++11
#$(CC) $(CFLAGS) MPIv2.cpp -I/home/gvuser/MPIversion/library/include -L/home/gvuser/MPIversion -L/home/gvuser/MPIversion/artifacts -o MPIv2 -lmpi -lpthread -lopenblas -std=c++11
#$(CC) $(CFLAGS) MPIv2.cpp -I/home/gvuser/MPIversion/library/include -I/home/gvuser/intel/compilers_and_libraries_2017.2.174/linux/mkl/include/ -L/home/gvuser/intel/mkl/lib/intel64_lin -L/home/gvuser/MPIversion/artifacts -o MPIv2 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm -ldl -lmpi -std=c++11
#$(CC) $(CFLAGS) MPIv2.cpp -DMKL_ILP64 -m64 -I/home/gvuser/MPIversion/library/include -I/home/gvuser/MPIversion/intel/compilers_and_libraries_2017.2.174/linux/mkl/include/ -L/home/gvuser/MPIversion/intel/mkl/lib/intel64_lin -L/home/gvuser/MPIversion/artifacts -o MPIv2 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lmpi -std=c++11
#$(CC) $(CFLAGS) MPIv2.cpp -I/home/gvuser/MPIversion/library/include -L/home/gvuser/MPIversion -L/home/gvuser/MPIversion/artifacts -o MPIv2 -fopenmp -lmpi -lpthread -lopenblas -std=c++17
#$(CC) $(CFLAGS) MPIv2.cpp -I/home/gvuser/MPIversion/library/include -L/home/gvuser/MPIversion -L/home/gvuser/MPIversion/artifacts -o MPIv2 -lmpi -lopenblas -fopenmp -std=c++14
HW: clean
	$(CC) $(CFLAGS) -DMKL_ILP64 -m64 -I/home/gvuser/MPIversion/library/include -I/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/include/ -L/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin -L/home/gvuser/MPIversion/artifacts MPIv2.cpp -o MPIv2 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lmpi -fopenmp -std=c++14

threaded: clean2
	$(CC) $(CFLAGS) -DMKL_ILP64 -m64 -I/home/gvuser/MPIversion/library/include -I/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/include/ -L/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin -L/home/gvuser/MPIversion/artifacts MPIv-multi-threaded.cpp -o MPIv3 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lmpi -fopenmp -std=c++14

clean:
	rm -f MPIv2

clean2:
	rm -f MPIv3

copy:
	cp MPIv.cpp MPIv.cu
#gvsu2017 is password for private key
#gvuser is the username