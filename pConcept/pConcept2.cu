//ANN proof of concept Cuda


//Ron Patrick - Capstone GVSU - Winter 2017

//MNIST here http://yann.lecun.com/exdb/mnist/

//If you don't want to install OpenMPI and Cuda toolkit 8.0,
//simply compile with g++ pConcept.cpp -o pConcept -std=c++11
//since the Makefile only works on my machine with Cuda and OpenMPI

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <bitset>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thrust/detail/config.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/detail/vector_base.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <mpi.h>
#include "helper_cuda.h"
#include "helper_string.h"
#include "intarray2bmp.hpp"
#include <cmath>
#include <numeric>
#include <limits.h>
#include <float.h>
#include <random>
#include <imebra/imebra.h>
using namespace std;
using namespace thrust;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define ULLI unsigned long long int
#define UNCHAR unsigned char

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr);

typedef thrust::tuple<ULLI, ULLI> uTuple;
typedef thrust::tuple<double, double> dTuple;
struct floatToDoubleFunctor : public thrust::unary_function<float,double> {
	__device__ double operator()(float t) {
		return (double)t;
	}
};

struct doRandomDoubles {
	__device__ double operator()(ULLI t) {
		thrust::default_random_engine defRandEngine;
		thrust::uniform_real_distribution<double> uniRealDist;
		defRandEngine.discard(t);
		return (uniRealDist(defRandEngine)*2.0)-1.0;
	}
};

struct forwardProp_functor : public thrust::unary_function<double,double> {
	double *connWeights;
	double in;
	ULLI ssize;
	forwardProp_functor(double _in, double  *_connWeights, ULLI _size) : in(_in), connWeights(_connWeights), ssize(_size) {}

	__device__ double operator()(double t) {
		double out=t;
		if(in) {
			//printf("ssize: %llu\n",ssize);
			for(int i=0;i<ssize;++i) {
				if(connWeights[i]) {
					out+=in*connWeights[i];
					//printf("i: %d t: %.15f out: %.15f size: %llu conn: %.15f\n",i,t,out,ssize,connWeights[i]);
				}
			}
		}
		return out;
	}
};
struct backwardProp_functor : public thrust::unary_function<double, double> {
	double *connWeights;
	double *netErr;
	ULLI ssize;
	backwardProp_functor(double *_connWeights, double *_netErr, ULLI _size) : connWeights(_connWeights), netErr(_netErr), ssize(_size){}
	__device__ double operator()(double t) {
		double output=t;
		for(int i=0;i<ssize;++i) {
			if(connWeights[i] && netErr[i]) {
				output+=connWeights[i]*netErr[i];
			}
		}
		return output;
	}
};

struct weightsUpdater : public thrust::unary_function<double, double> {
	double *connWeights;
	double netIJxlRate;
	ULLI ssize;
	weightsUpdater(double *_connWeights, double _lRate, double _netIJ, ULLI _size) : connWeights(_connWeights),
		netIJxlRate(_netIJ*_lRate), ssize(_size){}
	__device__ double operator()(double t) {
		double output=t;
		if(netIJxlRate) {
			for(int i=0;i<ssize;++i) {
				output-=connWeights[i]*netIJxlRate;
			}
		}
		return output;
	}
};

struct weightsHelper : public thrust::binary_function<double,double,double> {
  __device__ double operator()(double x, double y) { return x*(y*(1.0-y)); }
};

struct activationFunc : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		return 1.0 / (1.0 + exp(-t));
	}
};

struct derivativeFunc : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		return t*(1.0-t);
	}
};

template<typename T>
struct square {
	__device__ T operator()(const T& x) const { 
		return x * x;
	}
};

class neuralNet {
public:
	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix) : hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), epoch(0), learningRate(0.98) {
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);
	}

	void train(vector<vector<float>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, float RMSwant) {

		cout << "Setting up network...\n";
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;

		dataSetSize=pData.size();
		//dataSetSize=10;
		toDivideRMS=((double)dataSetSize)*(double)numOutputs;
		device_vector<float> data[dataSetSize];
		device_vector<double> labels[dataSetSize];
		float *temp;
		double *tempd;
		ULLI len=pData[0].size();
		ULLI llen=pLabels[0].size();
		for(int i=0;i<dataSetSize;++i) {
			temp=&pData[i][0];
			data[i]=device_vector<float>(temp, temp+len);
			tempd=&pLabels[i][0];
			labels[i]=device_vector<double>(tempd, tempd+llen);
		}
		//for(int i=0;i<pData[1].size();++i) {
		//	cout << "got: " << pData[1][i] << " expected: " << data[1][i] << endl;
		//}
		ULLI maxElement=*max_element(hiddenMatrix.begin(),hiddenMatrix.end());
		device_vector<double> net[layers];
		device_vector<double> netLocalDerivatives[layers];
		device_vector<double> netErrorSignals[layers];
		device_vector<double> connWeights[layers-1][maxElement];

		int index=0;
		for(auto h:hiddenMatrix) {
			net[index]=device_vector<double>(h);
			netLocalDerivatives[index]=device_vector<double>(h);
			netErrorSignals[index++]=device_vector<double>(h);
		}

		for(ULLI i=0;i<layers-1;++i) {
			for(ULLI j=0;j<maxElement;++j) {
				connWeights[i][j]=device_vector<double>(maxElement);
				thrust::transform(thrust::make_counting_iterator<ULLI>(0),thrust::make_counting_iterator<ULLI>(maxElement),connWeights[i][j].begin(),doRandomDoubles());
			}
		}
		//forwardProp_functor forwardProp(thrust::device_ptr<double>(net[0].data()), thrust::device_ptr<double>(connWeights[0][0].data()));

		cout << "Starting training...\n";

		ULLI hMany;
		for(int ii=0;ii<maxEpochs && RMSwanted<RMS;++ii) {
			RMS=0.0;
			for(int iii=0;iii<dataSetSize;++iii) {
				//printf("Item# %d\n",iii);

				//forward
				thrust::transform(data[iii].begin(),data[iii].end(),net[0].begin(),floatToDoubleFunctor());
				for(int i=1;i<layers;++i) {
					thrust::fill(net[i].begin(),net[i].end(),0.0);
					//auto iterBegin=thrust::make_transform_iterator(net[i].begin(),)
					hMany=hiddenMatrix[i];
					/*for(auto n:net[i]) {
						double temp=n;
						printf("net[%d] before: %.15f\n",i,temp);
					}*/	
					//printf("k: %d\n",hiddenMatrix[i-1]);
					for(int k=0;k<hiddenMatrix[i-1];++k) {
						thrust::transform(net[i].begin(),net[i].end(),net[i].begin(),forwardProp_functor(net[i-1][k],connWeights[i-1][k].data().get(),hMany));
					}
					/*for(auto n:net[i]) {
						double temp=n;
						printf("net[%d] after: %.15f\n",i,temp);
					}*/
					thrust::transform(net[i].begin(),net[i].end(),net[i].begin(),activationFunc());				
					thrust::transform(net[i].begin(),net[i].end(),netLocalDerivatives[i].begin(),derivativeFunc());
					//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(hiddenMatrix[i]),net[i].begin(),forwardProp);//forwardProp_functor(net[i].data().get(), connWeights[i-1].data().get()));
					/*ULLI index=0;
					for(auto t:net[0]) {
						cout << "index: " << index++ << ": " << t << endl;
					}
					exit(0);*/
					//thrust::transform(net[i].begin().net[i].end(),connWeights[i-1][])
					/*for(int j=0;j<hiddenMatrix[i];++j) {
						for(int k=0;k<hiddenMatrix[i-1];++k) {
							net[i][j]+=connWeights[i-1][k][j]*net[i-1][k];
						}
						net[i][j]=sigmoid(net[i][j]);
						netLocalDerivatives[i][j]=sigmoid_derivative(net[i][j]);
						//printf("net[%d][%d].output: %.15f derivative: %.15f\n",i,j,net[i][j].output,net[i][j].localDerivative);
					}*/
				}
				//exit(0);

				//backProp
				/*for(auto n:netErrorSignals[outputsIndex]) {
					double temp=n;
					printf("before %.15f\n",temp);
				}*/
				thrust::transform(net[outputsIndex].begin(),net[outputsIndex].end(),labels[iii].begin(),netErrorSignals[outputsIndex].begin(),thrust::minus<double>());
				/*for(auto n:netErrorSignals[outputsIndex]) {
					double temp=n;
					printf("after %.15f\n",temp);
				}
				exit(0);*/
				//thrust::transform(labels[iii].begin(),labels[iii].end(),net[outputsIndex].begin(),netErrorSignals[outputsIndex].begin(),thrust::minus<double>());
				thrust::transform(netErrorSignals[outputsIndex].begin(),netErrorSignals[outputsIndex].end(),netLocalDerivatives[outputsIndex].begin(),netErrorSignals[outputsIndex].begin(),thrust::multiplies<double>());
				RMS=thrust::transform_reduce(netErrorSignals[outputsIndex].begin(),netErrorSignals[outputsIndex].end(),square<double>(),RMS,thrust::plus<double>());
				//printf("RMS: %.15f\n",RMS);
				/*for(int i=0;i<numOutputs;++i) {
					double tempDouble=(net[outputsIndex][i]-(double)pLabels[iii][i])*netLocalDerivatives[outputsIndex][i];
					netErrorSignals[outputsIndex][i]=tempDouble;
					RMS+=tempDouble*tempDouble;
				}*/
				for(int i=layers-2;i>0;--i) {
					thrust::fill(netErrorSignals[i].begin(),netErrorSignals[i].end(),0.0);
					hMany=hiddenMatrix[i+1];
					for(int j=0;j<hiddenMatrix[i];++j) {
						thrust::transform(netErrorSignals[i].begin(),netErrorSignals[i].end(),netErrorSignals[i].begin(),backwardProp_functor(connWeights[i][j].data().get(),netErrorSignals[i+1].data().get(),hMany));
					}
					thrust::transform(netErrorSignals[i].begin(),netErrorSignals[i].end(),net[i].begin(),netErrorSignals[i].begin(),weightsHelper());
					for(int j=0;j<hiddenMatrix[i];++j) {
						thrust::transform(connWeights[i][j].begin(),connWeights[i][j].end(),connWeights[i][j].begin(),weightsUpdater(netErrorSignals[i+1].data().get(),learningRate,net[i][j],hMany));
					}
					/*for(int j=0;j<hiddenMatrix[i];++j) {
						netErrorSignals[i][j]=0.0;
						for(int k=0;k<hiddenMatrix[i+1];++k) {
							netErrorSignals[i][j]+=connWeights[i][j][k]*netErrorSignals[i+1][k];
							connWeights[i][j][k]-=learningRate*netErrorSignals[i+1][k]*net[i][j];
						}
						netErrorSignals[i][j]*=net[i][j]*(1.0-net[i][j]);
					}*/
				}
				hMany=hiddenMatrix[1];
				for(int j=0;j<hiddenMatrix[0];++j) {
					thrust::transform(connWeights[0][j].begin(),connWeights[0][j].end(),connWeights[0][j].begin(),weightsUpdater(netErrorSignals[1].data().get(),learningRate,net[0][j],hMany));
				}
			}
			RMS=sqrt(RMS/toDivideRMS);
			//RMS=abs(RMS/toDivideRMS);
			if(RMS<minRMS) {
				minRMS=RMS;
				//printf("\nminRMS Error: %.15f Iteration: %d",RMS,ii);
			}
			printf("current RMS: %.15f minRMS Error: %.15f Iteration: %d of %llu\r",RMS,minRMS,ii,maxEpochs);
			/*cout << endl << "Output: ";
			for(auto o:net[outputsIndex]) {
				double temp=o;
				printf("%.5f ",temp);
			}
			cout << endl;*/
		}
		cout << endl;
		for(int ii=0;ii<dataSetSize;++ii) {
			thrust::transform(data[ii].begin(),data[ii].end(),net[0].begin(),floatToDoubleFunctor());
			cout << "Inputs: ";
			for(auto i:net[0]) {
				double temp=i;
				printf("%.15f ",temp);
			}//*/
			for(int i=1;i<layers;++i) {
				thrust::fill(net[i].begin(),net[i].end(),0.0);
				hMany=hiddenMatrix[i];
				for(int k=0;k<hiddenMatrix[i-1];++k) {
					thrust::transform(net[i].begin(),net[i].end(),net[i].begin(),forwardProp_functor(net[i-1][k],connWeights[i-1][k].data().get(),hMany));
				}
				thrust::transform(net[i].begin(),net[i].end(),net[i].begin(),activationFunc());
			}
			cout << endl << "Output: ";
			for(auto o:net[outputsIndex]) {
				double temp=o;
				printf("%.15f ",temp);
			}
			cout << endl;		
		}

	}
private:
	ULLI epoch, maxElement, layers, maxEpochs;
	int outputsIndex, dataSetSize, numInputs, numOutputs;
	double RMS, minRMS, toDivideRMS, RMSwanted, learningRate;
	vector<int> hiddenMatrix;

	vector<vector<float>> neuralNet_weights_host;

	__host__ __device__ double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	__host__ __device__ double sigmoid_derivative(double x) {
		return x*(1.0-x);
	}	
};

#define BITS 3
void doMain(int my_rank, string hostname, int num_nodes) {

	vector<int> hiddenMatrix;
	//hiddenMatrix.push_back(BITS+(BITS/2));
	//hiddenMatrix.push_back(BITS+(BITS/2));
	for(int i=0;i<1;++i) {
		hiddenMatrix.push_back(BITS+(BITS/2));
		//hiddenMatrix.push_back(12);
	}
	neuralNet test(BITS,BITS,hiddenMatrix);
	//vector<vector<neuron_t>> countingTest;
	//vector<vector<double>> countingLabels;
	int size=pow(2,BITS);
	vector<vector<float>> countingTest;
	vector<vector<double>> countingLabels;
	for(int i=0;i<size;++i) {
		countingTest.push_back(vector<float>(BITS));
		countingLabels.push_back(vector<double>(BITS,0.0));
		//countingLabels[i]=vector<double>(BITS,0.0);
		//countingTest[i]=vector<neuron_t>(BITS);
		for(int j=0;j<BITS;++j) {
			//countingTest.back()[j].output=(double)bitset<BITS>(i)[(BITS-1)-j];
			//countingLabels.back()[j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
			countingTest[i][j]=(float)bitset<BITS>(i)[(BITS-1)-j];
			countingLabels[i][j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
		}
	}
	test.train(countingTest,countingLabels,1000000,0.00001);

	/*	
	vector<int> hiddenMatrix;
	//hiddenMatrix.push_back(784+(784/2));
	//hiddenMatrix.push_back(784+(784/2));
	hiddenMatrix.push_back(13);

	vector<vector<float>> testData(10000);
	ReadMNIST_float("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<float>> trainData(60000);
	ReadMNIST_float("train-images.idx3-ubyte",60000,784,trainData);
	vector<vector<double>> testLabels(10000);
	vector<vector<double>> trainLabels(60000);
	ifstream file("t10k-labels.idx1-ubyte",ios::binary);
	if(file.is_open()) {
		int placeHolder=0;
		file.read((char*)&placeHolder,sizeof(placeHolder));
		file.read((char*)&placeHolder,sizeof(placeHolder));
		for(int i=0;i<10000;++i) {
			UNCHAR temp=0;
			file.read((char*)&temp,1);
			for(UNCHAR j=0;j<10;++j) {
				if(j==temp) {
					testLabels[i].push_back(1.0);
				} else {
					testLabels[i].push_back(0.0);
				}
			}
		}
		file.close();
	}
	ifstream file2("train-labels.idx1-ubyte",ios::binary);
	if(file2.is_open()) {
		int placeHolder=0;
		file2.read((char*)&placeHolder,sizeof(placeHolder));
		file2.read((char*)&placeHolder,sizeof(placeHolder));
		for(int i=0;i<60000;++i) {
			UNCHAR temp=0;
			file2.read((char*)&temp,1);
			for(UNCHAR j=0;j<10;++j) {
				if(j==temp) {
					trainLabels[i].push_back(1.0);
				} else {
					trainLabels[i].push_back(0.0);
				}
			}
		}
		file2.close();
	}

	//vector<UNCHAR> temp;
	//for(auto p:trainData[1]) {
	//	temp.push_back((UNCHAR)(p*255.0f));
	//	cout << (int)temp.back() << endl;
	//}
	//UNCHAR* t=&temp[0];
	//intarray2bmp::intarray2bmp("outputtest.bmp",t,(UNCHAR)28,(UNCHAR)28,(UNCHAR)0,(UNCHAR)255);

	neuralNet go(784,10,hiddenMatrix);
	go.train(trainData,trainLabels,1000000,0.0001);//*/
}

int main(int argc, char *argv[]) {
	//Cuda doesn't like this first one
	//srandom(time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());
	srand((unsigned int)time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());

	int my_rank, num_nodes;
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    char my_host[100];
    gethostname(my_host, 100);
    string hostname=string(my_host);

	if(hostname=="quattro.cis.gvsu.edu" || hostname=="ULaptop" || hostname=="Usolid") {

		ULLI totalCudaMem=0;
		size_t totalFreeCudaMem;
		size_t mem_tot_0 = 0;
		size_t mem_free_0 = 0;
		int device_num;

		//This code is from deviceQuery.cpp as seen in /usr/local/cuda-8.0/samples
		int deviceCount = 0;
	    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	    if(deviceCount) {
	    	cudaGetDevice(&device_num);
			cudaMemGetInfo(&mem_free_0, & mem_tot_0);
			totalFreeCudaMem=mem_free_0;
			ULLI dmask=1;
			ULLI maxDiv=1;
			for(int i=0;i<sizeof(ULLI)*8;++i) {
				if(dmask&totalFreeCudaMem) {
					maxDiv=dmask/2;
				}
				dmask<<=1;
			}
			maxDiv/=8;
		}
		int dev=0;
		for (dev = 0; dev < deviceCount; ++dev)	 {
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);
			printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
			if(!dev) {
				char msg[256];
			    sprintf(msg, "  Total amount of global memory: %.0f MBytes (%llu bytes)\n", 
			    	(float)deviceProp.totalGlobalMem/1048576.0f, 
			    	(ULLI) deviceProp.totalGlobalMem);
			    totalCudaMem=(ULLI)deviceProp.totalGlobalMem;
			    printf("%s", msg);
			    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP: %d CUDA Cores\n", 
			    	deviceProp.multiProcessorCount, 
			    	_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			    	_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
			    printf("  GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n\n", 
			    	deviceProp.clockRate * 1e-3f, 
			    	deviceProp.clockRate * 1e-6f);
		        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);		    
				printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
		               deviceProp.maxThreadsDim[0],
		               deviceProp.maxThreadsDim[1],
		               deviceProp.maxThreadsDim[2]);
		        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
		               deviceProp.maxGridSize[0],
		               deviceProp.maxGridSize[1],
		               deviceProp.maxGridSize[2]);
		    }
		    cout << endl;
		}

		if(totalCudaMem) {
			cout << string(my_host) << ": total Cuda Memory: " << totalCudaMem << endl;
		}
	}

	doMain(my_rank, hostname, num_nodes);

	MPI_Finalize();	

	return 0;
}

int ReverseInt(int i) {
    UNCHAR ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr) {
    //arr.resize(NumberOfImages,DataOfAnImage);
    ifstream file(filename,ios::binary);
    if (file.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i) {
        	arr[i]=vector<float>();
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    UNCHAR temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    //arr[i][(n_rows*r)+c]= ((float)temp)/256.0f;
                    //cout << "from read: " << ((float)temp)/256.0f << ": ";
                    arr[i].push_back(((float)temp)/256.0f);
                    //cout << "from arr: " << arr[i].back() << endl;
                }
            }
        }
    }
    file.close();
}