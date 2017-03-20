//Artifical Neural Network with Cuda and Cublas matrix version

//Ron Patrick - Capstone GVSU - Winter 2017

#include <signal.h>
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
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
//#include <mpi.h>
#include "helper_cuda.h"
#include "helper_string.h"
#include <cmath>
#include <numeric>
#include <limits.h>
#include <float.h>
#include <random>
//#include "imebra/imebra.h"
#include <cublasXt.h>
#include <cublas_v2.h>
#include <curand.h>
#include <pthread.h>
#include "cudnn.h"
using namespace std;
using namespace thrust;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define ULLI unsigned long long int
#define UNCHAR unsigned char
#define INPUT 0
#define OUTPUT 1
#define HIDDEN 2
#ifndef doMNISTprob
#define doMNISTprob true
#endif
#ifndef doBinaryProb
#define doBinaryProb false
#endif
#ifndef BITS
#define BITS 5
#endif

int memoryTracker=0;
bool showCorrectNumTrain=false;
int showInterval=0;
pthread_mutex_t crmutex = PTHREAD_MUTEX_INITIALIZER;
bool threadExit=false;
int waitTime;
static pthread_barrier_t barrier;
static pthread_barrier_t barrier2;

void ReadMNIST_double(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr);
void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr);
void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end);
void print_matrix(device_vector<double> &A, int nr_rows_A, int nr_cols_A);

typedef thrust::tuple<ULLI, ULLI> uTuple;
typedef thrust::tuple<double, double> dTuple;
typedef thrust::tuple<ULLI, double, double> tTuple;
typedef thrust::device_vector<double>::iterator doubleIterator;
typedef thrust::tuple<doubleIterator, doubleIterator> iterTuple;
typedef thrust::zip_iterator<iterTuple> zipIterator;

void ctrlchandler(int sig) {
	printf("\nTrying to exit...\n");
	threadExit=true;
}

void memTracker(int in, bool printIt) {
	memoryTracker+=in;
	if(printIt) {
		cout << "Cuda memory tracker: Using(bytes): " << memoryTracker << " ";
		cout << "(Kb): " << (memoryTracker/1024) << " ";
		cout << "(Mb): " << ((memoryTracker/1024)/1024) << endl;
	}
}

struct floatToDoubleFunctor : public thrust::unary_function<float,double> {
	__device__ double operator()(float t) {
		return (double)t;
	}
};
struct fix_random_numbers : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		return (((double)t)*2.0)-1.0;
	}
};
struct fix_random_numbers_f : public thrust::unary_function<float, float> {
	__device__ float operator()(float t) {
		return (((float)t)*2.0f)-1.0f;
	}
};

void random_floats(float *A, int rowsA, int colsA) {
    curandGenerator_t cg;
    curandCreateGenerator(&cg, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(cg, (unsigned long long) clock());
    curandGenerateUniform(cg, A, rowsA * colsA);
}

void random_doubles(double *A, int rowsA, int colsA) {
    curandGenerator_t cg;
    curandCreateGenerator(&cg, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(cg, (unsigned long long) clock());
    curandGenerateUniformDouble(cg, A, rowsA * colsA);
}

struct update_w : public thrust::unary_function<int, void> {
	double *weights;
	double *newW;
	double lRate;
	update_w(double *w, double *_newW, double lr) : weights(w), newW(_newW), lRate(lr){}
	__device__ void operator()(int t) {
		double local=weights[t];
		double local2=lRate;
		double local3=newW[t];
		double local4=local-local2*local3;
		weights[t]=local4;
		//weights[t]=weights[t]-lRate*newW[t];
	}
};
struct update_b : public thrust::unary_function<int, void> {
	double *biases;
	double *newB;
	double lRate;
	update_b(double *b, double *_newB, double lr) : biases(b), newB(_newB), lRate(lr){}
	__device__ void operator()(int t) {
		double local=biases[t];
		double local2=lRate;
		double local3=newB[t];
		double local4=local-local2*local3;
		biases[t]=local4;
		//biases[t]=biases[t]-lRate*newB[t];
	}
};

struct update_wf : public thrust::unary_function<int, void> {
	float *weights;
	float *newW;
	float lRate;
	update_wf(float *w, float *_newW, float lr) : weights(w), newW(_newW), lRate(lr){}
	__device__ void operator()(int t) {
		float local=weights[t];
		float local2=lRate;
		float local3=newW[t];
		float local4=local-local2*local3;
		weights[t]=local4;
		//weights[t]=weights[t]-lRate*newW[t];
	}
};
struct update_bf : public thrust::unary_function<int, void> {
	float *biases;
	float *newB;
	float lRate;
	update_bf(float *b, float *_newB, float lr) : biases(b), newB(_newB), lRate(lr){}
	__device__ void operator()(int t) {
		float local=biases[t];
		float local2=lRate;
		float local3=newB[t];
		float local4=local-local2*local3;
		biases[t]=local4;
		//biases[t]=biases[t]-lRate*newB[t];
	}
};

template<typename T>
struct square {
	__device__ T operator()(const T& x) const { 
		return x * x;
	}
};
struct sigmoid_devrivative : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		double tt=1.0/(1.0+exp(-t));
		return tt*(1.0-tt);
	}
};
struct sigmoid : public thrust::unary_function<double, double> {
	sigmoid(){}
	__device__ double operator()(double t) {
		return 1.0 / (1.0 + exp(-t));
	}
};
struct exp_double : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		return exp(t);
	}
};

struct forwardFeed_helper : public thrust::unary_function<int, double> {
	double *inputs;
	double *biases;
	forwardFeed_helper(){}
	forwardFeed_helper(double *_inputs, double* _biases) : inputs(_inputs), biases(_biases){}
	__device__ double operator()(int t) {
	//__device__ double operator()(thrust::tuple<double, double> t) {
		double local=inputs[t];
		local+=biases[t];
		inputs[t]=local;
		return 1.0/(1.0+exp(-local));
	}
};
struct backProp_helper : public thrust::unary_function<int, double> {
	double *innerDelta;
	double *inputs;
	backProp_helper(){}
	backProp_helper(double* _innerDelta, double *_inputs) : innerDelta(_innerDelta), inputs(_inputs){}
	__device__ double operator()(int t) {
		double local=1.0/(1.0+exp(-inputs[t]));
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
	/*__device__ double operator()(thrust::tuple<double, double> t) {
		double local=1.0/(1.0+exp(-thrust::get<0>(t)));
		local=local*(1.0-local);
		return thrust::get<1>(t)*local;
	}*/
};
struct backProp_helper2 : public thrust::unary_function<double, double> {
	double *outputs;
	double *innerDelta;
	backProp_helper2(){}
	backProp_helper2(double *_outputs, double* _innerDelta) : innerDelta(_innerDelta), outputs(_outputs){}
	__device__ double operator()(int t) {
		double local=outputs[t];
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
	/*__device__ double operator()(thrust::tuple<double, double> t) {
		double local=thrust::get<0>(t);
		local=local*(1.0-local);
		return thrust::get<1>(t)*local;
	}*/
};
struct output_helper : public thrust::unary_function<int, double> {
	double *inputs;
	double *outputs;
	double *labels;
	double *innerDelta;
	output_helper(double *_outputs, double *_inputs, double* _innerDelta, double* _labels) : outputs(_outputs), inputs(_inputs), innerDelta(_innerDelta), labels(_labels){}
	__device__ double operator()(int t) {
		double local=outputs[t]-labels[t];
		double local2=1.0/(1.0+exp(-inputs[t]));
		local2=local2*(1.0-local2);
		return local2*local;
	}
};

struct forwardFeed_helperf : public thrust::unary_function<int, float> {
	float *inputs;
	float *biases;
	forwardFeed_helperf(){}
	forwardFeed_helperf(float *_inputs, float* _biases) : inputs(_inputs), biases(_biases){}
	__device__ float operator()(int t) {
	//__device__ float operator()(thrust::tuple<float, float> t) {
		float local=inputs[t];
		local+=biases[t];
		inputs[t]=local;
		return 1.0/(1.0+exp(-local));
	}
};
struct backProp_helperf : public thrust::unary_function<int, float> {
	float *innerDelta;
	float *inputs;
	backProp_helperf(){}
	backProp_helperf(float* _innerDelta, float *_inputs) : innerDelta(_innerDelta), inputs(_inputs){}
	__device__ float operator()(int t) {
		float local=1.0/(1.0+exp(-inputs[t]));
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
	/*__device__ float operator()(thrust::tuple<float, float> t) {
		float local=1.0/(1.0+exp(-thrust::get<0>(t)));
		local=local*(1.0-local);
		return thrust::get<1>(t)*local;
	}*/
};
struct backProp_helper2f : public thrust::unary_function<float, float> {
	float *outputs;
	float *innerDelta;
	backProp_helper2f(){}
	backProp_helper2f(float *_outputs, float* _innerDelta) : innerDelta(_innerDelta), outputs(_outputs){}
	__device__ float operator()(int t) {
		float local=outputs[t];
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
	/*__device__ float operator()(thrust::tuple<float, float> t) {
		float local=thrust::get<0>(t);
		local=local*(1.0-local);
		return thrust::get<1>(t)*local;
	}*/
};
struct output_helperf : public thrust::unary_function<int, float> {
	float *inputs;
	float *outputs;
	float *labels;
	float *innerDelta;
	output_helperf(float *_outputs, float *_inputs, float* _innerDelta, float* _labels) : outputs(_outputs), inputs(_inputs), innerDelta(_innerDelta), labels(_labels){}
	__device__ float operator()(int t) {
		float local=outputs[t]-labels[t];
		float local2=1.0/(1.0+exp(-inputs[t]));
		local2=local2*(1.0-local2);
		return local2*local;
	}
};

class NN_layerf {
public:

	device_vector<float> atNeuronOutputs;
	device_vector<float> atNeuronInputs;
	device_vector<float> weightsMatrix;
	device_vector<float> biases;
	device_vector<float> outerDeltaB;
	device_vector<float> outerDeltaW;
	device_vector<float> innerDeltaB;
	device_vector<float> innerDeltaW;

	NN_layerf(){}
	NN_layerf(int sizeThis, int sizeNext, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext) {

		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
		/*if(type!=OUTPUT) {
			weightsMatrix=device_vector<float>(allW);
		}
		if(type!=INPUT) {
			biases=device_vector<float>(allN);
		}*/
	}
	NN_layerf(int sizeThis, int sizeNext, int pBatchSize, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext), batchSize(pBatchSize) {

		setupLayer(true);
	}

	void setupLayer(bool newLayer) {
		atNeuronOutputs=device_vector<float>(batchSize*thisSize,0.0);
		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
		memTracker(allN*8,false);
		counterN=device_vector<int>(allN,0.0);
		counterW=device_vector<int>(allW,0.0);
		thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(allN),counterN.begin(),counterN.begin(),thrust::plus<int>());
		thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(allW),counterW.begin(),counterW.begin(),thrust::plus<int>());
		memTracker(allN*sizeof(int),false);
		memTracker(allW*sizeof(int),false);
		if(newLayer) {
			if(type!=INPUT) {
				atNeuronInputs=device_vector<float>(batchSize*thisSize,0.0);
				memTracker(allN*8,false);
				biases=device_vector<float>(thisSize*batchSize,0.0);
				memTracker(allN*8*3,false);
				outerDeltaB=device_vector<float>(allN,0.0);
				innerDeltaB=device_vector<float>(allN,0.0);
			} else {
				cudaFree(&atNeuronInputs);
				cudaFree(&biases);
				cudaFree(&outerDeltaB);
				cudaFree(&innerDeltaB);
			}
			if(type!=OUTPUT) {
				weightsMatrix=device_vector<float>(thisSize*nextSize);
				memTracker(allW*8*3,false);
				outerDeltaW=vector<float>(allW,0.0);
				innerDeltaW=vector<float>(allW,0.0);
				random_floats(thrust::raw_pointer_cast(&weightsMatrix[0]),thisSize,nextSize);
				thrust::transform(weightsMatrix.begin(),weightsMatrix.end(),weightsMatrix.begin(),fix_random_numbers_f());
				cout << "thisSize: " << thisSize << " nextSize: " << nextSize << " thisSize*nextSize: " << (thisSize*nextSize) << endl;
			} else {
				cudaFree(&weightsMatrix);
				cudaFree(&outerDeltaW);
				cudaFree(&innerDeltaW);
			}
		} else {
			if(type!=INPUT) {
				atNeuronInputs=device_vector<float>(batchSize*thisSize,0.0);
			} else {
				cudaFree(&atNeuronInputs);
				cudaFree(&biases);
			}
			if(type==OUTPUT) {
				cudaFree(&weightsMatrix);
			}
		}
	}

	int type, thisSize, nextSize, batchSize, allW, allN;
	device_vector<int> counterN;
	device_vector<int> counterW;
};

class NN_layer {
public:

	device_vector<double> atNeuronOutputs;
	device_vector<double> atNeuronInputs;
	device_vector<double> weightsMatrix;
	device_vector<double> biases;
	device_vector<double> outerDeltaB;
	device_vector<double> outerDeltaW;
	device_vector<double> innerDeltaB;
	device_vector<double> innerDeltaW;

	NN_layer(){}
	NN_layer(int sizeThis, int sizeNext, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext) {

		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
		/*if(type!=OUTPUT) {
			weightsMatrix=device_vector<double>(allW);
		}
		if(type!=INPUT) {
			biases=device_vector<double>(allN);
		}*/
	}
	NN_layer(int sizeThis, int sizeNext, int pBatchSize, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext), batchSize(pBatchSize) {

		setupLayer(true);
	}

	void setupLayer(bool newLayer) {
		atNeuronOutputs=device_vector<double>(batchSize*thisSize,0.0);
		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
		memTracker(allN*8,false);
		counterN=device_vector<int>(allN,0.0);
		counterW=device_vector<int>(allW,0.0);
		thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(allN),counterN.begin(),counterN.begin(),thrust::plus<int>());
		thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(allW),counterW.begin(),counterW.begin(),thrust::plus<int>());
		memTracker(allN*sizeof(int),false);
		memTracker(allW*sizeof(int),false);
		if(newLayer) {
			if(type!=INPUT) {
				atNeuronInputs=device_vector<double>(batchSize*thisSize,0.0);
				memTracker(allN*8,false);
				biases=device_vector<double>(thisSize*batchSize,0.0);
				memTracker(allN*8*3,false);
				outerDeltaB=device_vector<double>(allN,0.0);
				innerDeltaB=device_vector<double>(allN,0.0);
			} else {
				cudaFree(&atNeuronInputs);
				cudaFree(&biases);
				cudaFree(&outerDeltaB);
				cudaFree(&innerDeltaB);
			}
			if(type!=OUTPUT) {
				weightsMatrix=device_vector<double>(thisSize*nextSize);
				memTracker(allW*8*3,false);
				outerDeltaW=vector<double>(allW,0.0);
				innerDeltaW=vector<double>(allW,0.0);
				random_doubles(thrust::raw_pointer_cast(&weightsMatrix[0]),thisSize,nextSize);
				thrust::transform(weightsMatrix.begin(),weightsMatrix.end(),weightsMatrix.begin(),fix_random_numbers());
				cout << "thisSize: " << thisSize << " nextSize: " << nextSize << " thisSize*nextSize: " << (thisSize*nextSize) << endl;
			} else {
				cudaFree(&weightsMatrix);
				cudaFree(&outerDeltaW);
				cudaFree(&innerDeltaW);
			}
		} else {
			if(type!=INPUT) {
				atNeuronInputs=device_vector<double>(batchSize*thisSize,0.0);
			} else {
				cudaFree(&atNeuronInputs);
				cudaFree(&biases);
			}
			if(type==OUTPUT) {
				cudaFree(&weightsMatrix);
			}
		}
	}

	int type, thisSize, nextSize, batchSize, allW, allN;
	device_vector<int> counterN;
	device_vector<int> counterW;
};

struct idLink {
    int whichThread;
    int interval;
    device_vector<double> *data;
    device_vector<double> *labels;
    vector<NN_layer> *NNlayersQ;
    vector<int> *hiddenMatrix;
    double learningRate;
    int batchSize;
    cublasHandle_t handle;
};

void *fourthThread(void *thread_parm) {
	idLink data=*((idLink*) thread_parm);
	int myID=data.whichThread;
	int myDev=myID;//3-myID;
	//if(myID==1){myDev=1;}

	if(myDev) {
		cudaSetDevice(myDev);
		cudaDeviceEnablePeerAccess(0,0);//cudaDeviceEnablePeerAccess ( int  peerDevice, unsigned int  flags )
	} else {
		cudaDeviceEnablePeerAccess(1,0);
	}
	//cout << "myID started: " << myID << endl;
	cublasHandle_t handle;//=data.handle;
	cublasCreate(&handle);

	int howMany=data.interval;
	vector<int> hiddenMatrix=*data.hiddenMatrix;
	int layers=hiddenMatrix.size();
	int outputsIndex=layers-1;
	int batchSize=data.batchSize;
	int numOutputs=hiddenMatrix[outputsIndex];
	int mOut, ii, mPlus, nextSize, prevSize, thisSize;
	device_vector<double> *which;
	bool gotTime=false;
	high_resolution_clock::time_point startTime, endTime;
	int timeCountDown=10;

	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	//double toDivideRMS=data.learningRate/(double)batchSize;
	while(!threadExit) {

		for(int i=0;i<outputsIndex;++i) {
			ii=i+1;
			thrust::fill((*data.NNlayersQ)[ii].outerDeltaB.begin(),(*data.NNlayersQ)[ii].outerDeltaB.end(),0.0);
			thrust::fill((*data.NNlayersQ)[i].outerDeltaW.begin(),(*data.NNlayersQ)[i].outerDeltaW.end(),0.0);
		}//*/

		for(int h=0;h<howMany;++h) {
			//cout << "myID: " << myID << " howMany: " << howMany << "\n";

			if(!myID && !gotTime && !timeCountDown) {
				startTime=high_resolution_clock::now();
			}

			//forward propagation
			which=data.data;
			for(int i=0;i<outputsIndex;++i) {
				//cout << "myID: " << myID << " here\n";
				ii=i+1;
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[ii];
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, (*data.NNlayersQ)[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, (*data.NNlayersQ)[ii].atNeuronInputs.data().get(), nextSize);
				thrust::transform((*data.NNlayersQ)[ii].counterN.begin(),(*data.NNlayersQ)[ii].counterN.end(),(*data.NNlayersQ)[ii].atNeuronOutputs.begin(),forwardFeed_helper((*data.NNlayersQ)[ii].atNeuronInputs.data().get(),(*data.NNlayersQ)[ii].biases.data().get()));
				which=&(*data.NNlayersQ)[ii].atNeuronOutputs;
			}

			//Backward propagation
			mOut=outputsIndex-1;
			mPlus=outputsIndex;
			prevSize=hiddenMatrix[mOut];
			thrust::transform((*data.NNlayersQ)[outputsIndex].counterN.begin(),(*data.NNlayersQ)[outputsIndex].counterN.end(),(*data.NNlayersQ)[outputsIndex].innerDeltaB.begin(),output_helper((*data.NNlayersQ)[outputsIndex].atNeuronOutputs.data().get(),(*data.NNlayersQ)[outputsIndex].atNeuronInputs.data().get(),(*data.NNlayersQ)[outputsIndex].innerDeltaB.data().get(),(*data.labels).data().get()));
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numOutputs, prevSize, batchSize, alpha, (*data.NNlayersQ)[outputsIndex].innerDeltaB.data().get(), numOutputs, (*data.NNlayersQ)[mOut].atNeuronOutputs.data().get(), prevSize, beta, (*data.NNlayersQ)[mOut].innerDeltaW.data().get(), numOutputs);

			--mOut;
			for(int i=outputsIndex-1;i;--i) {
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[i+1];
				prevSize=hiddenMatrix[i-1];
				cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, batchSize, nextSize, alpha, (*data.NNlayersQ)[i].weightsMatrix.data().get(), nextSize, (*data.NNlayersQ)[i+1].innerDeltaB.data().get(), nextSize, beta, (*data.NNlayersQ)[i].innerDeltaB.data().get(), thisSize);
				if(i!=1) {
					thrust::transform((*data.NNlayersQ)[i].counterN.begin(),(*data.NNlayersQ)[i].counterN.end(),(*data.NNlayersQ)[i].innerDeltaB.begin(),backProp_helper2((*data.NNlayersQ)[i].atNeuronOutputs.data().get(),(*data.NNlayersQ)[i].innerDeltaB.data().get()));
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, (*data.NNlayersQ)[i].innerDeltaB.data().get(), thisSize, (*data.NNlayersQ)[i-1].atNeuronOutputs.data().get(), prevSize, beta, (*data.NNlayersQ)[mOut].innerDeltaW.data().get(), thisSize);
				} else {
					thrust::transform((*data.NNlayersQ)[i].counterN.begin(),(*data.NNlayersQ)[i].counterN.end(),(*data.NNlayersQ)[i].innerDeltaB.begin(),backProp_helper((*data.NNlayersQ)[i].innerDeltaB.data().get(),(*data.NNlayersQ)[i].atNeuronInputs.data().get()));
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, (*data.NNlayersQ)[i].innerDeltaB.data().get(), thisSize, (*data.data).data().get(), prevSize, beta, (*data.NNlayersQ)[mOut].innerDeltaW.data().get(), thisSize);
				}
				--mOut;
				--mPlus;
			}
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thrust::transform((*data.NNlayersQ)[ii].innerDeltaB.begin(),(*data.NNlayersQ)[ii].innerDeltaB.end(),(*data.NNlayersQ)[ii].outerDeltaB.begin(),(*data.NNlayersQ)[ii].outerDeltaB.begin(),thrust::plus<double>());
				thrust::transform((*data.NNlayersQ)[i].innerDeltaW.begin(),(*data.NNlayersQ)[i].innerDeltaW.end(),(*data.NNlayersQ)[i].outerDeltaW.begin(),(*data.NNlayersQ)[i].outerDeltaW.begin(),thrust::plus<double>());
			}//*/
			/*for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thrust::for_each((*data.NNlayersQ)[i].counterW.begin(),(*data.NNlayersQ)[i].counterW.end(),update_w((*data.NNlayersQ)[i].weightsMatrix.data().get(),(*data.NNlayersQ)[i].innerDeltaW.data().get(),toDivideRMS));
				thrust::for_each((*data.NNlayersQ)[ii].counterN.begin(),(*data.NNlayersQ)[ii].counterN.end(),update_b((*data.NNlayersQ)[ii].biases.data().get(),(*data.NNlayersQ)[ii].innerDeltaB.data().get(),toDivideRMS));
			}//*/
			/*for(int i=0;i<outputsIndex;++i) {
				thrust::for_each(make_counting_iterator(0),make_counting_iterator((*data.NNlayersQ)[i].allW),update_w((*data.NNlayersQ)[i].weightsMatrix.data().get(),(*data.outerDeltaW)[i].data().get(),toDivideRMS));
				thrust::for_each(make_counting_iterator(0),make_counting_iterator((*data.NNlayersQ)[i+1].allN),update_b((*data.NNlayersQ)[i+1].biases.data().get(),(*data.outerDeltaB)[i].data().get(),toDivideRMS));
			}//*/
			}
			if(!myID) {
				if(!gotTime) {
					if(timeCountDown) {
						--timeCountDown;
					} else {
						endTime=high_resolution_clock::now();
						double seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
						printf("Update time interval approximately %.5f seconds apart(%.5f seconds per)\n",(seconds*(double)howMany)+1.0,seconds);
						gotTime=true;
					}
				}
			}
		//}

		//cout << "thread: " << myID << " before barrier one\n";
		pthread_barrier_wait(&barrier);
		//cout << "thread: " << myID << " after barrier one\n";
		pthread_barrier_wait(&barrier2);
		//cout << "thread: " << myID << " after barrier two\n";

	}
	cublasDestroy(handle);
	free(thread_parm);
	pthread_exit(0);
}

struct divFour : public thrust::unary_function<double, double> {
	double what;
	divFour(double _what) : what(_what){}
	__device__ double operator()(double t) {
		return t/what;
	}
};

class neuralNet {
public:
	neuralNet(){}
	neuralNet(string _inFile) : inFile(_inFile) {
		cout << "Setting up network...\n";
		cublasCreate(&handle);
		loadState();
		cout << "Layers: ";
		for(auto h:hiddenMatrix) {
			cout << h << " ";
		}
		cout << "Batch size: " << batchSize << endl << endl;
	}

	//cublasXtHandle_t handlex;
	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize, int _numThreads) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize) {

		//cublasXtCreate(&handlex);
		//int dev[2]={0,1};
		//cublasXtDeviceSelect(handlex,2,dev);
		numThreads=_numThreads;
		cublasCreate(&handle);
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);

		for(int i=0;i<numThreads;++i) {
			NNlayersQ[i]=vector<NN_layer>(hiddenMatrix.size());
		}
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;
		cout << "Setting up network...\n";
		cout << "Layers: ";
		for(auto h:hiddenMatrix) {
			cout << h << " ";
		}
		batchSize=10000;
		cout << "Batch size: " << batchSize << endl << endl;
		/*int who;
		for(int i=3;i>0;--i) {
			if(i!=2) {
				cudaSetDevice(i);
				for(int j=3;j>0;--j) {
					if(i!=j) {
						cudaDeviceEnablePeerAccess(j,0);
						cudaDeviceCanAccessPeer(&who, i, j);//cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
						cout << "who returned: " << who << " for device: " << i << " peerDevice: " << j << endl;
					}
				}
			}
		}
		cudaSetDevice(3);//*/

		for(int j=0;j<numThreads;++j) {
			//cudaSetDevice(3-j);
			NNlayersQ[j][0]=NN_layer(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT);
			for(int i=1;i<outputsIndex;++i) {
				NNlayersQ[j][i]=NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN);
			}
			NNlayersQ[j][outputsIndex]=NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT);
		}
		for(int i=1;i<numThreads;++i) {
			for(int j=0;j<outputsIndex;++j) {
				thrust::copy(NNlayersQ[i-1][j].weightsMatrix.begin(),NNlayersQ[i-1][j].weightsMatrix.end(),NNlayersQ[i][j].weightsMatrix.begin());
				thrust::copy(NNlayersQ[i-1][j+1].biases.begin(),NNlayersQ[i-1][j+1].biases.end(),NNlayersQ[i][j+1].biases.begin());
			}
		}
	}

	void train_Quad(vector<vector<double>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, double lRate, vector<vector<double>> &pTestData, vector<vector<double>> &pTestLabels, bool vlRate) {

		if(!showInterval) {
			showInterval=10;
		}
		vector<int> bLabels;
		for(auto p:pLabels) {
			bLabels.push_back(std::max_element(p.begin(), p.end())-p.begin());
		}

		const double alf = 1;
		const double bet = 0;
		const double *alpha = &alf;
		const double *beta = &bet;

		if(lRate<0.0) {
			learningRate=0.05;
		} else {
			learningRate=lRate;
		}
		dataSetSize=60000;
		int testBatchSize=10000;

		int batchStart,batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		int testSetSize=pTestData.size();
		vector<int> btLabels;
		device_vector<double> testData[testSetSize/testBatchSize];
	
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back(std::max_element(p.begin(), p.end())-p.begin());
			}
		}

		device_vector<double> data[numThreads][dataSetSize/(batchSize*numThreads)];
		device_vector<double> labels[numThreads][dataSetSize/(batchSize*numThreads)];

		//Creating pre-made batches so I can simply copy them to layer[0]
		cout << "Making batches in memory...\n";
		int whichBatch=0;
		int iii=0;
		int itemsPerThread=dataSetSize/numThreads;
		int batchesEach;
		//cout << "itemsPerThread: " << itemsPerThread << endl;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			if(((iii+1)*itemsPerThread)==itemNum) {
				++iii;
				batchesEach=whichBatch;
				whichBatch=0;
			}
			//cout << "iii+1: " << (iii+1) << " itemNum: " << itemNum << " *:" << ((iii+1)*itemsPerThread) << " whichBatch: " << whichBatch << endl;
			data[iii][whichBatch]=vector<double>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			labels[iii][whichBatch]=vector<double>(batchSize*numOutputs);
			memTracker(numOutputs*batchSize*8,false);
			for(int b=0;b<batchSize;++b) {
				thrust::copy(pData[itemNum+b].begin(),pData[itemNum+b].end(),data[iii][whichBatch].begin()+batchStart);
				thrust::copy(pLabels[itemNum+b].begin(),pLabels[itemNum+b].end(),labels[iii][whichBatch].begin()+batchEnd);
				batchStart+=itemSize;
				batchEnd+=numOutputs;
			}
			++whichBatch;
		}
		whichBatch=0;
		for(int itemNum=0;itemNum<testSetSize;itemNum+=testBatchSize) {
			testData[whichBatch]=vector<double>(itemSize*testBatchSize);
			memTracker(itemSize*testBatchSize*8,false);
			batchStart=0;
			for(int j=0;j<testBatchSize;++j) {
				thrust::copy(pTestData[itemNum+j].begin(),pTestData[itemNum+j].end(),testData[whichBatch].begin()+batchStart);
				batchStart+=itemSize;
			}
			++whichBatch;
		}

		cout << "Starting training...\n";

		device_vector<double>::iterator iter;
		int position;
		int gotRight=0;
		//int numBatches=dataSetSize/batchSize;
		//toDivideRMS=learningRate/((double)numBatches*(double)batchSize);
		//toDivideRMS=learningRate/((double)batchSize*(double)showInterval);
		//toDivideRMS=learningRate/((double)batchSize*(double)num_nodes);//*(double)showInterval);
		toDivideRMS=learningRate/(double)batchSize;
		//toDivideRMS=learningRate/(double)showInterval;
		int maxGotRight=0, maxTestRight=-1, ii;
		device_vector<double> *which;
		double seconds, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		//int sInterval=showInterval;
		bool once=true;

		vector<pthread_t> threads;
		pthread_attr_t attr;
    	cpu_set_t cpus;
    	pthread_attr_init(&attr);

		divFour dThreads((double)numThreads);
		//multi_helper hTimes((double)numberOfProcessors);
		/*vector<double> tempDeltaB[outputsIndex];
		vector<double> tempDeltaW[outputsIndex];
		for(int i=0;i<outputsIndex;++i) {
			tempDeltaW[i]=vector<double>(NNlayersQ[0][i].allW,0.0);
			memTracker(NNlayersQ[0][i].allW*8,false);
			tempDeltaB[i]=vector<double>(NNlayersQ[0][i+1].allN,0.0);
			memTracker(NNlayersQ[0][i+1].allN*8,false);
		}*/
		memTracker(0,true);

		for(int epochNum=0;!threadExit && epochNum<maxEpochs && maxGotRight!=dataSetSize && maxTestRight!=testSetSize;++epochNum) {//epochNum+=sInterval) {
			startTime=high_resolution_clock::now();
			if(once) {
			    for(int j=0;j<numThreads;++j) {

     				CPU_ZERO(&cpus);
       				CPU_SET(j, &cpus);
       				pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

			        threads.push_back(pthread_t());
			        idLink *arg = (idLink*)malloc(sizeof(*arg));
			        (*arg).whichThread=j;
			        (*arg).data=&data[j][0];
			        (*arg).labels=&labels[j][0];
			        (*arg).hiddenMatrix=&hiddenMatrix;
			        (*arg).interval=batchesEach;
			        (*arg).NNlayersQ=&NNlayersQ[j];
			        (*arg).learningRate=learningRate;
			        (*arg).batchSize=batchSize;
			        (*arg).handle=handle;
			        pthread_create(&threads.at(j), &attr,  fourthThread, arg);
			    }
			    once=false;
			}
			pthread_barrier_wait(&barrier);
			//cout << "all got to here\n";
			for(int i=1;i<numThreads;++i) {
				for(int j=0;j<outputsIndex;++j) {
					ii=j+1;
					thrust::transform(NNlayersQ[0][j].outerDeltaW.begin(),NNlayersQ[0][j].outerDeltaW.end(),NNlayersQ[i][j].outerDeltaW.begin(),NNlayersQ[0][j].outerDeltaW.begin(),thrust::plus<double>());
					thrust::transform(NNlayersQ[0][ii].outerDeltaB.begin(),NNlayersQ[0][ii].outerDeltaB.end(),NNlayersQ[i][ii].outerDeltaB.begin(),NNlayersQ[0][ii].outerDeltaB.begin(),thrust::plus<double>());
					//thrust::transform(NNlayersQ[0][j].innerDeltaW.begin(),NNlayersQ[0][j].innerDeltaW.end(),NNlayersQ[i][j].innerDeltaW.begin(),NNlayersQ[0][j].innerDeltaW.begin(),thrust::plus<double>());
					//thrust::transform(NNlayersQ[0][ii].innerDeltaB.begin(),NNlayersQ[0][ii].innerDeltaB.end(),NNlayersQ[i][ii].innerDeltaB.begin(),NNlayersQ[0][ii].innerDeltaB.begin(),thrust::plus<double>());
				}
			}
			/*for(int j=0;j<outputsIndex;++j) {
				ii=j+1;
				thrust::for_each(NNlayersQ[0][j].counterW.begin(),NNlayersQ[0][j].counterW.end(),update_w(&NNlayersQ[0][j].weightsMatrix[0],&NNlayersQ[0][j].outerDeltaW[0],toDivideRMS));
				thrust::for_each(NNlayersQ[0][ii].counterN.begin(),NNlayersQ[0][ii].counterN.end(),update_b(&NNlayersQ[0][ii].biases[0],&NNlayersQ[0][ii].outerDeltaB[0],toDivideRMS));
			}//*/
			for(int i=0;i<numThreads;++i) {
				for(int j=0;j<outputsIndex;++j) {
					ii=j+1;
					//thrust::for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(NNlayersQ[i][j].weightsMatrix.data().get(),NNlayersQ[0][j].innerDeltaW.data().get(),toDivideRMS));
					//thrust::for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(NNlayersQ[i][ii].biases.data().get(),NNlayersQ[0][ii].innerDeltaB.data().get(),toDivideRMS));
					//thrust::for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(&NNlayersQ[i][j].weightsMatrix[0],&tempDeltaW[j][0],toDivideRMS));
					//thrust::for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(&NNlayersQ[i][ii].biases[0],&tempDeltaB[j][0],toDivideRMS));
					thrust::for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(NNlayersQ[i][j].weightsMatrix.data().get(),NNlayersQ[i][j].outerDeltaW.data().get(),toDivideRMS));
					thrust::for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(NNlayersQ[i][ii].biases.data().get(),NNlayersQ[i][ii].outerDeltaB.data().get(),toDivideRMS));
				}
			}//*/
			if(showIntervalCountDown) {
				--showIntervalCountDown;
				endTime=high_resolution_clock::now();
				seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
				totalTime+=seconds;
				pthread_barrier_wait(&barrier2);
				continue;
			} else {
				showIntervalCountDown=showInterval;
				/*for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					//MPI_Allreduce(&NNlayersQ[0][i].outerDeltaW[0],&NNlayersQ[1][i].outerDeltaW[0],NNlayersQ[0][i].allW,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][ii].outerDeltaB[0],&NNlayersQ[1][ii].outerDeltaB[0],NNlayersQ[0][ii].allN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][i].innerDeltaW[0],&tempDeltaW[i][0],NNlayersQ[0][i].allW,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][i+1].innerDeltaB[0],&tempDeltaB[i][0],NNlayersQ[0][i+1].allN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					thrust::transform(NNlayersQ[0][i].weightsMatrix.begin(),NNlayersQ[0][i].weightsMatrix.end(),NNlayersQ[2][i].weightsMatrix.begin(),hTimes);
					thrust::transform(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[2][ii].biases.begin(),hTimes);
					//MPI_Allreduce(&NNlayersQ[2][i].weightsMatrix[0],&NNlayersQ[1][i].weightsMatrix[0],NNlayersQ[0][i].allW,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[2][ii].biases[0],&NNlayersQ[1][ii].biases[0],NNlayersQ[0][ii].allN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					thrust::transform(NNlayersQ[1][i].weightsMatrix.begin(),NNlayersQ[1][i].weightsMatrix.end(),NNlayersQ[0][i].weightsMatrix.begin(),dThreads);
					thrust::transform(NNlayersQ[1][ii].biases.begin(),NNlayersQ[1][ii].biases.end(),NNlayersQ[0][ii].biases.begin(),dThreads);
				}*/
				/*for(int i=0;i<numThreads;++i) {
					for(int j=0;j<outputsIndex;++j) {
						ii=j+1;
						//thrust::for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(&NNlayersQ[i][j].weightsMatrix[0],&NNlayersQ[1][j].outerDeltaW[0],toDivideRMS));
						//thrust::for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(&NNlayersQ[i][ii].biases[0],&NNlayersQ[1][ii].outerDeltaB[0],toDivideRMS));
						thrust::copy(NNlayersQ[0][j].weightsMatrix.begin(),NNlayersQ[0][j].weightsMatrix.end(),NNlayersQ[i][j].weightsMatrix.begin());
						thrust::copy(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[i][ii].biases.begin());
					}
				}*/
			}
			gotRight=0;
			whichBatch=0;
			iii=0;
			for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {

				if(((iii+1)*itemsPerThread)==itemNum) {
					++iii;
					whichBatch=0;
				}

				//forward propagation
				which=&data[iii][whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayersQ[0][i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayersQ[0][ii].atNeuronInputs.data().get(), nextSize);
					thrust::transform(NNlayersQ[0][ii].counterN.begin(),NNlayersQ[0][ii].counterN.end(),NNlayersQ[0][ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayersQ[0][ii].atNeuronInputs.data().get(),NNlayersQ[0][ii].biases.data().get()));
					which=&NNlayersQ[0][ii].atNeuronOutputs;
				}

				batchStart=0;
				batchEnd=numOutputs;
				//printf("\nbatch starting at: %d\n",itemNum);
				for(int b=0;b<batchSize;++b) {
					iter = thrust::max_element(NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchEnd);
					position = iter - NNlayersQ[0][outputsIndex].atNeuronOutputs.begin();
					position -= batchStart;
					/*printf("output: %d expected: %d\n",position,bLabels[itemNum+b]);
					for(int ot=batchStart;ot<batchEnd;++ot) {
						double oo=NNlayersQ[0][outputsIndex].atNeuronOutputs[ot];
						printf("%.5f ",oo);
					}
					printf("\n");//*/
					if(position==bLabels[itemNum+b]) {
						++gotRight;
					}
					batchStart=batchEnd;
					batchEnd+=numOutputs;					
				}
				++whichBatch;
			}

			if(gotRight>maxGotRight){maxGotRight=gotRight;}
			printf("Epoch: %d-Got %d of %d-max right: %d-lRate: %.5f-",epochNum,gotRight,dataSetSize,maxGotRight,learningRate);
			gotRight=0;		
			whichBatch=0;
			for(int t=0;t<testSetSize;t+=testBatchSize) {
				which=&testData[whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, testBatchSize, thisSize, alpha, NNlayersQ[0][i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayersQ[0][ii].atNeuronInputs.data().get(), nextSize);
					thrust::transform(NNlayersQ[0][ii].counterN.begin(),NNlayersQ[0][ii].counterN.end(),NNlayersQ[0][ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayersQ[0][ii].atNeuronInputs.data().get(),NNlayersQ[0][ii].biases.data().get()));
					which=&NNlayersQ[0][ii].atNeuronOutputs;
				}

				batchStart=0;
				batchEnd=numOutputs;
				//printf("\nbatch starting at: %d\n",t);
				for(int b=0;b<testBatchSize;++b) {
					iter = thrust::max_element(NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchEnd);
					position = iter - NNlayersQ[0][outputsIndex].atNeuronOutputs.begin();
					position -= batchStart;
					/*printf("output: %d expected: %d\n",position,btLabels[t+b]);
					for(int ot=batchStart;ot<batchEnd;++ot) {
						double oo=NNlayersQ[0][outputsIndex].atNeuronOutputs[ot];
						printf("%.5f ",oo);
					}
					printf("\n");//*/
					if(position==btLabels[t+b]) {
						++gotRight;
					}
					batchStart=batchEnd;
					batchEnd+=numOutputs;
				}

				++whichBatch;
			}
			if(gotRight>maxTestRight){maxTestRight=gotRight;}

			endTime=high_resolution_clock::now();
			seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
			totalTime+=seconds;
			double errRate=(1.0-((double)gotRight/(double)testSetSize))*100.0;
			printf("Test-Got %d of %d-max right: %d-totTime: %.5f-eRate:%.5f perc\n",gotRight,testSetSize,maxTestRight,totalTime,errRate);
			if(testSetSize!=gotRight) {
				/*for(int i=1;i<numThreads;++i) {
					for(int j=0;j<outputsIndex;++j) {
						ii=j+1;
						thrust::copy(NNlayersQ[0][j].weightsMatrix.begin(),NNlayersQ[0][j].weightsMatrix.end(),NNlayersQ[i][j].weightsMatrix.begin());
						thrust::copy(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[i][ii].biases.begin());
					}
				}*/
			} else {
				threadExit=true;
			}
			pthread_barrier_wait(&barrier2);
		}
    	int status;
		void * result;
    	for (int i=0; i < numThreads; ++i) {
        	if ((status = pthread_join(threads.at(i), &result)) != 0) {
            	fprintf (stderr, "join error %d: %s\n", status, strerror(status));
	        }
    	}
		//saveStateQ("MPIv2-");
	}
	int numThreads;

	void saveStateQ(string outFile) {
		outFile+="Cuda-"+to_string(dataSetSize);
		cout << "Writing weights to file: " << outFile << endl;
		ofstream oFile(outFile, ios::binary|ios::out);
		if(oFile.is_open()) {
			oFile.write((char*)&epoch,sizeof(ULLI));
			oFile.write((char*)&layers,sizeof(ULLI));
			for(int i=0;i<hiddenMatrix.size();++i) {
				oFile.write((char*)&hiddenMatrix[i],sizeof(int));
			}
			oFile.write((char*)&batchSize,sizeof(int));
			oFile.write((char*)&learningRate,sizeof(double));
			for(int i=0;i<outputsIndex;++i) {
				for(int j=0;j<NNlayersQ[0][i].allW;++j) {
					double o=NNlayersQ[0][i].weightsMatrix[j];
					oFile.write((char*)&o,sizeof(double));
				}
			}
			for(int i=1;i<layers;++i) {
				for(int j=0;j<NNlayersQ[0][i].allN;++j) {
					double o=NNlayersQ[0][i].biases[j];
					oFile.write((char*)&o,sizeof(double));
				}
			}
			oFile.close();
		}
		cout << "Done\n";
	}

	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize) {

		//cublasXtCreate(&handlex);
		//int dev[3]={1,2,3};
		//cublasXtDeviceSelect(handlex,3,dev);
		if(batchSize<100) {
			batchSize=10000;
		}
		cublasCreate(&handle);
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);

		NNlayers=vector<NN_layer>(hiddenMatrix.size());
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;
		cout << "Setting up network...\n";
		cout << "Layers: ";
		for(auto h:hiddenMatrix) {
			cout << h << " ";
		}
		cout << "Batch size: " << batchSize << endl << endl;

		NNlayers[0]=NN_layer(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT);
		for(int i=1;i<outputsIndex;++i) {
			NNlayers[i]=NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN);
		}
		NNlayers[outputsIndex]=NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT);
	}

	void train_MatMul(vector<vector<float>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, double lRate, vector<vector<float>> &pTestData, vector<vector<double>> &pTestLabels, bool vlRate) {

		if(!showInterval) {
			showInterval=10;
		}
		vector<UNCHAR> bLabels;
		for(auto p:pLabels) {
			bLabels.push_back((UNCHAR)(thrust::max_element(p.begin(), p.end())-p.begin()));
		}

		if(lRate<0.0) {
			learningRate=0.05;
		} else {
			learningRate=lRate;
		}
		if(!doDataSetSize) {
			doDataSetSize=60000;
		}
		dataSetSize=doDataSetSize;
		const double alf = 1;
		const double bet = 0;
		const double *alpha = &alf;
		const double *beta = &bet;
		int batchStart,batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		int testSetSize=pTestData.size();
		vector<UNCHAR> btLabels;
		device_vector<double> testData[testSetSize/batchSize];
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back((UNCHAR)(thrust::max_element(p.begin(), p.end())-p.begin()));
			}
		} else {
			cudaFree(&testData);
		}
		int numBatches=dataSetSize/batchSize;
		device_vector<double> data[numBatches];
		device_vector<double> labels[numBatches];

		//float *temp;
		//double *tempd;
		//ULLI len=pData[0].size();
		//ULLI llen=pLabels[0].size();
		/*for(int i=0;i<dataSetSize;++i) {
			temp=&pData[i][0];
			dataTemp[i]=device_vector<float>(temp, temp+len);
			tempd=&pLabels[i][0];
			labelsTemp[i]=device_vector<double>(tempd, tempd+llen);
		}*/

		//Creating pre-made batches so I can simply copy them to layer[0]
		cout << "Making batches in video memory...\n";
		int whichBatch=0;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			data[whichBatch]=device_vector<double>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			labels[whichBatch]=device_vector<double>(batchSize*numOutputs);
			memTracker(numOutputs*batchSize*8,false);
			for(int b=0;b<batchSize;++b) {
				//temp=&pData[itemNum+b][0];
				//dataTemp=device_vector<float>(temp, temp+len);
				//tempd=&pLabels[itemNum+b][0];
				//labelsTemp=device_vector<double>(tempd, tempd+llen);
				//thrust::transform(dataTemp[itemNum+b].begin(),dataTemp[itemNum+b].end(),dataTransposeTemp.begin()+batchStart,floatToDoubleFunctor());
				//thrust::transform(dataTemp.begin(),dataTemp.end(),data[whichBatch].begin()+batchStart,floatToDoubleFunctor());
				//thrust::transform((device_vector<float>(temp, temp+len)).begin(),(device_vector<float>(temp, temp+len)).end(),data[whichBatch].begin()+batchStart,floatToDoubleFunctor());
				thrust::copy(pData[itemNum+b].begin(),pData[itemNum+b].end(),data[whichBatch].begin()+batchStart);//,floatToDoubleFunctor());
				//thrust::copy(dataTemp[itemNum+b].begin(),dataTemp[itemNum+b].end(),dataTransposeTemp.begin()+batchStart);
				//thrust::copy(labelsTemp[itemNum+b].begin(),labelsTemp[itemNum+b].end(),batchLabels.begin()+batchEnd);
				//thrust::copy(labelsTemp.begin(),labelsTemp.end(),labels[whichBatch].begin()+batchEnd);
				thrust::copy(pLabels[itemNum+b].begin(),pLabels[itemNum+b].end(),labels[whichBatch].begin()+batchEnd);
				batchStart+=itemSize;
				batchEnd+=numOutputs;
			}
			//cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, numOutputs, alpha, batchLabels.data().get(), numOutputs, beta, batchLabels.data().get(), numOutputs, labels[whichBatch].data().get(), batchSize);
			//cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, itemSize, alpha, dataTransposeTemp.data().get(), itemSize, beta, dataTransposeTemp.data().get(), itemSize, data[whichBatch].data().get(), batchSize);
			++whichBatch;
		}
		whichBatch=0;
		for(int i=0;i<testSetSize;i+=batchSize) {
			testData[whichBatch]=device_vector<double>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			batchStart=0;
			for(int j=0;j<batchSize;++j) {
				//temp=&pTestData[i+j][0];
				//dataTemp=device_vector<float>(temp, temp+len);
				//tempd=&pTestLabels[i][0];
				//labelsTemp=device_vector<double>(tempd, tempd+llen);
				//thrust::transform(dataTemp.begin(),dataTemp.end(),testData[whichBatch].begin()+batchStart,floatToDoubleFunctor());
				//thrust::transform((device_vector<float>(temp, temp+len)).begin(),(device_vector<float>(temp, temp+len)).end(),testData[i].begin(),floatToDoubleFunctor());
				thrust::copy(pTestData[i+j].begin(),pTestData[i+j].end(),testData[whichBatch].begin()+batchStart);
				//thrust::copy(labelsTemp.begin(),labelsTemp.end(),testLabels[i].begin());
				batchStart+=itemSize;
			}
			++whichBatch;
		}

		int mOut=outputsIndex-2;
		/*zipIterator begin2[outputsIndex];
		zipIterator end2[outputsIndex];
		zipIterator begin1[outputsIndex];
		zipIterator end1[outputsIndex];
		for(int i=outputsIndex-1;i;--i) {
			begin2[i]=zipIterator(thrust::make_tuple(NNlayers[i].atNeuronOutputs.begin(), innerDeltaB[mOut].begin()));
			end2[i]=zipIterator(thrust::make_tuple(NNlayers[i].atNeuronOutputs.end(), innerDeltaB[mOut].end()));
			begin1[i]=zipIterator(thrust::make_tuple(NNlayers[i].atNeuronInputs.begin(), innerDeltaB[mOut].begin()));
			end1[i]=zipIterator(thrust::make_tuple(NNlayers[i].atNeuronInputs.end(), innerDeltaB[mOut--].end()));
		}
		backProp_helper2 backProp2;
		backProp_helper backProp;

		//zipIterator fBegin[layers];
		//zipIterator fEnd[layers];
		forwardFeed_helper forwardFeed[layers];
		for(int i=1;i<layers;++i) {
			forwardFeed[i]=forwardFeed_helper(NNlayers[i].atNeuronInputs.data().get(),NNlayers[i].biases.data().get());
			//fBegin[i]=zipIterator(thrust::make_tuple(NNlayers[i].atNeuronInputs.begin(),NNlayers[i].biases.begin()));
			//fEnd[i]=zipIterator(thrust::make_tuple(NNlayers[i].atNeuronInputs.end(),NNlayers[i].biases.end()));
		}*/
		//forwardFeed_helper forwardFeed;

		cout << "Starting training...\n";
		memTracker(0,true);
		//cudaFree(&dataTemp);
		//cudaFree(&labelsTemp);
		//cudaFree(&dataTransposeTemp);
		//cudaFree(&batchLabels);

		thrust::device_vector<double>::iterator iter;
		int position;
		int gotRight=0, prevSize;
		//toDivideRMS=learningRate;
		//toDivideRMS=learningRate/((double)numBatches*(double)batchSize);
		toDivideRMS=learningRate/(double)batchSize;
		//toDivideRMS=learningRate/(double)numBatches;
		int maxGotRight=0, maxTestRight=-1, ii;
		device_vector<double> *which;
		double origLearningRate=learningRate, seconds, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		double lastNoShowTime=0.0;
		int timeEstCountDown=10, mPlus;

		for(int epochNum=0;!threadExit && epochNum<maxEpochs && maxGotRight!=dataSetSize && maxTestRight!=testSetSize;++epochNum) {
			whichBatch=0;
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thrust::fill(NNlayers[ii].outerDeltaB.begin(),NNlayers[ii].outerDeltaB.end(),0.0);
				thrust::fill(NNlayers[i].outerDeltaW.begin(),NNlayers[i].outerDeltaW.end(),0.0);
			}//*/
			if(!showIntervalCountDown) {
				gotRight=0;
			}
			startTime=high_resolution_clock::now();
			for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {

				//forward propagation
				which=&data[whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					//cublasXtDgemm(handlex, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
					//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
					//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[ii].allN),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayers[ii].atNeuronInputs.data().get(),NNlayers[ii].biases.data().get()));
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
					thrust::transform(NNlayers[ii].counterN.begin(),NNlayers[ii].counterN.end(),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayers[ii].atNeuronInputs.data().get(),NNlayers[ii].biases.data().get()));
					which=&NNlayers[ii].atNeuronOutputs;
				}

				//first check how many we got right
				if(!showIntervalCountDown) {
					batchStart=0;
					batchEnd=numOutputs;
					//printf("\nbatch starting at: %d\n",itemNum);
					for(int b=0;b<batchSize;++b) {
						iter = thrust::max_element(NNlayers[outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayers[outputsIndex].atNeuronOutputs.begin()+batchEnd);
						position = iter - NNlayers[outputsIndex].atNeuronOutputs.begin();
						position -= batchStart;
						//printf("output: %d expected: %d\n",position,bLabels[itemNum+b]);
						//for(int ot=batchStart;ot<batchEnd;++ot) {
						//	double oo=NNlayers[outputsIndex].atNeuronOutputs[ot];
						//	printf("%.5f ",oo);
						//}
						//printf("\n");
						if(position==bLabels[itemNum+b]) {
							++gotRight;
						}
						batchStart=batchEnd;
						batchEnd+=numOutputs;					
					}
				}

				//Backward propagation
				mOut=outputsIndex-1;
				mPlus=outputsIndex;
				prevSize=hiddenMatrix[mOut];
				//which=&innerDeltaB[mOut];
				thrust::transform(NNlayers[outputsIndex].counterN.begin(),NNlayers[outputsIndex].counterN.end(),NNlayers[outputsIndex].innerDeltaB.begin(),output_helper(NNlayers[outputsIndex].atNeuronOutputs.data().get(),NNlayers[outputsIndex].atNeuronInputs.data().get(),NNlayers[outputsIndex].innerDeltaB.data().get(),labels[whichBatch].data().get()));
				//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[outputsIndex].allN),(*which).begin(),output_helper(NNlayers[outputsIndex].atNeuronOutputs.data().get(),NNlayers[outputsIndex].atNeuronInputs.data().get(),(*which).data().get(),labels[whichBatch].data().get()));
				//thrust::transform(counterBegin,nodesCounterEnd[outputsIndex],innerDeltaB[mOut].begin(),output_helper(NNlayers[outputsIndex].atNeuronOutputs.data().get(),NNlayers[outputsIndex].atNeuronInputs.data().get(),innerDeltaB[mOut].data().get(),labels[whichBatch].data().get()));
				//cublasXtDgemm(handlex, CUBLAS_OP_N, CUBLAS_OP_T, numOutputs, prevSize, batchSize, alpha, innerDeltaB[mOut].data().get(), numOutputs, NNlayers[mOut].atNeuronOutputs.data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), numOutputs);
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numOutputs, prevSize, batchSize, alpha, NNlayers[outputsIndex].innerDeltaB.data().get(), numOutputs, NNlayers[mOut].atNeuronOutputs.data().get(), prevSize, beta, NNlayers[mOut].innerDeltaW.data().get(), numOutputs);
				//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numOutputs, prevSize, batchSize, alpha, (*which).data().get(), numOutputs, NNlayers[mOut].atNeuronOutputs.data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), numOutputs);

				--mOut;
				for(int i=outputsIndex-1;i;--i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1];
					prevSize=hiddenMatrix[i-1];
					//which=&innerDeltaB[mOut];
					//cublasXtDgemm(handlex, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, batchSize, nextSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, innerDeltaB[mOut+1].data().get(), nextSize, beta, innerDeltaB[mOut].data().get(), thisSize);
					cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, batchSize, nextSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, NNlayers[i+1].innerDeltaB.data().get(), nextSize, beta, NNlayers[i].innerDeltaB.data().get(), thisSize);
					//cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, batchSize, nextSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, innerDeltaB[mOut+1].data().get(), nextSize, beta, (*which).data().get(), thisSize);
					//thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allW),update_w(NNlayers[i].weightsMatrix.data().get(),innerDeltaW[i].data().get(),toDivideRMS));
					////thrust::for_each(counterBegin,weightsCounterEnd[i],update_w(NNlayers[i].weightsMatrix.data().get(),innerDeltaW[i].data().get(),toDivideRMS));
					//thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[mPlus].allN),update_b(NNlayers[mPlus].biases.data().get(),innerDeltaB[i].data().get(),toDivideRMS));
					////thrust::for_each(counterBegin,nodesCounterEnd[mPlus],update_b(NNlayers[mPlus].biases.data().get(),innerDeltaB[i].data().get(),toDivideRMS));
					//zipIterator begin(thrust::make_tuple(NNlayers[i].atNeuronOutputs.begin(), innerDeltaB[mOut].begin()));
					//zipIterator end(thrust::make_tuple(NNlayers[i].atNeuronOutputs.end(), innerDeltaB[mOut].end()));
					//thrust::transform(begin,end,innerDeltaB[mOut].begin(),backProp_helper2());
					//thrust::transform(begin2[i],end2[i],innerDeltaB[mOut].begin(),backProp2);
					if(i!=1) {
						thrust::transform(NNlayers[i].counterN.begin(),NNlayers[i].counterN.end(),NNlayers[i].innerDeltaB.begin(),backProp_helper2(NNlayers[i].atNeuronOutputs.data().get(),NNlayers[i].innerDeltaB.data().get()));
						//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allN),(*which).begin(),backProp_helper2(NNlayers[i].atNeuronOutputs.data().get(),(*which).data().get()));
						//thrust::transform(counterBegin,nodesCounterEnd[i],innerDeltaB[mOut].begin(),backProp_helper2(NNlayers[i].atNeuronOutputs.data().get(),innerDeltaB[mOut].data().get()));
						//cublasXtDgemm(handlex, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, innerDeltaB[mOut].data().get(), thisSize, NNlayers[i-1].atNeuronOutputs.data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), thisSize);
						cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, NNlayers[i].innerDeltaB.data().get(), thisSize, NNlayers[i-1].atNeuronOutputs.data().get(), prevSize, beta, NNlayers[mOut].innerDeltaW.data().get(), thisSize);
						//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, (*which).data().get(), thisSize, NNlayers[i-1].atNeuronOutputs.data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), thisSize);
					} else {
						//zipIterator begin(thrust::make_tuple(NNlayers[i].atNeuronInput.begin(), innerDeltaB[mOut].begin()));
						//zipIterator end(thrust::make_tuple(NNlayers[i].atNeuronInputs.end(), innerDeltaB[mOut].end()));
						//thrust::transform(begin,end,innerDeltaB[mOut].begin(),backProp_helper());
						//thrust::transform(begin1[i],end1[i],innerDeltaB[mOut].begin(),backProp);
						thrust::transform(NNlayers[i].counterN.begin(),NNlayers[i].counterN.end(),NNlayers[i].innerDeltaB.begin(),backProp_helper(NNlayers[i].innerDeltaB.data().get(),NNlayers[i].atNeuronInputs.data().get()));
						//thrust::transform(counterBegin,nodesCounterEnd[i],innerDeltaB[mOut].begin(),backProp_helper(innerDeltaB[mOut].data().get(),NNlayers[i].atNeuronInputs.data().get()));
						//cublasXtDgemm(handlex, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, innerDeltaB[mOut].data().get(), thisSize, data[whichBatch].data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), thisSize);
						cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, NNlayers[i].innerDeltaB.data().get(), thisSize, data[whichBatch].data().get(), prevSize, beta, NNlayers[mOut].innerDeltaW.data().get(), thisSize);

					}
					--mOut;
					--mPlus;
				}

				//thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[0].allW),update_w(NNlayers[0].weightsMatrix.data().get(),innerDeltaW[0].data().get(),toDivideRMS));
				////thrust::for_each(counterBegin,weightsCounterEnd[0],update_w(NNlayers[0].weightsMatrix.data().get(),innerDeltaW[0].data().get(),toDivideRMS));
				//thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[1].allN),update_b(NNlayers[1].biases.data().get(),innerDeltaB[0].data().get(),toDivideRMS));
				////thrust::for_each(counterBegin,nodesCounterEnd[1],update_b(NNlayers[1].biases.data().get(),innerDeltaB[0].data().get(),toDivideRMS));
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thrust::transform(NNlayers[ii].innerDeltaB.begin(),NNlayers[ii].innerDeltaB.end(),NNlayers[ii].outerDeltaB.begin(),NNlayers[ii].outerDeltaB.begin(),thrust::plus<double>());
					thrust::transform(NNlayers[i].innerDeltaW.begin(),NNlayers[i].innerDeltaW.end(),NNlayers[i].outerDeltaW.begin(),NNlayers[i].outerDeltaW.begin(),thrust::plus<double>());
				}//*/
				/*for(int i=0;i<outputsIndex;++i) {
					thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allW),update_w(NNlayers[i].weightsMatrix.data().get(),innerDeltaW[i].data().get(),toDivideRMS));
					thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i+1].allN),update_b(NNlayers[i+1].biases.data().get(),innerDeltaB[i].data().get(),toDivideRMS));
				}//*/
				++whichBatch;
			}
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thrust::for_each(NNlayers[i].counterW.begin(),NNlayers[i].counterW.end(),update_w(NNlayers[i].weightsMatrix.data().get(),NNlayers[i].outerDeltaW.data().get(),toDivideRMS));
				thrust::for_each(NNlayers[ii].counterN.begin(),NNlayers[ii].counterN.end(),update_b(NNlayers[ii].biases.data().get(),NNlayers[ii].outerDeltaB.data().get(),toDivideRMS));
			}//*/

			if(!showIntervalCountDown) {
				if(gotRight>maxGotRight){maxGotRight=gotRight;}
				printf("Epoch: %d-Got %d of %d-max right: %d-lRate: %.5f",epochNum,gotRight,dataSetSize,maxGotRight,learningRate);
				printf("-");
				gotRight=0;
			
				whichBatch=0;
				for(int t=0;t<testSetSize;t+=batchSize) {
					which=&testData[whichBatch];
					for(int i=0;i<outputsIndex;++i) {
						ii=i+1;
						thisSize=hiddenMatrix[i];
						nextSize=hiddenMatrix[ii];
						//cublasXtDgemm(handlex, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
						cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
						thrust::transform(NNlayers[ii].counterN.begin(),NNlayers[ii].counterN.end(),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayers[ii].atNeuronInputs.data().get(),NNlayers[ii].biases.data().get()));
						//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[ii].allN),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed[ii]);
						which=&NNlayers[ii].atNeuronOutputs;
					}
					if(!showIntervalCountDown) {
						batchStart=0;
						batchEnd=numOutputs;
						//printf("\nbatch starting at: %d\n",t);
						for(int b=0;b<batchSize;++b) {
							iter = thrust::max_element(NNlayers[outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayers[outputsIndex].atNeuronOutputs.begin()+batchEnd);
							position = iter - NNlayers[outputsIndex].atNeuronOutputs.begin();
							position -= batchStart;
							/*printf("output: %d expected: %d\n",position,btLabels[t+b]);
							for(int ot=batchStart;ot<batchEnd;++ot) {
								double oo=NNlayers[outputsIndex].atNeuronOutputs[ot];
								printf("%.5f ",oo);
							}
							printf("\n");//*/
							if(position==btLabels[t+b]) {
								++gotRight;
							}
							batchStart=batchEnd;
							batchEnd+=numOutputs;
						}
					}
					++whichBatch;
				}
				if(gotRight>maxTestRight && testSetSize){maxTestRight=gotRight;}
			}
			if(vlRate) {
				//if(epochNum>1) {
					double cutOff=0.92;
					double percLearned=(double)gotRight/(double)testSetSize;
					if(percLearned<0.99 && percLearned>cutOff) {
						percLearned=1.0-percLearned;
						//percLearned=(1.0-percLearned)*2.0;
						//percLearned=(1.0-percLearned)/2.0;
						//percLearned=pow(1.0-percLearned,(double)layers);
						//percLearned=pow(1.0-percLearned,2.0);
						//alfLearn=-(percLearned*(learningRate/2.0)+(learningRate/2.0));
						learningRate=(cutOff*origLearningRate)+percLearned;//-(percLearned*origLearningRate);
						toDivideRMS=learningRate/(double)batchSize;
						//toDivideRMS=learningRate/(double)numBatches;
					} else {
						if(percLearned<0.99) {
							learningRate=origLearningRate;
							toDivideRMS=learningRate/(double)batchSize;
							//toDivideRMS=learningRate/(double)numBatches;
						}
					}
				//}
			}
			endTime=high_resolution_clock::now();
			seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
			totalTime+=seconds;
			if(!showIntervalCountDown) {
				double errRate=(1.0-((double)gotRight/(double)testSetSize))*100.0;
				printf("Test-Got %d of %d-max right: %d-sec: %.5f-totTime: %.5f-errRate: %.5f\n",gotRight,testSetSize,maxTestRight,lastNoShowTime,totalTime,errRate);
				showIntervalCountDown=showInterval;
				/*if(maxTestRight!=gotRight) {
					pthread_mutex_lock(&crmutex);
					counterGo=true;
					pthread_mutex_unlock(&crmutex);
				}*/
			} else {
				lastNoShowTime=seconds;
				--showIntervalCountDown;
				if(timeEstCountDown) {
					--timeEstCountDown;
					if(!timeEstCountDown) {
						//printf("(yes it's running...)\n");
						printf("Update time interval approximately %.5f seconds apart\n",(lastNoShowTime*(double)showInterval)+5.0);
						/*waitTime=(int)(lastNoShowTime*(double)showInterval);
						pthread_mutex_lock(&crmutex);
						counterGo=true;
						pthread_mutex_unlock(&crmutex);*/
					}
				}
			}
		}
		/*if(showInterval) {
			pthread_mutex_lock(&crmutex);
			counterExit=true;
			pthread_mutex_unlock(&crmutex);
			//void *result;
			//pthread_join(counter, &result);
		}*/
		cublasDestroy(handle);
		//cublasXtDestroy(handlex);
		//saveState("MPIv-Cuda-");
		//sleep(5);
	}

	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize, bool floata, bool floatb) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize) {

		cout << "in float\n";
		cudaSetDevice(3);
		if(batchSize<100) {
			batchSize=10000;
		}
		cublasCreate(&handle);
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);

		NNlayersf=vector<NN_layerf>(hiddenMatrix.size());
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;
		cout << "Setting up network...\n";
		cout << "Layers: ";
		for(auto h:hiddenMatrix) {
			cout << h << " ";
		}
		cout << "Batch size: " << batchSize << endl << endl;

		NNlayersf[0]=NN_layerf(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT);
		for(int i=1;i<outputsIndex;++i) {
			NNlayersf[i]=NN_layerf(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN);
		}
		NNlayersf[outputsIndex]=NN_layerf(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT);
	}

	void train_MatMulf(vector<vector<float>> &pData, vector<vector<float>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, float lRate, vector<vector<float>> &pTestData, vector<vector<float>> &pTestLabels, bool vlRate) {


		cout << "in other float\n";
		if(!showInterval) {
			showInterval=10;
		}
		vector<UNCHAR> bLabels;
		for(auto p:pLabels) {
			bLabels.push_back((UNCHAR)(thrust::max_element(p.begin(), p.end())-p.begin()));
		}

		if(lRate<0.0f) {
			learningRatef=0.05f;
		} else {
			learningRatef=lRate;
		}
		if(!doDataSetSize) {
			doDataSetSize=60000;
		}
		dataSetSize=doDataSetSize;
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		int batchStart,batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		int testSetSize=pTestData.size();
		vector<UNCHAR> btLabels;
		device_vector<float> testData[testSetSize/batchSize];
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back((UNCHAR)(thrust::max_element(p.begin(), p.end())-p.begin()));
			}
		} else {
			cudaFree(&testData);
		}
		int numBatches=dataSetSize/batchSize;
		device_vector<float> data[numBatches];
		device_vector<float> labels[numBatches];

		//Creating pre-made batches so I can simply copy them to layer[0]
		cout << "Making batches in video memory...\n";
		int whichBatch=0;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			data[whichBatch]=device_vector<float>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			labels[whichBatch]=device_vector<float>(batchSize*numOutputs);
			memTracker(numOutputs*batchSize*8,false);
			for(int b=0;b<batchSize;++b) {
				thrust::copy(pData[itemNum+b].begin(),pData[itemNum+b].end(),data[whichBatch].begin()+batchStart);//,floatToDoubleFunctor());
				thrust::copy(pLabels[itemNum+b].begin(),pLabels[itemNum+b].end(),labels[whichBatch].begin()+batchEnd);
				batchStart+=itemSize;
				batchEnd+=numOutputs;
			}
			++whichBatch;
		}
		whichBatch=0;
		for(int i=0;i<testSetSize;i+=batchSize) {
			testData[whichBatch]=device_vector<float>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			batchStart=0;
			for(int j=0;j<batchSize;++j) {
				thrust::copy(pTestData[i+j].begin(),pTestData[i+j].end(),testData[whichBatch].begin()+batchStart);
				batchStart+=itemSize;
			}
			++whichBatch;
		}

		int mOut=outputsIndex-2;
		cout << "Starting training...\n";
		memTracker(0,true);

		thrust::device_vector<float>::iterator iter;
		int position;
		int gotRight=0, prevSize;
		toDivideRMSf=learningRatef/(float)batchSize;
		int maxGotRight=0, maxTestRight=-1, ii;
		device_vector<float> *which;
		float origlearningRatef=learningRatef, seconds, totalTime=0.0f;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		float lastNoShowTime=0.0f;
		int timeEstCountDown=10, mPlus;

		for(int epochNum=0;!threadExit && epochNum<maxEpochs && maxGotRight!=dataSetSize && maxTestRight!=testSetSize;++epochNum) {
			whichBatch=0;
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thrust::fill(NNlayersf[ii].outerDeltaB.begin(),NNlayersf[ii].outerDeltaB.end(),0.0f);
				thrust::fill(NNlayersf[i].outerDeltaW.begin(),NNlayersf[i].outerDeltaW.end(),0.0f);
			}//*/
			if(!showIntervalCountDown) {
				gotRight=0;
			}
			startTime=high_resolution_clock::now();
			for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {

				//forward propagation
				which=&data[whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayersf[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayersf[ii].atNeuronInputs.data().get(), nextSize);
					thrust::transform(NNlayersf[ii].counterN.begin(),NNlayersf[ii].counterN.end(),NNlayersf[ii].atNeuronOutputs.begin(),forwardFeed_helperf(NNlayersf[ii].atNeuronInputs.data().get(),NNlayersf[ii].biases.data().get()));
					which=&NNlayersf[ii].atNeuronOutputs;
				}

				//first check how many we got right
				if(!showIntervalCountDown) {
					batchStart=0;
					batchEnd=numOutputs;
					//printf("\nbatch starting at: %d\n",itemNum);
					for(int b=0;b<batchSize;++b) {
						iter = thrust::max_element(NNlayersf[outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayersf[outputsIndex].atNeuronOutputs.begin()+batchEnd);
						position = iter - NNlayersf[outputsIndex].atNeuronOutputs.begin();
						position -= batchStart;
						//printf("output: %d expected: %d\n",position,bLabels[itemNum+b]);
						//for(int ot=batchStart;ot<batchEnd;++ot) {
						//	float oo=NNlayersf[outputsIndex].atNeuronOutputs[ot];
						//	printf("%.5f ",oo);
						//}
						//printf("\n");
						if(position==bLabels[itemNum+b]) {
							++gotRight;
						}
						batchStart=batchEnd;
						batchEnd+=numOutputs;					
					}
				}

				//Backward propagation
				mOut=outputsIndex-1;
				mPlus=outputsIndex;
				prevSize=hiddenMatrix[mOut];
				thrust::transform(NNlayersf[outputsIndex].counterN.begin(),NNlayersf[outputsIndex].counterN.end(),NNlayersf[outputsIndex].innerDeltaB.begin(),output_helperf(NNlayersf[outputsIndex].atNeuronOutputs.data().get(),NNlayersf[outputsIndex].atNeuronInputs.data().get(),NNlayersf[outputsIndex].innerDeltaB.data().get(),labels[whichBatch].data().get()));
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numOutputs, prevSize, batchSize, alpha, NNlayersf[outputsIndex].innerDeltaB.data().get(), numOutputs, NNlayersf[mOut].atNeuronOutputs.data().get(), prevSize, beta, NNlayersf[mOut].innerDeltaW.data().get(), numOutputs);

				--mOut;
				for(int i=outputsIndex-1;i;--i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1];
					prevSize=hiddenMatrix[i-1];
					cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, batchSize, nextSize, alpha, NNlayersf[i].weightsMatrix.data().get(), nextSize, NNlayersf[i+1].innerDeltaB.data().get(), nextSize, beta, NNlayersf[i].innerDeltaB.data().get(), thisSize);
					if(i!=1) {
						thrust::transform(NNlayersf[i].counterN.begin(),NNlayersf[i].counterN.end(),NNlayersf[i].innerDeltaB.begin(),backProp_helper2f(NNlayersf[i].atNeuronOutputs.data().get(),NNlayersf[i].innerDeltaB.data().get()));
						cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, NNlayersf[i].innerDeltaB.data().get(), thisSize, NNlayersf[i-1].atNeuronOutputs.data().get(), prevSize, beta, NNlayersf[mOut].innerDeltaW.data().get(), thisSize);
					} else {
						thrust::transform(NNlayersf[i].counterN.begin(),NNlayersf[i].counterN.end(),NNlayersf[i].innerDeltaB.begin(),backProp_helperf(NNlayersf[i].innerDeltaB.data().get(),NNlayersf[i].atNeuronInputs.data().get()));
						cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, NNlayersf[i].innerDeltaB.data().get(), thisSize, data[whichBatch].data().get(), prevSize, beta, NNlayersf[mOut].innerDeltaW.data().get(), thisSize);

					}
					--mOut;
					--mPlus;
				}
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thrust::transform(NNlayersf[ii].innerDeltaB.begin(),NNlayersf[ii].innerDeltaB.end(),NNlayersf[ii].outerDeltaB.begin(),NNlayersf[ii].outerDeltaB.begin(),thrust::plus<float>());
					thrust::transform(NNlayersf[i].innerDeltaW.begin(),NNlayersf[i].innerDeltaW.end(),NNlayersf[i].outerDeltaW.begin(),NNlayersf[i].outerDeltaW.begin(),thrust::plus<float>());
				}//*/
				++whichBatch;
			}
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thrust::for_each(NNlayersf[i].counterW.begin(),NNlayersf[i].counterW.end(),update_wf(NNlayersf[i].weightsMatrix.data().get(),NNlayersf[i].outerDeltaW.data().get(),toDivideRMSf));
				thrust::for_each(NNlayersf[ii].counterN.begin(),NNlayersf[ii].counterN.end(),update_bf(NNlayersf[ii].biases.data().get(),NNlayersf[ii].outerDeltaB.data().get(),toDivideRMSf));
			}//*/

			if(!showIntervalCountDown) {
				if(gotRight>maxGotRight){maxGotRight=gotRight;}
				printf("Epoch: %d-Got %d of %d-max right: %d-lRate: %.5f",epochNum,gotRight,dataSetSize,maxGotRight,learningRatef);
				printf("-");
				gotRight=0;
			
				whichBatch=0;
				for(int t=0;t<testSetSize;t+=batchSize) {
					which=&testData[whichBatch];
					for(int i=0;i<outputsIndex;++i) {
						ii=i+1;
						thisSize=hiddenMatrix[i];
						nextSize=hiddenMatrix[ii];
						cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayersf[i].weightsMatrix.data().get(), nextSize, (*which).data().get(), thisSize, beta, NNlayersf[ii].atNeuronInputs.data().get(), nextSize);
						thrust::transform(NNlayersf[ii].counterN.begin(),NNlayersf[ii].counterN.end(),NNlayersf[ii].atNeuronOutputs.begin(),forwardFeed_helperf(NNlayersf[ii].atNeuronInputs.data().get(),NNlayersf[ii].biases.data().get()));
						which=&NNlayersf[ii].atNeuronOutputs;
					}
					if(!showIntervalCountDown) {
						batchStart=0;
						batchEnd=numOutputs;
						//printf("\nbatch starting at: %d\n",t);
						for(int b=0;b<batchSize;++b) {
							iter = thrust::max_element(NNlayersf[outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayersf[outputsIndex].atNeuronOutputs.begin()+batchEnd);
							position = iter - NNlayersf[outputsIndex].atNeuronOutputs.begin();
							position -= batchStart;
							/*printf("output: %d expected: %d\n",position,btLabels[t+b]);
							for(int ot=batchStart;ot<batchEnd;++ot) {
								float oo=NNlayersf[outputsIndex].atNeuronOutputs[ot];
								printf("%.5f ",oo);
							}
							printf("\n");//*/
							if(position==btLabels[t+b]) {
								++gotRight;
							}
							batchStart=batchEnd;
							batchEnd+=numOutputs;
						}
					}
					++whichBatch;
				}
				if(gotRight>maxTestRight && testSetSize){maxTestRight=gotRight;}
			}
			endTime=high_resolution_clock::now();
			seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0f;
			totalTime+=seconds;
			if(!showIntervalCountDown) {
				float errRate=(1.0f-((float)gotRight/(float)testSetSize))*100.0f;
				printf("Test-Got %d of %d-max right: %d-sec: %.5f-totTime: %.5f-errRate: %.5f\n",gotRight,testSetSize,maxTestRight,lastNoShowTime,totalTime,errRate);
				showIntervalCountDown=showInterval;
			} else {
				lastNoShowTime=seconds;
				--showIntervalCountDown;
				if(timeEstCountDown) {
					--timeEstCountDown;
					if(!timeEstCountDown) {
						printf("Update time interval approximately %.5f seconds apart\n",(lastNoShowTime*(float)showInterval)+5.0f);
					}
				}
			}
		}
		cublasDestroy(handle);
	}

	void saveState(string outFile) {
		outFile+="-"+to_string(dataSetSize);
		cout << "Writing weights to file: " << outFile << endl;
		ofstream oFile(outFile, ios::binary|ios::out);
		if(oFile.is_open()) {
			oFile.write((char*)&epoch,sizeof(ULLI));
			oFile.write((char*)&layers,sizeof(ULLI));
			for(int i=0;i<hiddenMatrix.size();++i) {
				oFile.write((char*)&hiddenMatrix[i],sizeof(int));
			}
			oFile.write((char*)&batchSize,sizeof(int));
			oFile.write((char*)&learningRate,sizeof(double));
			for(int i=0;i<outputsIndex;++i) {
				for(int j=0;j<NNlayers[i].allW;++j) {
					double o=NNlayers[i].weightsMatrix[j];
					oFile.write((char*)&o,sizeof(double));
				}
			}
			for(int i=1;i<layers;++i) {
				for(int j=0;j<NNlayers[i].allN;++j) {
					double o=NNlayers[i].biases[j];
					oFile.write((char*)&o,sizeof(double));
				}
			}
			oFile.close();
		}
		cout << "Done\n";
	}

	void loadState() {
		cout << "Reading weights from file: " << inFile << endl;
		ifstream oFile(inFile, ios::binary|ios::in);
		if(oFile.is_open()) {
			oFile.read((char*)&epoch,sizeof(ULLI));
			oFile.read((char*)&layers,sizeof(ULLI));
			hiddenMatrix.clear();
			for(int i=0;i<layers;++i) {
				int l=0;
				oFile.read((char*)&l,sizeof(int));
				hiddenMatrix.push_back(l);
			}
			oFile.read((char*)&batchSize,sizeof(int));
			oFile.read((char*)&learningRate,sizeof(double));

			outputsIndex=layers-1;
			numInputs=hiddenMatrix[0]-1;
			numOutputs=hiddenMatrix[outputsIndex];

			NNlayers.clear();
			int type=INPUT;
			for(int i=0;i<outputsIndex;++i) {
				if(i){type=HIDDEN;}
				NNlayers.push_back(NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,type));
				for(int j=0;j<NNlayers[i].allW;++j) {
					double o=0.0;
					oFile.read((char*)&o,sizeof(double));
					NNlayers[i].weightsMatrix.push_back(o);
				}		
				NNlayers[i].setupLayer(false);
			}
			NNlayers.push_back(NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT));
			NNlayers.back().setupLayer(false);
			for(int i=1;i<layers;++i) {
				for(int j=0;j<NNlayers[i].allN;++j) {
					double o=0.0;
					oFile.read((char*)&o,sizeof(double));
					NNlayers[i].biases.push_back(o);
				}		
			}			
			oFile.close();
		}
		cout << "Done\n";
	}
	vector<NN_layer> NNlayers;
	vector<NN_layerf> NNlayersf;
	vector<NN_layer> NNlayersQ[2];

private:
	ULLI epoch, maxElement, layers, maxEpochs;//, maxWeightsMatrix, maxDeltaMatrix;
	int outputsIndex, dataSetSize, numInputs, numOutputs, batchSize;
	double RMS, minRMS, toDivideRMS, RMSwanted, learningRate;
	float toDivideRMSf, learningRatef;
	vector<int> hiddenMatrix;
	cublasHandle_t handle;
	ULLI itemSize;
	string inFile;
	ULLI neededEpochs;

	vector<vector<double>> neuralNet_weights_host;
};

void doMain(vector<int> &inputHiddenLayers, int batchSize, int doDataSetSize, double lRate, string inFile, string outFile, bool vlRate, int numDevs) {

	if(doMNISTprob) {
		vector<int> hiddenMatrix;
		if(!inputHiddenLayers.size()) {
			hiddenMatrix.push_back(200);
			hiddenMatrix.push_back(100);
			//hiddenMatrix.push_back(10);
			//hiddenMatrix.push_back(784+(784/2));
			//hiddenMatrix.push_back(784+(784/2));
			//hiddenMatrix.push_back(784);
			//hiddenMatrix.push_back(784);
		} else {
			for(auto h:inputHiddenLayers) {
				hiddenMatrix.push_back(h);
			}
		}

		vector<vector<double>> testData(10000);
		ReadMNIST_double("t10k-images.idx3-ubyte",10000,784,testData);
		vector<vector<double>> trainData(60000);
		ReadMNIST_double("train-images.idx3-ubyte",60000,784,trainData);//*/
		vector<vector<float>> testDataf(10000);
		ReadMNIST_float("t10k-images.idx3-ubyte",10000,784,testDataf);
		vector<vector<float>> trainDataf(60000);
		ReadMNIST_float("train-images.idx3-ubyte",60000,784,trainDataf);//*/

		vector<vector<double>> testLabels(10000);
		vector<vector<double>> trainLabels(60000);
		vector<vector<float>> testLabelsf(10000);
		vector<vector<float>> trainLabelsf(60000);	
		//vector<UNCHAR> testLabels2;//(10000);
		//vector<UNCHAR> trainLabels2;//(60000);
		ifstream file("t10k-labels.idx1-ubyte",ios::binary);
		if(file.is_open()) {
			int placeHolder=0;
			file.read((char*)&placeHolder,sizeof(placeHolder));
			file.read((char*)&placeHolder,sizeof(placeHolder));
			for(int i=0;i<10000;++i) {
				testLabels[i]=vector<double>(10,0.0);
				testLabelsf[i]=vector<float>(10,0.0f);
				//testLabels[i]=vector<float>(10,0.0f);
				UNCHAR temp=0;
				file.read((char*)&temp,1);
				for(UNCHAR j=0;j<10;++j) {
					if(j==temp) {
						//testLabels[i].push_back(1.0);
						//testLabels[i][j]=1.0f;
						testLabels[i][j]=1.0;
						testLabelsf[i][j]=1.0f;
						//testLabels2.push_back(temp);
					} /*else {
						//testLabels[i].push_back(0.0);
						testLabels[i][j]=0.0;
					}*/
				}
			}
			file.close();
		}
		//cout << "testLabels2 size: " << testLabels2.size() << endl;
		ifstream file2("train-labels.idx1-ubyte",ios::binary);
		if(file2.is_open()) {
			int placeHolder=0;
			file2.read((char*)&placeHolder,sizeof(placeHolder));
			file2.read((char*)&placeHolder,sizeof(placeHolder));
			for(int i=0;i<60000;++i) {
				trainLabels[i]=vector<double>(10,0.0);
				trainLabelsf[i]=vector<float>(10,0.0f);
				//trainLabels[i]=vector<float>(10,0.0f);
				UNCHAR temp=0;
				file2.read((char*)&temp,1);
				for(UNCHAR j=0;j<10;++j) {
					if(j==temp) {
						//trainLabels[i].push_back(1.0);
						//trainLabels[i][j]=1.0f;
						trainLabels[i][j]=1.0;
						trainLabelsf[i][j]=1.0f;
						//trainLabels2.push_back(temp);
					} /*else {
						//trainLabels[i].push_back(0.0);
						trainLabels[i][j]=0.0;
					}*/
				}
			}
			file2.close();
		}
		//cout << "trainLabels2 size: " << trainLabels2.size() << endl;

		//vector<UNCHAR> temp;
		//for(auto p:trainData[1]) {
		//	temp.push_back((UNCHAR)(p*255.0f));
		//	cout << (int)temp.back() << endl;
		//}
		//UNCHAR* t=&temp[0];
		//intarray2bmp::intarray2bmp("outputtest.bmp",t,(UNCHAR)28,(UNCHAR)28,(UNCHAR)0,(UNCHAR)255);
		neuralNet go;
		if(inFile=="") {
			//go=neuralNet(784,10,hiddenMatrix,batchSize,numDevs);
			//go=neuralNet(784,10,hiddenMatrix,batchSize);
			go=neuralNet(784,10,hiddenMatrix,batchSize,true,true);
		} else {
			go=neuralNet(inFile);
		}
		auto start = high_resolution_clock::now();
		//go.train_floats(trainData,trainLabels,1000000,0.0001,trainLabels2);
		//go.train(trainData,trainLabels,1000000,0.0001,trainLabels2, doDataSetSize);//*/
		//go.train_Quad(trainData,trainLabels, 1000000, 0.0001, doDataSetSize, lRate, testData, testLabels, vlRate);//*/
		go.train_MatMulf(trainDataf,trainLabelsf, 1000000, 0.0001, doDataSetSize, lRate, testDataf, testLabelsf, vlRate);//*/
		//go.train_MatMul(trainData,trainLabels, 1000000, 0.0001, doDataSetSize, lRate, testData, testLabels, vlRate);//*/
		//go.evaluate(testData,testLabels,testLabels2, doDataSetSize);
		auto endTime = high_resolution_clock::now();
		printTime(start,endTime);
	}
	if(doBinaryProb) {
		vector<int> hiddenMatrix;
		hiddenMatrix.push_back(BITS+(BITS/2));
		//hiddenMatrix.push_back(BITS+(BITS/2));
		for(int i=0;i<1;++i) {
			hiddenMatrix.push_back(BITS+(BITS/2));
			//hiddenMatrix.push_back(12);
		}
		//vector<vector<neuron_t>> countingTest;
		//vector<vector<double>> countingLabels;
		int size=pow(2,BITS);
		neuralNet test(BITS,BITS,hiddenMatrix,batchSize,numDevs);
		vector<vector<double>> countingTest;
		vector<vector<double>> countingLabels;
		for(int i=0;i<size;++i) {
			countingTest.push_back(vector<double>(BITS));
			countingLabels.push_back(vector<double>(BITS,0.0));
			//countingLabels[i]=vector<double>(BITS,0.0);
			//countingTest[i]=vector<neuron_t>(BITS);
			for(int j=0;j<BITS;++j) {
				//countingTest.back()[j].output=(double)bitset<BITS>(i)[(BITS-1)-j];
				//countingLabels.back()[j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
				countingTest[i][j]=(double)bitset<BITS>(i)[(BITS-1)-j];
				countingLabels[i][j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
			}
		}
		test.train_Quad(countingTest,countingLabels,1000000,0.00001,size,lRate,countingTest,countingLabels,vlRate);
	}
}

int main(int argc, char *argv[]) {
	/*cudaSetDevice(1);
	cudaDeviceReset();
	cudaSetDevice(2);
	cudaDeviceReset();
	return 0;*/

	struct sigaction ctrlc;
	ctrlc.sa_handler=ctrlchandler;
	ctrlc.sa_flags=0;
	sigemptyset(&ctrlc.sa_mask);
	sigaction(SIGQUIT,&ctrlc,NULL);

	string inFile="";
	string outFile="";
	int doDataSetSize=0;
	int batchSize=5;
	if(doBinaryProb) {
		batchSize=4;
	}
	double lRate=-1.0;
	bool vlRate=false;
	if(!vlRate){}
	vector<int> inputHiddenLayers;
	showInterval=0;
    for(int i=1;i<argc;++i) {
        string temp=string(argv[i]);
        if(temp.find("showInterval=")!=string::npos) {
        	sscanf(argv[i],"showInterval=%d",&showInterval);
        	continue;
        }
        if(temp.find("showTrain")!=string::npos) {
        	showCorrectNumTrain=true;
        	continue;
        }
        if(temp.find("vlRate")!=string::npos) {
        	vlRate=true;
        }
        if(temp.find("outWeights=")!=string::npos) {
        	outFile=temp.substr(11,temp.size());
        	continue;
        }
        if(temp.find("inWeights=")!=string::npos) {
        	inFile=temp.substr(10,temp.size());
        	continue;
        }
        if(temp.find("setSize=")!=string::npos) {
        	sscanf(argv[i],"setSize=%d",&doDataSetSize);
        	continue;
        }
        if(temp.find("batchSize=")!=string::npos) {
        	sscanf(argv[i],"batchSize=%d",&batchSize);
        	continue;
        }
        if(temp.find("learningRate=")!=string::npos) {
        	sscanf(argv[i],"learningRate=%lf",&lRate);
        	continue;
        }
        if(temp.find("layers=")!=string::npos) {
        	temp.erase(0,7);
        	int where;
        	int what=1;
       		while(what) {
       			if(temp.find(",")!=string::npos) {
       				where=temp.find(",");
       				string temp2=string(temp.begin(),temp.begin()+where);
       				sscanf(temp2.c_str(),"%d",&what);
       				inputHiddenLayers.push_back(what);
       				temp.erase(0,where+1);
       			} else {
       				what=0;
       			}
       		}
       		sscanf(temp.c_str(),"%d",&what);
       		inputHiddenLayers.push_back(what);
        }
    }

	//Cuda doesn't like this first one
	//srandom(time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());
	srand((unsigned int)time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());

	/*int my_rank, num_nodes;
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    char my_host[100];
    gethostname(my_host, 100);
    string hostname=string(my_host);
    printf("%s\n",hostname.c_str());*/

	/*int deviceCount = 0;
	size_t mem_tot_0 = 0;
	size_t mem_free_0 = 0;*/
	int numDevs=2;

	pthread_barrier_init(&barrier, NULL, numDevs+1);
	pthread_barrier_init(&barrier2, NULL, numDevs+1);

		/*ULLI totalCudaMem=0;
		size_t totalFreeCudaMem;
		int device_num;

		//This code is from deviceQuery.cpp as seen in /usr/local/cuda-8.0/samples
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
			/*printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
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
	    	cudaMemGetInfo(&mem_free_0, &mem_tot_0);
			cout << "Total free cuda memory: " << mem_free_0 << endl;*/
		//}

		//if(totalCudaMem) {
			//cout << string(my_host) << ": total Cuda Memory: " << totalCudaMem << endl;
			//cout << "Total Cuda Memory: " << totalCudaMem << endl;
		//}

	//}
	cudaSetDevice(0);

	//cuda thread test

	/*
	double test=(double)doDataSetSize/4.0;
	int test2=(int)test;
	if((double)test2!=test) {
		cout << "setSize must be divisible by four\n";
		MPI_Finalize();
		return 0;
	}
	device_vector<double> data(25,0.0);
	device_vector<double> labels(25,0.0);

	cudaMemGetInfo(&mem_free_0, &mem_tot_0);
	//totalFreeCudaMem=mem_free_0;
	cout << "Total free cuda memory: " << mem_free_0 << endl;

	//random_doubles(thrust::raw_pointer_cast(&data[0]),5,5);
	//random_doubles(thrust::raw_pointer_cast(&labels[0]),5,5);
	for(int i=0;i<25;++i) {
		data[i]=(double)i;
		labels[i]=(double)i;
	}
	int index=0;
	int index2=0;
	int testI=2500;
	vector<pthread_t> threads;
    for(int j=0;j<deviceCount;++j) {
        threads.push_back(pthread_t());
        idLink *arg = (idLink*)malloc(sizeof(*arg));
        (*arg).whichThread=j;
        (*arg).batchStart=index;
        index+=test2;
        (*arg).batchEnd=index;
        (*arg).testStart=index2;
        index2+=testI;
        (*arg).testEnd=index2;
        (*arg).data=&data;
        (*arg).labels=&labels;
        pthread_create (&threads.at(j), NULL,  cudaThread, arg);
    }

	cudaMemGetInfo(&mem_free_0, &mem_tot_0);
	//totalFreeCudaMem=mem_free_0;
	cout << "Total free cuda memory: " << mem_free_0 << endl;

    int status;
	void * result;
    for (int i=0; i < deviceCount; ++i) {
        if ((status = pthread_join(threads.at(i), &result)) != 0) {
            fprintf (stderr, "join error %d: %s\n", status, strerror(status));
            exit (1);
        }
    }*/
	doMain(inputHiddenLayers, batchSize, doDataSetSize, lRate, inFile, outFile, vlRate, numDevs);

	//MPI_Finalize();
	//doMain(0,"",0);

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

void ReadMNIST_double(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr) {
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
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
        	arr[i]=vector<double>();
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    UNCHAR temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    //arr[i][(n_rows*r)+c]= ((float)temp)/256.0f;
                    //cout << "from read: " << ((float)temp)/256.0f << ": ";
                    arr[i].push_back(((double)temp)/256.0);
                    //arr[i].push_back((float)temp);
                    //cout << "from arr: " << arr[i].back() << endl;
                }
            }
        }
    }
    file.close();
}

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr) {
    arr.resize(NumberOfImages,vector<float>(DataOfAnImage));
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
                    //arr[i].push_back((float)temp);
                    //cout << "from arr: " << arr[i].back() << endl;
                }
            }
        }
    }
    file.close();
}

void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end) {
	double seconds=duration_cast<microseconds>(end-start).count()/1000000.0;
	cout << "Processing time (milliseconds): " << duration_cast<milliseconds>(end - start).count() << endl;
	cout << "Processing time (microseconds): " << duration_cast<microseconds>(end - start).count() << endl;
	cout << "Processing time (nanoseconds): " << duration_cast<nanoseconds>(end - start).count() << endl;
	printf("Processing time (seconds): %.04f\n",seconds);
}

void print_matrix(device_vector<double> &A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            //cout << A[j * nr_rows_A + i] << " ";
            double o=A[j*nr_rows_A+i];
            printf("%.4f ",o);
            //printf("%.10f ",A[j*nr_rows_A+i]);
        }
        cout << endl;
    }
    //cout << endl;
}
