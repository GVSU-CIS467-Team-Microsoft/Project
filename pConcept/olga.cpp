//Artifical Neural Network with Cuda and Cublas matrix version

//Ron Patrick - Capstone GVSU - Winter 2017

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
#include <cublas_v2.h>
#include <curand.h>
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

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr);
void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end);
void print_matrix(device_vector<double> &A, int nr_rows_A, int nr_cols_A);
bool vlRate=false;

template<typename T>
struct KernelArray {    
	T* _array;
	int _size;
};
// Function to convert a KernelArray to a thrust::device_vector
template<typename T>
thrust::host_vector<T> kernelArrToHostVec(KernelArray<T>& kArray) {
    thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(kArray._array);
    thrust::host_vector<T> hVec(dev_ptr, dev_ptr + kArray._size);
    return hVec;
} 

typedef thrust::tuple<ULLI, ULLI> uTuple;
typedef thrust::tuple<double, double> dTuple;
typedef thrust::tuple<ULLI, double, double> tTuple;

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

template<typename T>
struct square {
	__device__ T operator()(const T& x) const { 
		return x * x;
	}
};
struct sigmoid_devrivative_double : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		return t*(1.0-t);
	}
};
struct sigmoid_double : public thrust::unary_function<double, double> {
	sigmoid_double(){}
	__device__ double operator()(double t) {
		return 1.0 / (1.0 + exp(-t));
	}
};
struct exp_double : public thrust::unary_function<double, double> {
	__device__ double operator()(double t) {
		return exp(t);
	}
};
struct softmax_helper : public thrust::unary_function<int, void> {
	double *softmaxTemp;
	double *outputs;
	int batchSize;
	int total;
	softmax_helper(double *_softmaxTemp, double *_outputs, int _batchSize, int _total) : total(_total), outputs(_outputs), softmaxTemp(_softmaxTemp), batchSize(_batchSize) {}
	__device__ void operator()(int t) {
		int local=batchSize;
		double local2=0.0;
		int local3=total;
		int local4=t;
		for(int i=0;i<local3;++i) {
			local2+=softmaxTemp[local4];
			local4+=local;
		}
		outputs[t]=local2;
	}
};
struct softmax_helper2 : public thrust::unary_function<int, void> {
	double *toNext;
	double *softmaxTemp;
	double *outputs;
	int batchSize;
	int total;
	softmax_helper2(double *_toNext, double *_softmaxTemp, double *_outputs, int _batchSize, int _total) : 
		toNext(_toNext), total(_total), outputs(_outputs), softmaxTemp(_softmaxTemp), batchSize(_batchSize) {}

	__device__ void operator()(int t) {
		int local1=t%batchSize;
		//if(outputs[local1]<0.000000001 && outputs[local1]>-0.000000001) {
		//	toNext[t]=0.0;
		//	return;
		//}
		toNext[t]=softmaxTemp[t]/outputs[local1];
		//printf("toNext[%d]: %.15f\tsoftmaxTemp[t]: %.15f\toutputs[%d]: %.15f\n",t,toNext[t],softmaxTemp[t],local1,outputs[local1]);
	}
};

class NN_layer {
public:

	device_vector<double> atNeuronOutputs;
	device_vector<double> atNeuronInputs;
	device_vector<double> weightsMatrix;
	device_vector<double> deltas;
	device_vector<double> derivatives;
	device_vector<double> derivTemp;

	NN_layer(){}
	NN_layer(int sizeThis, int sizeNext, int pBatchSize, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext), batchSize(pBatchSize) {

		if(type!=OUTPUT) {
			weightsMatrix=device_vector<double>(thisSize*nextSize);
		}
	}
	NN_layer(int sizeThis, int sizeNext, int pBatchSize, int pType, cublasHandle_t h) : 
			type(pType), handle(h), thisSize(sizeThis), nextSize(sizeNext), batchSize(pBatchSize) {

		setupLayer(true, h);
	}

	void setupLayer(bool newLayer, cublasHandle_t h) {
		atNeuronOutputs=device_vector<double>(batchSize*thisSize,0.0);
		if(!newLayer) {handle=h;}
		if(type!=INPUT) {
			atNeuronInputs=device_vector<double>(batchSize*thisSize,0.0);
			deltas=device_vector<double>(batchSize*thisSize,0.0);
		} else {
			cudaFree(&atNeuronInputs);
			cudaFree(&deltas);
			cudaFree(&derivatives);
			cudaFree(&derivTemp);
		}
		if(type!=OUTPUT) {
			if(newLayer) {
				weightsMatrix=device_vector<double>(thisSize*nextSize);
				random_doubles(thrust::raw_pointer_cast(&weightsMatrix[0]),thisSize,nextSize);
				thrust::transform(weightsMatrix.begin(),weightsMatrix.end(),weightsMatrix.begin(),fix_random_numbers());
				cout << "thisSize: " << thisSize << " nextSize: " << nextSize << " thisSize*nextSize: " << (thisSize*nextSize) << endl;
			}
			//cudaFree(&softmaxTemp);
			//cudaFree(&softmaxOutputs);
		} else {
			//softmaxTemp=device_vector<double>(batchSize*thisSize);
			//softmaxOutputs=device_vector<double>(batchSize);
			//derivatives=device_vector<double>(batchSize*thisSize,0.0);
			//derivTemp=device_vector<double>(batchSize*thisSize,0.0);			
			cudaFree(&derivatives);
			cudaFree(&derivTemp);
			cudaFree(&weightsMatrix);
		}
		if(type==HIDDEN) {
			derivatives=device_vector<double>(batchSize*thisSize,0.0);
			derivTemp=device_vector<double>(batchSize*thisSize,0.0);
		}
	}

	//void forwardProp(device_vector<double> &toNext, int rows, int cols) {
		/*if(type==INPUT) {
			const double alf=1.0, bet=0.0;
			const double *alpha=&alf, *beta=&bet;
			//cout << "alpha: " << *alpha << " beta: " << *beta << endl;
			//cout << "at input layers weights matrix:\n";
			//print_matrix(weightsMatrix,thisSize,nextSize);
			//cout << "at input layer atNeuronOutputs before Dgemm:\n";
			//print_matrix(atNeuronOutputs,thisSize,batchSize);
			cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, batchSize, nextSize, thisSize, alpha, atNeuronOutputs.data().get(), thisSize, weightsMatrix.data().get(), thisSize, beta, toNext.data().get(), batchSize);
			//cout << "at input layer toNext:\n";
			//print_matrix(toNext,batchSize,nextSize);
			//sleep(10);
			return;
		}*/
		/*if(type==OUTPUT) {
			thrust::transform(atNeuronInputs.begin(),atNeuronInputs.end(),softmaxTemp.begin(),exp_double());
			thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(batchSize),softmax_helper(softmaxTemp.data().get(),softmaxOutputs.data().get(),batchSize,thisSize));
			thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(thisSize*batchSize),softmax_helper2(toNext.data().get(),softmaxTemp.data().get(),softmaxOutputs.data().get(),batchSize,thisSize));
			return;
		}*/
		/*const double alf=1.0, bet=0.0;
		const double *alpha=&alf, *beta=&bet;		
		//printf("In forward prop before sigmoid atNeuronInputs:\n");
		//print_matrix(atNeuronInputs,batchSize,thisSize);		
		thrust::transform(atNeuronInputs.begin(),atNeuronInputs.end(),atNeuronOutputs.begin(),sigmoid_double());
		//printf("In forward prop after sigmoid atNeuronOutputs:\n");
		//print_matrix(atNeuronOutputs,batchSize,thisSize);
		thrust::fill(atNeuronOutputs.begin()+(batchSize*(thisSize-1)),atNeuronOutputs.end(),1.0);
		//printf("In forward prop after fill biases atNeuronOutputs:\n");
		//print_matrix(atNeuronOutputs,batchSize,thisSize);
		//printf("thisSize: %d\n",thisSize);
		//sleep(2);
		thrust::transform(atNeuronOutputs.begin(),atNeuronOutputs.end(),derivTemp.begin(),sigmoid_devrivative_double());
		//thrust::transform(atNeuronInputs.begin(),atNeuronInputs.end(),derivTemp.begin(),sigmoid_devrivative_double());
		//cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, thisSize-1, batchSize, alpha, derivTemp.data().get(), batchSize, beta, derivTemp.data().get(), batchSize, derivatives.data().get(), thisSize-1);
		cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, thisSize, batchSize, alpha, derivTemp.data().get(), batchSize, beta, derivTemp.data().get(), batchSize, derivatives.data().get(), thisSize);
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, nextSize, thisSize, alpha, atNeuronOutputs.data().get(), batchSize, weightsMatrix.data().get(), thisSize, beta, toNext.data().get(), batchSize);
	}*/
	int type, thisSize, nextSize, batchSize;
private:
	cublasHandle_t handle;
};

class neuralNet {
public:
	neuralNet(){}
	neuralNet(string _inFile) : inFile(_inFile) {
		cublasCreate(&handle);
		loadState();
	}
	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), epoch(0), learningRate(0.05), 
		batchSize(pBatchSize) {

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

		maxWeightsMatrix=0;
		for(int i=0;i<layers;++i) {
			if(i<outputsIndex) {
				++hiddenMatrix[i];
			}
			if(i) {
				if(hiddenMatrix[i-1]*hiddenMatrix[i]>maxWeightsMatrix) {
					maxWeightsMatrix=hiddenMatrix[i-1]*hiddenMatrix[i];
				}
			}
		}
		weightsTemp=device_vector<double>(maxWeightsMatrix,0.0);

		NNlayers[0]=NN_layer(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT,handle);
		for(int i=1;i<outputsIndex;++i) {
			NNlayers[i]=NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN,handle);
		}
		NNlayers[outputsIndex]=NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT,handle);
	}

	void train_MatMul(vector<vector<float>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, 
		float RMSwant, vector<UNCHAR> &bLabels, int doDataSetSize, double lRate) {

		if(lRate<0.0) {
			learningRate=.05;
		} else {
			learningRate=lRate;
		}
		double alfLearn = -learningRate;
    	const double betAdd = 1;
    	const double *alphaLearn = &alfLearn;
    	const double *betaAdd = &betAdd;
    	const double alf = 1;
    	const double bet = 0;
    	const double *alpha = &alf;
    	const double *beta = &bet;
    	int batchStart,batchEnd, thisSize, nextSize;
    	RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		dataSetSize=pData.size();
		if(doDataSetSize) {
			dataSetSize=doDataSetSize;
		} else {
			dataSetSize=1000;
		}

		toDivideRMS=((double)dataSetSize)*(double)numOutputs;
		device_vector<double> data[dataSetSize/batchSize];
		device_vector<float> dataTemp[dataSetSize];
		device_vector<double> dataTransposeTemp((itemSize+1)*batchSize);
		device_vector<double> labels[dataSetSize/batchSize];
		device_vector<double> labelsTemp[dataSetSize];
		device_vector<double> batchLabels(numOutputs*batchSize,0.0);
		device_vector<double> outputsTemp(batchSize*numOutputs,0.0);
		device_vector<double> softmaxTemp(batchSize*numOutputs);
		device_vector<double> softmaxOutputs(batchSize);

		float *temp;
		double *tempd;
		ULLI len=pData[0].size();
		ULLI llen=pLabels[0].size();
		for(int i=0;i<dataSetSize;++i) {
			temp=&pData[i][0];
			dataTemp[i]=device_vector<float>(temp, temp+len);
			tempd=&pLabels[i][0];
			labelsTemp[i]=device_vector<double>(tempd, tempd+llen);
		}

		//Creating pre-made batches so I can simply copy them to layer[0]
		cout << "Making batches...\n";
		int whichBatch=0;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			int biasInputIndex=itemSize;
			data[whichBatch]=device_vector<double>((itemSize+1)*batchSize);
			labels[whichBatch]=device_vector<double>(batchSize*numOutputs);
			for(int b=0;b<batchSize;++b) {
				thrust::transform(dataTemp[itemNum+b].begin(),dataTemp[itemNum+b].end(),dataTransposeTemp.begin()+batchStart,floatToDoubleFunctor());
				//thrust::copy(dataTemp[itemNum+b].begin(),dataTemp[itemNum+b].end(),dataTransposeTemp.begin()+batchStart);
				dataTransposeTemp[biasInputIndex]=1.0;
				thrust::copy(labelsTemp[itemNum+b].begin(),labelsTemp[itemNum+b].end(),batchLabels.begin()+batchEnd);
				batchStart+=itemSize+1;
				batchEnd+=numOutputs;
				biasInputIndex+=itemSize+1;
				cudaFree(&dataTemp[itemNum+b]);
				cudaFree(&labelsTemp[itemNum+b]);
			}
			cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, numOutputs, alpha, batchLabels.data().get(), numOutputs, beta, batchLabels.data().get(), numOutputs, labels[whichBatch].data().get(), batchSize);
			cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, itemSize+1, alpha, dataTransposeTemp.data().get(), itemSize+1, beta, dataTransposeTemp.data().get(), itemSize+1, data[whichBatch].data().get(), batchSize);
			++whichBatch;
		}

		cout << "Starting training...\n";
		cudaFree(&dataTransposeTemp);
		cudaFree(&batchLabels);

		thrust::device_vector<double>::iterator iter;
		int position;
		int gotRight=0, maxGotRight=0;
		ULLI neededEpochs=0;
		int tnSize, ttSize, sizeOutputs=numOutputs*batchSize;

		//for(int epochNum=0;epochNum<maxEpochs && (RMSwanted<RMS || RMS!=RMS) && gotRight!=dataSetSize;++epochNum) {
		for(int epochNum=0;epochNum<maxEpochs && gotRight!=dataSetSize;++epochNum) {
			RMS=0.0;
			gotRight=0;
			whichBatch=0;
			for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
				if(batchSize<11) {
					printf("Items %d thru %d\r",itemNum,itemNum+batchSize);
				}

				//Don't need to copy from data[] device_vector to neuron vector, simply start the ANN at the data in [whichBatch]

				//forward propagation
				thisSize=hiddenMatrix[0];
				nextSize=hiddenMatrix[1];
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, nextSize, thisSize, alpha, data[whichBatch].data().get(), batchSize, NNlayers[0].weightsMatrix.data().get(), thisSize, beta, NNlayers[1].atNeuronInputs.data().get(), batchSize);
				for(int i=1;i<outputsIndex;++i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1];					
					//NNlayers[i].forwardProp(NNlayers[i+1].atNeuronInputs,batchSize,hiddenMatrix[i+1]);
					thrust::transform(NNlayers[i].atNeuronInputs.begin(),NNlayers[i].atNeuronInputs.end(),NNlayers[i].atNeuronOutputs.begin(),sigmoid_double());
					thrust::fill(NNlayers[i].atNeuronOutputs.begin()+(batchSize*(thisSize-1)),NNlayers[i].atNeuronOutputs.end(),1.0);
					//thrust::transform(NNlayers[i].atNeuronOutputs.begin(),NNlayers[i].atNeuronOutputs.end(),NNlayers[i].derivTemp.begin(),sigmoid_devrivative_double());
					//thrust::transform(NNlayers[i].atNeuronInputs.begin(),NNlayers[i].atNeuronInputs.end(),NNlayers[i].derivTemp.begin(),sigmoid_devrivative_double());
					//cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, thisSize, batchSize, alpha, NNlayers[i].derivTemp.data().get(), batchSize, beta, NNlayers[i].derivTemp.data().get(), batchSize, NNlayers[i].derivatives.data().get(), thisSize);
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batchSize, nextSize, thisSize, alpha, NNlayers[i].atNeuronOutputs.data().get(), batchSize, NNlayers[i].weightsMatrix.data().get(), thisSize, beta, NNlayers[i+1].atNeuronInputs.data().get(), batchSize);
				}
				//NNlayers[outputsIndex].forwardProp(NNlayers[outputsIndex].atNeuronOutputs,batchSize,numOutputs);
				//thrust::transform(NNlayers[outputsIndex].atNeuronInputs.begin(),NNlayers[outputsIndex].atNeuronInputs.end(),NNlayers[outputsIndex].atNeuronOutputs.begin(),sigmoid_double());
				//thrust::transform(NNlayers[outputsIndex].atNeuronOutputs.begin(),NNlayers[outputsIndex].atNeuronOutputs.end(),NNlayers[outputsIndex].derivTemp.begin(),sigmoid_devrivative_double());
				//thrust::transform(NNlayers[outputsIndex].atNeuronInputs.begin(),NNlayers[outputsIndex].atNeuronInputs.end(),NNlayers[outputsIndex].derivTemp.begin(),sigmoid_devrivative_double());
				//cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, numOutputs, batchSize, alpha, NNlayers[outputsIndex].derivTemp.data().get(), batchSize, beta, NNlayers[outputsIndex].derivTemp.data().get(), batchSize, NNlayers[outputsIndex].derivatives.data().get(), numOutputs);

				thrust::transform(NNlayers[outputsIndex].atNeuronInputs.begin(),NNlayers[outputsIndex].atNeuronInputs.end(),softmaxTemp.begin(),exp_double());
				thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(batchSize),softmax_helper(softmaxTemp.data().get(),softmaxOutputs.data().get(),batchSize,numOutputs));
				thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(sizeOutputs),softmax_helper2(NNlayers[outputsIndex].atNeuronOutputs.data().get(),softmaxTemp.data().get(),softmaxOutputs.data().get(),batchSize,numOutputs));

				//first check how many we got right
				batchStart=0;
				batchEnd=numOutputs;
				//printf("\nbatch starting at: %d\n",itemNum);
				cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, numOutputs, batchSize, alpha, NNlayers[outputsIndex].atNeuronOutputs.data().get(), batchSize, beta, NNlayers[outputsIndex].atNeuronOutputs.data().get(), batchSize, outputsTemp.data().get(), numOutputs);
				for(int b=0;b<batchSize;++b) {
					if(doMNISTprob) {
						iter = thrust::max_element(outputsTemp.begin()+batchStart, outputsTemp.begin()+batchEnd);
						position = iter - outputsTemp.begin();
						position -= batchStart;
						/*printf("output: %d expected: %d\n",position,bLabels[itemNum+b]);
						for(int ot=batchStart;ot<batchEnd;++ot) {
							double oo=outputsTemp[ot];
							printf("%.5f ",oo);
						}
						printf("\n");//*/
						if(position==bLabels[itemNum+b]) {
							++gotRight;
						}
					}
					if(doBinaryProb) {
						cout << "outputs: ";
						for(int j=batchStart;j<batchEnd;++j) {
							double oo=outputsTemp[j];
							printf("%.6f ",oo);
						}
						cout << endl << "Expectd: ";
						for(auto l:labels[itemNum+b]) {
							double ll=l;
							printf("%.6f ",ll);
						}
						cout << endl;
					}
					batchStart=batchEnd;
					batchEnd+=numOutputs;					
				}

				//Backward propagation
				thrust::transform(NNlayers[outputsIndex].atNeuronOutputs.begin(),NNlayers[outputsIndex].atNeuronOutputs.end(),labels[whichBatch].begin(),outputsTemp.begin(),thrust::minus<double>());
				cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, numOutputs, batchSize, alpha, outputsTemp.data().get(), batchSize, beta, outputsTemp.data().get(), batchSize, NNlayers[outputsIndex].deltas.data().get(), numOutputs);
				//thrust::transform(NNlayers[outputsIndex].derivatives.begin(),NNlayers[outputsIndex].derivatives.end(),NNlayers[outputsIndex].deltas.begin(),NNlayers[outputsIndex].deltas.begin(),thrust::multiplies<double>());

				int i=outputsIndex-1;
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[i+1];
				ttSize=thisSize-1;
				tnSize=nextSize-1;
				cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nextSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), thisSize, beta, NNlayers[i].weightsMatrix.data().get(), thisSize, weightsTemp.data().get(), nextSize);
				cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ttSize, batchSize, nextSize, alpha, weightsTemp.data().get(), nextSize, NNlayers[i+1].deltas.data().get(), nextSize, beta, NNlayers[i].deltas.data().get(), ttSize);
				//thrust::transform(NNlayers[i].derivatives.begin(),NNlayers[i].derivatives.end(),NNlayers[i].deltas.begin(),NNlayers[i].deltas.begin(),thrust::multiplies<double>());				
				--i;
				for(;i;--i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1];
					tnSize=nextSize-1;
					ttSize=thisSize-1;
					cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, nextSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), thisSize, beta, NNlayers[i].weightsMatrix.data().get(), thisSize, weightsTemp.data().get(), nextSize);
					cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ttSize, batchSize, tnSize, alpha, weightsTemp.data().get(), nextSize, NNlayers[i+1].deltas.data().get(), tnSize, beta, NNlayers[i].deltas.data().get(), ttSize);
					//thrust::transform(NNlayers[i].derivatives.begin(),NNlayers[i].derivatives.end(),NNlayers[i].deltas.begin(),NNlayers[i].deltas.begin(),thrust::multiplies<double>());
				}

				int j=outputsIndex-1;
				thisSize=hiddenMatrix[0];
				nextSize=hiddenMatrix[1]-1;				
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, thisSize, batchSize, alpha, NNlayers[1].deltas.data().get(), nextSize, data[whichBatch].data().get(), batchSize, beta, weightsTemp.data().get(), nextSize);
				cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, nextSize, alphaLearn, weightsTemp.data().get(), nextSize, betaAdd, NNlayers[0].weightsMatrix.data().get(), thisSize, NNlayers[0].weightsMatrix.data().get(), thisSize);				
				for(int i=1;i<j;++i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1]-1;
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, thisSize, batchSize, alpha, NNlayers[i+1].deltas.data().get(), nextSize, NNlayers[i].atNeuronOutputs.data().get(), batchSize, beta, weightsTemp.data().get(), nextSize);
					cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, nextSize, alphaLearn, weightsTemp.data().get(), nextSize, betaAdd, NNlayers[i].weightsMatrix.data().get(), thisSize, NNlayers[i].weightsMatrix.data().get(), thisSize);
				}
				thisSize=hiddenMatrix[j];
				nextSize=hiddenMatrix[j+1];
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, thisSize, batchSize, alpha, NNlayers[j+1].deltas.data().get(), nextSize, NNlayers[j].atNeuronOutputs.data().get(), batchSize, beta, weightsTemp.data().get(), nextSize);
				cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, nextSize, alphaLearn, weightsTemp.data().get(), nextSize, betaAdd, NNlayers[j].weightsMatrix.data().get(), thisSize, NNlayers[j].weightsMatrix.data().get(), thisSize);				
				++whichBatch;
			}
			if(batchSize<11) {printf("\n");}
			//printf("Iteration: %d of %llu -- Got %d of %d correct -- learningRate: %.15f\n",epochNum,maxEpochs,gotRight,dataSetSize,*alphaLearn);
			if(gotRight>maxGotRight){maxGotRight=gotRight;}
			printf("Iteration: %d -- Got %d of %d -- max right: %d -- lRate: %.10f\n",epochNum,gotRight,dataSetSize,maxGotRight,*alphaLearn);
			if(vlRate) {
				//if(epochNum>1) {
					double percLearned=(double)gotRight/(double)dataSetSize;
					if(percLearned<0.99 && percLearned>0.9) {
						percLearned=(1.0-percLearned)*9.5;
						//percLearned=(1.0-percLearned)*2.0;
						//percLearned=(1.0-percLearned)/2.0;
						//percLearned=pow(1.0-percLearned,(double)layers);
						//percLearned=pow(1.0-percLearned,2.0);
						//alfLearn=-(percLearned*(learningRate/2.0)+(learningRate/2.0));
						alfLearn=-(percLearned*learningRate);
					} else {
						if(percLearned<0.9) {
							alfLearn=-learningRate;
						}
					}
				//}
			}
			neededEpochs=epochNum;
		}
		cout << "Epochs needed: " << neededEpochs << endl;
		saveState("weights");
		cublasDestroy(handle);
	}

	void saveState(string outFile) {
		outFile+="-"+to_string(dataSetSize);
		cout << "Writing weights to file: " << outFile << endl;
		ofstream oFile(outFile, ios::binary|ios::out);
		if(oFile.is_open()) {
			oFile.write((char*)&epoch,sizeof(ULLI));
			oFile.write((char*)&layers,sizeof(ULLI));
			oFile.write((char*)&maxWeightsMatrix,sizeof(ULLI));
			for(int i=0;i<hiddenMatrix.size();++i) {
				oFile.write((char*)&hiddenMatrix[i],sizeof(int));
			}
			oFile.write((char*)&batchSize,sizeof(int));
			oFile.write((char*)&learningRate,sizeof(double));
			for(int i=0;i<outputsIndex;++i) {
				int end=NNlayers[i].thisSize*NNlayers[i].nextSize;
				for(int j=0;j<end;++j) {
					double o=NNlayers[i].weightsMatrix[j];
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
			oFile.read((char*)&maxWeightsMatrix,sizeof(ULLI));
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
			weightsTemp=device_vector<double>(maxWeightsMatrix,0.0);

			NNlayers.clear();
			int type=INPUT;
			for(int i=0;i<outputsIndex;++i) {
				if(i){type=HIDDEN;}
				NNlayers.push_back(NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,type));
				int end=NNlayers[i].thisSize*NNlayers[i].nextSize;
				for(int j=0;j<end;++j) {
					double o=0.0;
					oFile.read((char*)&o,sizeof(double));
					NNlayers[i].weightsMatrix.push_back(o);
				}
				NNlayers[i].setupLayer(false, handle);
			}
			NNlayers.push_back(NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT));
			NNlayers.back().setupLayer(false, handle);
			oFile.close();
		}
		cout << "Done\n";
	}
	vector<NN_layer> NNlayers;

private:
	ULLI epoch, maxElement, layers, maxEpochs, maxWeightsMatrix;
	int outputsIndex, dataSetSize, numInputs, numOutputs, batchSize;
	double RMS, minRMS, toDivideRMS, RMSwanted, learningRate;
	vector<int> hiddenMatrix;
	cublasHandle_t handle;
	ULLI itemSize;
	string inFile;

	vector<vector<double>> neuralNet_weights_host;
	device_vector<double> weightsTemp;
};

void doMain(int my_rank, string hostname, int num_nodes, vector<int> &inputHiddenLayers, int batchSize, int doDataSetSize, double lRate, string inFile, string outFile) {

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

		vector<vector<float>> testData(10000);
		ReadMNIST_float("t10k-images.idx3-ubyte",10000,784,testData);
		vector<vector<float>> trainData(60000);
		ReadMNIST_float("train-images.idx3-ubyte",60000,784,trainData);
		vector<vector<double>> testLabels(10000);
		vector<vector<double>> trainLabels(60000);
		//vector<vector<float>> testLabels(10000);
		//vector<vector<float>> trainLabels(60000);	
		vector<UNCHAR> testLabels2;//(10000);
		vector<UNCHAR> trainLabels2;//(60000);
		ifstream file("t10k-labels.idx1-ubyte",ios::binary);
		if(file.is_open()) {
			int placeHolder=0;
			file.read((char*)&placeHolder,sizeof(placeHolder));
			file.read((char*)&placeHolder,sizeof(placeHolder));
			for(int i=0;i<10000;++i) {
				testLabels[i]=vector<double>(10,0.0);
				//testLabels[i]=vector<float>(10,0.0f);
				UNCHAR temp=0;
				file.read((char*)&temp,1);
				for(UNCHAR j=0;j<10;++j) {
					if(j==temp) {
						//testLabels[i].push_back(1.0);
						//testLabels[i][j]=1.0f;
						testLabels[i][j]=1.0;
						testLabels2.push_back(temp);
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
				//trainLabels[i]=vector<float>(10,0.0f);
				UNCHAR temp=0;
				file2.read((char*)&temp,1);
				for(UNCHAR j=0;j<10;++j) {
					if(j==temp) {
						//trainLabels[i].push_back(1.0);
						//trainLabels[i][j]=1.0f;
						trainLabels[i][j]=1.0;
						trainLabels2.push_back(temp);
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
			go=neuralNet(784,10,hiddenMatrix,batchSize);
		} else {
			go=neuralNet(inFile);
		}
		auto start = high_resolution_clock::now();
		//go.train_floats(trainData,trainLabels,1000000,0.0001,trainLabels2);
		//go.train(trainData,trainLabels,1000000,0.0001,trainLabels2, doDataSetSize);//*/
		go.train_MatMul(trainData,trainLabels,1000000,0.0001,trainLabels2, doDataSetSize, lRate);//*/
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
		neuralNet test(BITS,BITS,hiddenMatrix,batchSize);
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
		vector<UNCHAR> placeHolder;
		test.train_MatMul(countingTest,countingLabels,1000000,0.00001,placeHolder,size,lRate);
	}
}

int main(int argc, char *argv[]) {


	///////  Save and load weights test code
	/*vector<int> hMatrix;
	hMatrix.push_back(131);
	hMatrix.push_back(137);
	neuralNet test(341,10,hMatrix,10);
	hMatrix.insert(hMatrix.begin(),341);
	hMatrix.push_back(10);
	for(int i=0;i<3;++i){++hMatrix[i];}
	for(int i=0;i<3;++i) {
		for(int j=0;j<(hMatrix[i]*hMatrix[i+1]);++j) {
			test.NNlayers[i].weightsMatrix[j]=(double)j;
		}
	}
	cout << "from original\n";
	for(int i=0;i<20;++i) {
		cout << test.NNlayers[1].weightsMatrix[i] << " ";
	}
	cout << endl;
	cout << "from file\n";
	test.saveState("weights");
	neuralNet test2("weights");
	for(int i=0;i<test2.NNlayers.size();++i) {
		cout << test2.NNlayers[i].thisSize << endl;
	}
	for(int i=0;i<20;++i) {
		cout << test.NNlayers[1].weightsMatrix[i] << endl;
	}	
	return 0;//*/

	///		softmax tester code ----- 
	/*int rows=4,cols=5,rc=rows*cols;
	const double alfl=0.5;
	const double betAdd=1.0;
	const double *alphaLearn=&alfl;
	const double *betaAdd=&betAdd;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;	
	device_vector<double> testt(rc);
	device_vector<double> testt2(rc);
	for(int i=0;i<rc;++i) {
		testt[i]=(double)i;
		testt2[i]=(double)i;
	}
	cublasHandle_t handle;
	cublasCreate(&handle);
	device_vector<double> softmaxTemp(rc);
	device_vector<double> softmaxOutputs(rows);
	cout << "testt not transposed:\n";
	print_matrix(testt,rows,cols);
	cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, cols, rows, alpha, testt.data().get(), rows, beta, testt.data().get(), rows, testt2.data().get(), cols);
	thrust::copy(testt2.begin(),testt2.end(),testt.begin());
	for(int i=0;i<rc;++i) {
		testt2[i]=(double)i;
	}	
	cout << "testt transposed:\n";
	print_matrix(testt,cols,rows);
	cout << "testt2 original:\n";
	print_matrix(testt2,rows,cols);
	cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, alphaLearn, testt.data().get(), cols, betaAdd, testt2.data().get(), rows, testt2.data().get(), rows);
	cout << "testt2 with testt transposed back and then added to testt2:\n";
	print_matrix(testt2,rows,cols);
	return 0;
	thrust::transform(testt2.begin(),testt2.end(),softmaxTemp.begin(),exp_double());
	thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(rows),softmax_helper(softmaxTemp.data().get(),softmaxOutputs.data().get(),rows,cols));
	thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(rc),softmax_helper2(testt.data().get(),softmaxTemp.data().get(),softmaxOutputs.data().get(),rows,cols));
	cout << "second:\n";
	print_matrix(testt,rows,cols);
	cublasDestroy(handle);
	return 0;//*/

	string inFile="";
	string outFile="";
	int doDataSetSize=0;
	int batchSize=5;
	if(doBinaryProb) {
		batchSize=4;
	}
	double lRate=-1.0;
	vector<int> inputHiddenLayers;
    for(int i=1;i<argc;++i) {
        string temp=string(argv[i]);
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
			//cout << "Total Cuda Memory: " << totalCudaMem << endl;
		}
	}

	doMain(my_rank, hostname, num_nodes, inputHiddenLayers, batchSize, doDataSetSize, lRate, inFile, outFile);

	MPI_Finalize();
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