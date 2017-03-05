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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <mpi.h>
#include "helper_cuda.h"
#include "helper_string.h"
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

int memoryTracker=0;

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr);
void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end);
void print_matrix(device_vector<double> &A, int nr_rows_A, int nr_cols_A);

typedef thrust::tuple<ULLI, ULLI> uTuple;
typedef thrust::tuple<double, double> dTuple;
typedef thrust::tuple<ULLI, double, double> tTuple;
typedef thrust::device_vector<double>::iterator doubleIterator;
typedef thrust::tuple<doubleIterator, doubleIterator> iterTuple;
typedef thrust::zip_iterator<iterTuple> zipIterator;

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
struct backProp_helper2 : public thrust::unary_function<thrust::tuple<double, double>, double> {
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

class NN_layer {
public:

	device_vector<double> atNeuronOutputs;
	device_vector<double> atNeuronInputs;
	device_vector<double> weightsMatrix;
	device_vector<double> biases;

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
		if(newLayer) {
			if(type!=INPUT) {
				atNeuronInputs=device_vector<double>(batchSize*thisSize,0.0);
				memTracker(allN*8,false);
				biases=device_vector<double>(thisSize*batchSize,0.0);
				memTracker(allN*8,false);
			} else {
				cudaFree(&atNeuronInputs);
				cudaFree(&biases);
			}
			if(type!=OUTPUT) {
				weightsMatrix=device_vector<double>(thisSize*nextSize);
				memTracker(allW*8,false);
				random_doubles(thrust::raw_pointer_cast(&weightsMatrix[0]),thisSize,nextSize);
				thrust::transform(weightsMatrix.begin(),weightsMatrix.end(),weightsMatrix.begin(),fix_random_numbers());
				cout << "thisSize: " << thisSize << " nextSize: " << nextSize << " thisSize*nextSize: " << (thisSize*nextSize) << endl;
			} else {
				cudaFree(&weightsMatrix);
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
	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize) {

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

		/*maxWeightsMatrix=0;
		maxDeltaMatrix=0;
		for(int i=0;i<layers;++i) {
			if(maxWeightsMatrix<hiddenMatrix[i]*batchSize) {
				maxWeightsMatrix=hiddenMatrix[i]*batchSize;
			}
			if(i) {
				if(hiddenMatrix[i-1]*hiddenMatrix[i]>maxWeightsMatrix) {
					maxWeightsMatrix=hiddenMatrix[i-1]*hiddenMatrix[i];
				}
			}
		}*/
		//weightsTemp=device_vector<double>(maxWeightsMatrix,0.0);
		//deltaTemp=device_vector<double>(maxWeightsMatrix);
		//memoryTracker+=maxWeightsMatrix*8;

		NNlayers[0]=NN_layer(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT);
		for(int i=1;i<outputsIndex;++i) {
			NNlayers[i]=NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN);
		}
		NNlayers[outputsIndex]=NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT);
	}

	void train_MatMul(vector<vector<float>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, double lRate, vector<vector<float>> &pTestData, vector<vector<double>> &pTestLabels, bool vlRate) {

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
			doDataSetSize=1000;
		}
		dataSetSize=doDataSetSize;
		//double alfLearn = -learningRate;
		//const double betAdd = 1;
		//const double *alphaLearn = &alfLearn;
		//const double *betaAdd = &betAdd;
		const double alf = 1;
		const double bet = 0;
		const double *alpha = &alf;
		const double *beta = &bet;
		int batchStart,batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		//dataSetSize=pData.size();
		int testSetSize=pTestData.size();
		vector<UNCHAR> btLabels;
		device_vector<double> testData[testSetSize/batchSize];
		//device_vector<double> testLabels[testSetSize];		
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back((UNCHAR)(thrust::max_element(p.begin(), p.end())-p.begin()));
			}
		} else {
			cudaFree(&testData);
			//cudaFree(&testLabels);
		}
		//cout << "testSetSize: " << testSetSize << endl;

		//toDivideRMS=((double)dataSetSize)*(double)numOutputs;
		device_vector<double> data[dataSetSize/batchSize];
		//device_vector<float> dataTemp;//[dataSetSize];
		//device_vector<double> dataTransposeTemp(itemSize*batchSize);
		device_vector<double> labels[dataSetSize/batchSize];
		//device_vector<double> labelsTemp;//[dataSetSize];
		//device_vector<double> batchLabels(numOutputs*batchSize,0.0);
		//device_vector<double> outputsTemp(batchSize*numOutputs,0.0);
		//device_vector<double> outerDeltaB[layers-1];
		//device_vector<double> outerDeltaW[layers-1];
		device_vector<double> innerDeltaB[layers-1];
		device_vector<double> innerDeltaW[layers-1];

		for(int i=0;i<outputsIndex;++i){
			//outerDeltaW[i]=device_vector<double>(NNlayers[i].allW);
			innerDeltaW[i]=device_vector<double>(NNlayers[i].allW,0.0);
			memTracker(NNlayers[i].allW*8,false);
			//outerDeltaB[i]=device_vector<double>(NNlayers[i+1].allN);
			innerDeltaB[i]=device_vector<double>(NNlayers[i+1].allN,0.0);
			memTracker(NNlayers[i+1].allN*8,false);
		}

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
		cout << "Making batches...\n";
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
		//int numBatches=dataSetSize/batchSize;
		toDivideRMS=learningRate/(double)batchSize;
		//toDivideRMS=learningRate/(double)numBatches;
		//cout << "toDivideRMS: " << toDivideRMS << endl;
		int maxGotRight=0, maxTestRight=-1, ii;
		device_vector<double> which;
		double origLearningRate=learningRate, seconds, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		//int prevMax=0;

		for(int epochNum=0;epochNum<maxEpochs && maxGotRight!=dataSetSize && maxTestRight!=testSetSize;++epochNum) {
			RMS=0.0;
			gotRight=0;
			whichBatch=0;
			/*for(int i=0;i<outputsIndex;++i) {
				thrust::fill(outerDeltaB[i].begin(),outerDeltaB[i].end(),0.0);
				thrust::fill(outerDeltaW[i].begin(),outerDeltaW[i].end(),0.0);
			}//*/
			startTime=high_resolution_clock::now();
			for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
				if(batchSize<11) {
					printf("Items %d thru %d\r",itemNum,itemNum+batchSize);
				}

				//forward propagation
				which=data[whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, which.data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
					thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[ii].allN),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayers[ii].atNeuronInputs.data().get(),NNlayers[ii].biases.data().get()));
					//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[ii].allN),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed[ii]);
					//thrust::transform(fBegin[ii],fEnd[ii],NNlayers[ii].atNeuronOutputs.begin(),forwardFeed);
					which=NNlayers[ii].atNeuronOutputs;
					//double local=inputs[t];
					//local+=biases[t];
					//inputs[t]=local;
					//return 1.0/(1.0+exp(-local));
				}

				//first check how many we got right
				/*batchStart=0;
				batchEnd=numOutputs;
				//printf("\nbatch starting at: %d\n",itemNum);
				for(int b=0;b<batchSize;++b) {
					//iter = thrust::max_element(outputsTemp.begin()+batchStart, outputsTemp.begin()+batchEnd);
					//position = iter - outputsTemp.begin();
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
				}//*/

				//Backward propagation
				mOut=outputsIndex-1;
				prevSize=hiddenMatrix[mOut];
				thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[outputsIndex].allN),innerDeltaB[mOut].begin(),output_helper(NNlayers[outputsIndex].atNeuronOutputs.data().get(),NNlayers[outputsIndex].atNeuronInputs.data().get(),innerDeltaB[mOut].data().get(),labels[whichBatch].data().get()));
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, numOutputs, prevSize, batchSize, alpha, innerDeltaB[mOut].data().get(), numOutputs, NNlayers[mOut].atNeuronOutputs.data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), numOutputs);

				--mOut;
				for(int i=outputsIndex-1;i;--i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1];
					prevSize=hiddenMatrix[i-1];
					cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, thisSize, batchSize, nextSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, innerDeltaB[mOut+1].data().get(), nextSize, beta, innerDeltaB[mOut].data().get(), thisSize);
					if(i!=1) {
						//zipIterator begin(thrust::make_tuple(NNlayers[i].atNeuronOutputs.begin(), innerDeltaB[mOut].begin()));
						//zipIterator end(thrust::make_tuple(NNlayers[i].atNeuronOutputs.end(), innerDeltaB[mOut].end()));
						//thrust::transform(begin,end,innerDeltaB[mOut].begin(),backProp_helper2());
						//thrust::transform(begin2[i],end2[i],innerDeltaB[mOut].begin(),backProp2);
						thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allN),innerDeltaB[mOut].begin(),backProp_helper2(NNlayers[i].atNeuronOutputs.data().get(),innerDeltaB[mOut].data().get()));
						cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, innerDeltaB[mOut].data().get(), thisSize, NNlayers[i-1].atNeuronOutputs.data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), thisSize);
					} else {
						//zipIterator begin(thrust::make_tuple(NNlayers[i].atNeuronInput.begin(), innerDeltaB[mOut].begin()));
						//zipIterator end(thrust::make_tuple(NNlayers[i].atNeuronInputs.end(), innerDeltaB[mOut].end()));
						//thrust::transform(begin,end,innerDeltaB[mOut].begin(),backProp_helper());
						//thrust::transform(begin1[i],end1[i],innerDeltaB[mOut].begin(),backProp);
						thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allN),innerDeltaB[mOut].begin(),backProp_helper(innerDeltaB[mOut].data().get(),NNlayers[i].atNeuronInputs.data().get()));
						cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, thisSize, prevSize, batchSize, alpha, innerDeltaB[mOut].data().get(), thisSize, data[whichBatch].data().get(), prevSize, beta, innerDeltaW[mOut].data().get(), thisSize);
					}
					--mOut;
				}
				/*for(int i=0;i<outputsIndex;++i) {
					thrust::transform(innerDeltaB[i].begin(),innerDeltaB[i].end(),outerDeltaB[i].begin(),outerDeltaB[i].begin(),thrust::plus<double>());
					thrust::transform(innerDeltaW[i].begin(),innerDeltaW[i].end(),outerDeltaW[i].begin(),outerDeltaW[i].begin(),thrust::plus<double>());
				}//*/
				for(int i=0;i<outputsIndex;++i) {
					thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allW),update_w(NNlayers[i].weightsMatrix.data().get(),innerDeltaW[i].data().get(),toDivideRMS));
					thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i+1].allN),update_b(NNlayers[i+1].biases.data().get(),innerDeltaB[i].data().get(),toDivideRMS));
				}//*/
				++whichBatch;
			}
			/*for(int i=0;i<outputsIndex;++i) {
				thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i].allW),update_w(NNlayers[i].weightsMatrix.data().get(),outerDeltaW[i].data().get(),toDivideRMS));
				thrust::for_each(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[i+1].allN),update_b(NNlayers[i+1].biases.data().get(),outerDeltaB[i].data().get(),toDivideRMS));
			}//*/
			if(batchSize<11) {printf("\n");}
			//printf("Iteration: %d of %llu -- Got %d of %d correct -- learningRate: %.15f\n",epochNum,maxEpochs,gotRight,dataSetSize,*alphaLearn);
			if(gotRight>maxGotRight){maxGotRight=gotRight;}
			//printf("Training epoch: %d -- Got %d of %d -- max right: %d -- lRate: %.5f",epochNum,gotRight,dataSetSize,maxGotRight,*alphaLearn);
			printf("Training epoch: %d -- lRate: %.5f",epochNum,learningRate);//*alphaLearn);
			if(!testSetSize) {
				printf("\n");
			} else {
				printf(" -- ");
			}

			gotRight=0;
			whichBatch=0;
			for(int t=0;t<testSetSize;t+=batchSize) {
				which=testData[whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nextSize, batchSize, thisSize, alpha, NNlayers[i].weightsMatrix.data().get(), nextSize, which.data().get(), thisSize, beta, NNlayers[ii].atNeuronInputs.data().get(), nextSize);
					thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[ii].allN),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(NNlayers[ii].atNeuronInputs.data().get(),NNlayers[ii].biases.data().get()));
					//thrust::transform(thrust::make_counting_iterator(0),thrust::make_counting_iterator(NNlayers[ii].allN),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed[ii]);
					which=NNlayers[ii].atNeuronOutputs;
				}
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
				++whichBatch;
			}
			if(gotRight>maxTestRight && testSetSize){maxTestRight=gotRight;}
			if(vlRate) {
				//if(epochNum>1) {
					double cutOff=0.94;
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
			printf("Testing -- Got %d of %d -- max right: %d -- sec: %.5f -- totalTime: %.5f\n",gotRight,testSetSize,maxTestRight,seconds,totalTime);
		}
		saveState("Olga2-");
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

private:
	ULLI epoch, maxElement, layers, maxEpochs;//, maxWeightsMatrix, maxDeltaMatrix;
	int outputsIndex, dataSetSize, numInputs, numOutputs, batchSize;
	double RMS, minRMS, toDivideRMS, RMSwanted, learningRate;
	vector<int> hiddenMatrix;
	cublasHandle_t handle;
	ULLI itemSize;
	string inFile;
	ULLI neededEpochs;

	vector<vector<double>> neuralNet_weights_host;
	//device_vector<double> weightsTemp;
	//device_vector<double> deltaTemp;
};

void doMain(int my_rank, string hostname, int num_nodes, vector<int> &inputHiddenLayers, int batchSize, int doDataSetSize, double lRate, string inFile, string outFile, bool vlRate) {

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
		//vector<UNCHAR> testLabels2;//(10000);
		//vector<UNCHAR> trainLabels2;//(60000);
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
				//trainLabels[i]=vector<float>(10,0.0f);
				UNCHAR temp=0;
				file2.read((char*)&temp,1);
				for(UNCHAR j=0;j<10;++j) {
					if(j==temp) {
						//trainLabels[i].push_back(1.0);
						//trainLabels[i][j]=1.0f;
						trainLabels[i][j]=1.0;
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
			go=neuralNet(784,10,hiddenMatrix,batchSize);
		} else {
			go=neuralNet(inFile);
		}
		auto start = high_resolution_clock::now();
		//go.train_floats(trainData,trainLabels,1000000,0.0001,trainLabels2);
		//go.train(trainData,trainLabels,1000000,0.0001,trainLabels2, doDataSetSize);//*/
		go.train_MatMul(trainData,trainLabels, 1000000, 0.0001, doDataSetSize, lRate, testData, testLabels, vlRate);//*/
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
		test.train_MatMul(countingTest,countingLabels,1000000,0.00001,size,lRate,countingTest,countingLabels,vlRate);
	}
}

int main(int argc, char *argv[]) {

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

	doMain(my_rank, hostname, num_nodes, inputHiddenLayers, batchSize, doDataSetSize, lRate, inFile, outFile, vlRate);

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