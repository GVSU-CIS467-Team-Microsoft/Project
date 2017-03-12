//Artifical Neural Network with Cuda and Cublas matrix version

//Ron Patrick - Capstone GVSU - Winter 2017

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mpi.h>
#include <cmath>
#include <numeric>
#include <limits.h>
#include <float.h>
#include <random>
#include <imebra/imebra.h>
#include <pthread.h>
//#include <openblas/cblas.h>
#include "cblas.h"
#include <signal.h>
using namespace std;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define ULLI unsigned long long int
#define UNCHAR unsigned char
#define INPUT 0
#define OUTPUT 1
#define HIDDEN 2
#define MASTER  0
#define TAG     0

int memoryTracker=0;
int showInterval=0;
pthread_mutex_t crmutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t crmutex2 = PTHREAD_MUTEX_INITIALIZER;
bool threadExit=false;
static pthread_barrier_t barrier;
static pthread_barrier_t barrier2;
int numberOfProcessors;
int bitBarrier1, bitBarrier2, allSig;
int MPIbitBarrier1, MPIbitBarrier2, MPIallSig;
int my_rank, num_nodes, startBatch, endBatch, totalNum;
string hostname;

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr);
void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end);
void print_matrix(vector<double> &A, int nr_rows_A, int nr_cols_A);

/*typedef tuple<ULLI, ULLI> uTuple;
typedef tuple<double, double> dTuple;
typedef tuple<ULLI, double, double> tTuple;
typedef vector<double>::iterator doubleIterator;
typedef tuple<doubleIterator, doubleIterator> iterTuple;
typedef zip_iterator<iterTuple> zipIterator;*/

void ctrlchandler(int sig) {
	printf("\nTrying to exit...\n");
	threadExit=true;
}

void memTracker(int in, bool printIt) {
	memoryTracker+=in;
	if(printIt) {
		cout << hostname << ": memory tracker: Using(bytes): " << memoryTracker << " ";
		cout << "(Kb): " << (memoryTracker/1024) << " ";
		cout << "(Mb): " << ((memoryTracker/1024)/1024) << endl;
	}
}

struct floatToDoubleFunctor : public unary_function<float,double> {
	double operator()(float t) {
		return (double)t;
	}
};
struct fix_random_numbers : public unary_function<double, double> {
	double operator()(double t) {
		t=(double)rand()/RAND_MAX;
		return (t*2.0)-1.0;
	}
};
struct fix_random_numbers2 : public unary_function<double, double> {
	double operator()(double t) {
		return (double)rand()/RAND_MAX;
		//return (t*2.0)-1.0;
	}
};

/*void random_floats(float *A, int rowsA, int colsA) {
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
}*/

struct update_w : public unary_function<int, void> {
	double *weights;
	double *newW;
	double lRate;
	update_w(double *w, double *_newW, double lr) : weights(w), newW(_newW), lRate(lr){}
	void operator()(int t) {
		double local=weights[t];
		double local2=lRate;
		double local3=newW[t];
		double local4=local-local2*local3;
		weights[t]=local4;
		//weights[t]=weights[t]-lRate*newW[t];
	}
};
struct update_b : public unary_function<int, void> {
	double *biases;
	double *newB;
	double lRate;
	update_b(double *b, double *_newB, double lr) : biases(b), newB(_newB), lRate(lr){}
	void operator()(int t) {
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
	T operator()(const T& x) const { 
		return x * x;
	}
};
struct sigmoid_devrivative : public unary_function<double, double> {
	double operator()(double t) {
		double tt=1.0/(1.0+exp(-t));
		return tt*(1.0-tt);
	}
};
struct sigmoid : public unary_function<double, double> {
	sigmoid(){}
	double operator()(double t) {
		return 1.0 / (1.0 + exp(-t));
	}
};
struct exp_double : public unary_function<double, double> {
	double operator()(double t) {
		return exp(t);
	}
};

struct forwardFeed_helper : public unary_function<int, double> {
	double *inputs;
	double *biases;
	forwardFeed_helper(){}
	forwardFeed_helper(double *_inputs, double* _biases) : inputs(_inputs), biases(_biases){}
	//double operator()(tuple<double, double> t) {
	double operator()(int t) {
		//cout << "ffhelper: t: " << t << endl;
		double local=inputs[t];
		local+=biases[t];
		inputs[t]=local;
		return 1.0/(1.0+exp(-local));
	}
};
struct backProp_helper : public unary_function<int, double> {
	double *innerDelta;
	double *inputs;
	backProp_helper(){}
	backProp_helper(double* _innerDelta, double *_inputs) : innerDelta(_innerDelta), inputs(_inputs){}
	double operator()(int t) {
		double local=1.0/(1.0+exp(-inputs[t]));
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
	/*double operator()(tuple<double, double> t) {
		double local=1.0/(1.0+exp(-get<0>(t)));
		local=local*(1.0-local);
		return get<1>(t)*local;
	}*/
};
struct backProp_helper2 : public unary_function<tuple<double, double>, double> {
	double *outputs;
	double *innerDelta;
	backProp_helper2(){}
	backProp_helper2(double *_outputs, double* _innerDelta) : innerDelta(_innerDelta), outputs(_outputs){}
	double operator()(int t) {
		double local=outputs[t];
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
	/*double operator()(tuple<double, double> t) {
		double local=get<0>(t);
		local=local*(1.0-local);
		return get<1>(t)*local;
	}*/
};
struct output_helper : public unary_function<int, double> {
	double *inputs;
	double *outputs;
	double *labels;
	double *innerDelta;
	output_helper(double *_outputs, double *_inputs, double* _innerDelta, double* _labels) : outputs(_outputs), inputs(_inputs), innerDelta(_innerDelta), labels(_labels){}
	double operator()(int t) {
		double local=outputs[t]-labels[t];
		double local2=1.0/(1.0+exp(-inputs[t]));
		local2=local2*(1.0-local2);
		return local2*local;
	}
};
struct divThreads : public unary_function<double, double> {
	double what;
	divThreads(double _what) : what(_what){}
	double operator()(double t) {
		return t/what;
	}
};
struct multi_helper : public unary_function<double, double> {
	double what;
	multi_helper(double _what) : what(_what){}
	double operator()(double t) {
		return t*what;
	}
};

class NN_layer {
public:

	vector<double> atNeuronOutputs;
	vector<double> atNeuronInputs;
	vector<double> weightsMatrix;
	vector<double> biases;
	vector<double> outerDeltaB;
	vector<double> outerDeltaW;
	vector<double> innerDeltaB;
	vector<double> innerDeltaW;

	NN_layer(){}
	NN_layer(int sizeThis, int sizeNext, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext) {

		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
	}
	NN_layer(int sizeThis, int sizeNext, int pBatchSize, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext), batchSize(pBatchSize) {

		setupLayer(true);
	}

	void setupLayer(bool newLayer) {
		atNeuronOutputs=vector<double>(batchSize*thisSize,0.0);
		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
		for(int i=0;i<allN;++i){
			counterN.push_back(i);
		}
		for(int i=0;i<allW;++i) {
			counterW.push_back(i);
		}
		memTracker(allN*8,false);
		memTracker(allW*sizeof(int),false);
		memTracker(allN*sizeof(int),false);
		if(newLayer) {
			if(type!=INPUT) {
				atNeuronInputs=vector<double>(batchSize*thisSize,0.0);
				memTracker(allN*8,false);
				biases=vector<double>(thisSize*batchSize,0.0);
				//transform(biases.begin(),biases.end(),biases.begin(),fix_random_numbers2());
				memTracker(allN*8*3,false);
				outerDeltaB=vector<double>(allN,0.0);
				innerDeltaB=vector<double>(allN,0.0);
			}
			if(type!=OUTPUT) {
				weightsMatrix=vector<double>(thisSize*nextSize,0.0);
				memTracker(allW*8*3,false);
				transform(weightsMatrix.begin(),weightsMatrix.end(),weightsMatrix.begin(),fix_random_numbers());
				outerDeltaW=vector<double>(allW,0.0);
				innerDeltaW=vector<double>(allW,0.0);
				//if(!my_rank) {
				//	cout << "thisSize: " << thisSize << " nextSize: " << nextSize << " thisSize*nextSize: " << (thisSize*nextSize) << endl;
				//}
			}
		} else {
			if(type!=INPUT) {
				atNeuronInputs=vector<double>(batchSize*thisSize,0.0);
				outerDeltaB=vector<double>(allN,0.0);
				innerDeltaB=vector<double>(allN,0.0);
			}
			if(type!=OUTPUT) {
				outerDeltaW=vector<double>(allW,0.0);
				innerDeltaW=vector<double>(allW,0.0);
			}
		}
	}

	int type, thisSize, nextSize, batchSize, allW, allN;
	vector<int> counterN;
	vector<int> counterW;
};

struct idLink {
    int whichThread;
    int interval;
    vector<double> *data;
    vector<double> *labels;
    vector<NN_layer> *NNlayersQ;
    vector<int> *hiddenMatrix;
    double learningRate;
    int batchSize;
};

void *fourthThread(void *thread_parm) {
	idLink data=*((idLink*) thread_parm);
	int myID=data.whichThread;
	int howMany=data.interval;
	vector<int> hiddenMatrix=*data.hiddenMatrix;
	int layers=hiddenMatrix.size();
	int outputsIndex=layers-1;
	int batchSize=data.batchSize;
	int numOutputs=hiddenMatrix[outputsIndex];
	int mOut, ii, mPlus, nextSize, prevSize, thisSize;
	vector<double> *which;
	bool gotTime=false;
	high_resolution_clock::time_point startTime, endTime;
	int timeCountDown=10;

	//double toDivideRMS=data.learningRate/(double)batchSize;
	while(!threadExit) {

		/*for(int i=0;i<outputsIndex;++i) {
			ii=i+1;
			fill((*data.NNlayersQ)[ii].outerDeltaB.begin(),(*data.NNlayersQ)[ii].outerDeltaB.end(),0.0);
			fill((*data.NNlayersQ)[i].outerDeltaW.begin(),(*data.NNlayersQ)[i].outerDeltaW.end(),0.0);
		}//*/

		//for(int h=0;h<howMany;++h) {
			//cout << "myID: " << myID << " h: " << h << " howMany: " << howMany << "\n";

			if(!myID && !gotTime && !timeCountDown && !my_rank) {
				startTime=high_resolution_clock::now();
			}

			//forward propagation
			which=data.data;
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[ii];
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &(*data.NNlayersQ)[i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &(*data.NNlayersQ)[ii].atNeuronInputs[0], nextSize);
				transform((*data.NNlayersQ)[ii].counterN.begin(),(*data.NNlayersQ)[ii].counterN.end(),(*data.NNlayersQ)[ii].atNeuronOutputs.begin(),forwardFeed_helper(&(*data.NNlayersQ)[ii].atNeuronInputs[0],&(*data.NNlayersQ)[ii].biases[0]));
				which=&(*data.NNlayersQ)[ii].atNeuronOutputs;
			}

			//Backward propagation
			mOut=outputsIndex-1;
			mPlus=outputsIndex;
			prevSize=hiddenMatrix[mOut];
			transform((*data.NNlayersQ)[outputsIndex].counterN.begin(),(*data.NNlayersQ)[outputsIndex].counterN.end(),(*data.NNlayersQ)[outputsIndex].innerDeltaB.begin(),output_helper(&(*data.NNlayersQ)[outputsIndex].atNeuronOutputs[0],&(*data.NNlayersQ)[outputsIndex].atNeuronInputs[0],&(*data.NNlayersQ)[outputsIndex].innerDeltaB[0],&(*data.labels)[0]));
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, numOutputs, prevSize, batchSize, 1.0, &(*data.NNlayersQ)[outputsIndex].innerDeltaB[0], numOutputs, &(*data.NNlayersQ)[mOut].atNeuronOutputs[0], prevSize, 0.0, &(*data.NNlayersQ)[mOut].innerDeltaW[0], numOutputs);

			--mOut;
			for(int i=outputsIndex-1;i;--i) {
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[i+1];
				prevSize=hiddenMatrix[i-1];
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, thisSize, batchSize, nextSize, 1.0, &(*data.NNlayersQ)[i].weightsMatrix[0], nextSize, &(*data.NNlayersQ)[i+1].innerDeltaB[0], nextSize, 0.0, &(*data.NNlayersQ)[i].innerDeltaB[0], thisSize);
				if(i!=1) {
					transform((*data.NNlayersQ)[i].counterN.begin(),(*data.NNlayersQ)[i].counterN.end(),(*data.NNlayersQ)[i].innerDeltaB.begin(),backProp_helper2(&(*data.NNlayersQ)[i].atNeuronOutputs[0],&(*data.NNlayersQ)[i].innerDeltaB[0]));
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &(*data.NNlayersQ)[i].innerDeltaB[0], thisSize, &(*data.NNlayersQ)[i-1].atNeuronOutputs[0], prevSize, 0.0, &(*data.NNlayersQ)[mOut].innerDeltaW[0], thisSize);
				} else {
					transform((*data.NNlayersQ)[i].counterN.begin(),(*data.NNlayersQ)[i].counterN.end(),(*data.NNlayersQ)[i].innerDeltaB.begin(),backProp_helper(&(*data.NNlayersQ)[i].innerDeltaB[0],&(*data.NNlayersQ)[i].atNeuronInputs[0]));
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &(*data.NNlayersQ)[i].innerDeltaB[0], thisSize, &(*data.data)[0], prevSize, 0.0, &(*data.NNlayersQ)[mOut].innerDeltaW[0], thisSize);
				}
				--mOut;
				--mPlus;
			}
			/*for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				transform((*data.NNlayersQ)[ii].innerDeltaB.begin(),(*data.NNlayersQ)[ii].innerDeltaB.end(),(*data.NNlayersQ)[ii].outerDeltaB.begin(),(*data.NNlayersQ)[ii].outerDeltaB.begin(),plus<double>());
				transform((*data.NNlayersQ)[i].innerDeltaW.begin(),(*data.NNlayersQ)[i].innerDeltaW.end(),(*data.NNlayersQ)[i].outerDeltaW.begin(),(*data.NNlayersQ)[i].outerDeltaW.begin(),plus<double>());
			}//*/
			/*for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				for_each((*data.NNlayersQ)[i].counterW.begin(),(*data.NNlayersQ)[i].counterW.end(),update_w(&(*data.NNlayersQ)[i].weightsMatrix[0],&(*data.NNlayersQ)[i].innerDeltaW[0],toDivideRMS));
				for_each((*data.NNlayersQ)[ii].counterN.begin(),(*data.NNlayersQ)[ii].counterN.end(),update_b(&(*data.NNlayersQ)[ii].biases[0],&(*data.NNlayersQ)[ii].innerDeltaB[0],toDivideRMS));
			}//*/
			/*for(int i=0;i<outputsIndex;++i) {
				for_each(make_counting_iterator(0),make_counting_iterator(&(*data.NNlayersQ)[i].allW),update_w(&(*data.NNlayersQ)[i].weightsMatrix[0],&(*data.outerDeltaW)[i][0],toDivideRMS));
				for_each(make_counting_iterator(0),make_counting_iterator(&(*data.NNlayersQ)[i+1].allN),update_b(&(*data.NNlayersQ)[i+1].biases[0],&(*data.outerDeltaB)[i][0],toDivideRMS));
			}//*/
			if(!myID && !my_rank) {
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
	free(thread_parm);
	pthread_exit(0);
}

class neuralNet {
public:
	neuralNet(){}

	int numThreads;

	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize, int _numThreads) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize), numThreads(_numThreads) {

		openblas_set_num_threads(1);
		numThreads=numberOfProcessors;
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);
		batchSize=60000/totalNum;
		for(int i=0;i<numThreads;++i) {
			NNlayersQ.push_back(vector<NN_layer>());
		}

		for(int i=0;i<numThreads;++i) {
			NNlayersQ[i]=vector<NN_layer>(hiddenMatrix.size());
		}
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;
		if(!my_rank) {
			cout << "Setting up multi-threaded network...\n";
			cout << "Layers: ";
			for(auto h:hiddenMatrix) {
				cout << h << " ";
			}
			cout << "Batch size: " << batchSize << endl << endl;
		}

		for(int j=0;j<numThreads;++j) {
			NNlayersQ[j][0]=NN_layer(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT);
			for(int i=1;i<outputsIndex;++i) {
				NNlayersQ[j][i]=NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN);
			}
			NNlayersQ[j][outputsIndex]=NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT);
		}
		for(int i=1;i<numThreads;++i) {
			for(int j=0;j<outputsIndex;++j) {
				copy(NNlayersQ[i-1][j].weightsMatrix.begin(),NNlayersQ[i-1][j].weightsMatrix.end(),NNlayersQ[i][j].weightsMatrix.begin());
				copy(NNlayersQ[i-1][j+1].biases.begin(),NNlayersQ[i-1][j+1].biases.end(),NNlayersQ[i][j+1].biases.begin());
			}
		}
		for(int i=0;i<outputsIndex;++i) {
			MPI_Bcast(&NNlayersQ[0][i].weightsMatrix[0],NNlayersQ[0][i].allW,MPI_DOUBLE,MASTER,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&NNlayersQ[0][i+1].biases[0],NNlayersQ[0][i+1].allN,MPI_DOUBLE,MASTER,MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

void train_Quad(vector<vector<double>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, double lRate, vector<vector<double>> &pTestData, vector<vector<double>> &pTestLabels, bool vlRate) {

		if(!showInterval) {
			showInterval=10;
		}
		vector<int> bLabels;
		for(auto p:pLabels) {
			bLabels.push_back(max_element(p.begin(), p.end())-p.begin());
		}

		if(lRate<0.0) {
			learningRate=0.05;
		} else {
			learningRate=lRate;
		}
		dataSetSize=60000;
		doDataSetSize=dataSetSize;
		int testBatchSize;
		if(batchSize>10000) {
			testBatchSize=10000;
		} else {
			testBatchSize=100;
		}

		int batchStart,batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		int testSetSize=pTestData.size();
		vector<int> btLabels;
		vector<double> testData[testSetSize/testBatchSize];
	
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back(max_element(p.begin(), p.end())-p.begin());
			}
		}

		vector<double> data[totalNum];
		vector<double> labels[totalNum];

		//Creating pre-made batches so I can simply copy them to layer[0]
		if(!my_rank) {
			cout << "Making batches in memory...\n";
		}
		int whichBatch=0;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			data[whichBatch]=vector<double>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			labels[whichBatch]=vector<double>(batchSize*numOutputs);
			memTracker(numOutputs*batchSize*8,false);
			for(int b=0;b<batchSize;++b) {
				copy(pData[itemNum+b].begin(),pData[itemNum+b].end(),data[whichBatch].begin()+batchStart);
				copy(pLabels[itemNum+b].begin(),pLabels[itemNum+b].end(),labels[whichBatch].begin()+batchEnd);
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
				copy(pTestData[itemNum+j].begin(),pTestData[itemNum+j].end(),testData[whichBatch].begin()+batchStart);
				batchStart+=itemSize;
			}
			++whichBatch;
		}

		int mOut=outputsIndex-2;

		if(!my_rank) {
			cout << "Starting training...\n";
		}

		vector<double>::iterator iter;
		int position;
		int gotRight=0, prevSize;
		//int numBatches=dataSetSize/batchSize;
		//toDivideRMS=learningRate/((double)numBatches*(double)batchSize);
		//toDivideRMS=learningRate/((double)batchSize*(double)showInterval);
		//toDivideRMS=learningRate/((double)batchSize*(double)num_nodes);//*(double)showInterval);
		toDivideRMS=learningRate/(double)batchSize;
		//toDivideRMS=learningRate/(double)showInterval;
		int maxGotRight=0, maxTestRight=-1, ii;
		vector<double> *which;
		double seconds, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		int mPlus, sInterval=showInterval;
		bool once=true;

		vector<pthread_t> threads;
		pthread_attr_t attr;
    	cpu_set_t cpus;
    	pthread_attr_init(&attr);

		//divThreads dThreads((double)num_nodes);
		divThreads dThreads((double)totalNum);
		multi_helper hTimes((double)numberOfProcessors);
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
			whichBatch=0;
			gotRight=0;
			startTime=high_resolution_clock::now();
			if(once) {
			    for(int j=0;j<numThreads;++j) {

     				CPU_ZERO(&cpus);
       				CPU_SET(j, &cpus);
       				pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

			        threads.push_back(pthread_t());
			        idLink *arg = (idLink*)malloc(sizeof(*arg));
			        (*arg).whichThread=j;
			        (*arg).data=&data[j+startBatch];
			        (*arg).labels=&labels[j+startBatch];
			        (*arg).hiddenMatrix=&hiddenMatrix;
			        (*arg).interval=sInterval;
			        (*arg).NNlayersQ=&NNlayersQ[j];
			        (*arg).learningRate=learningRate;
			        (*arg).batchSize=batchSize;
			        pthread_create(&threads.at(j), &attr,  fourthThread, arg);
			    }
			    once=false;
			}
			pthread_barrier_wait(&barrier);
			for(int i=1;i<numThreads;++i) {
				for(int j=0;j<outputsIndex;++j) {
					ii=j+1;
					//transform(NNlayersQ[0][j].outerDeltaW.begin(),NNlayersQ[0][j].outerDeltaW.end(),NNlayersQ[i][j].outerDeltaW.begin(),NNlayersQ[0][j].outerDeltaW.begin(),plus<double>());
					//transform(NNlayersQ[0][ii].outerDeltaB.begin(),NNlayersQ[0][ii].outerDeltaB.end(),NNlayersQ[i][ii].outerDeltaB.begin(),NNlayersQ[0][ii].outerDeltaB.begin(),plus<double>());
					transform(NNlayersQ[0][j].innerDeltaW.begin(),NNlayersQ[0][j].innerDeltaW.end(),NNlayersQ[i][j].innerDeltaW.begin(),NNlayersQ[0][j].innerDeltaW.begin(),plus<double>());
					transform(NNlayersQ[0][ii].innerDeltaB.begin(),NNlayersQ[0][ii].innerDeltaB.end(),NNlayersQ[i][ii].innerDeltaB.begin(),NNlayersQ[0][ii].innerDeltaB.begin(),plus<double>());
				}
			}
			/*for(int j=0;j<outputsIndex;++j) {
				ii=j+1;
				for_each(NNlayersQ[0][j].counterW.begin(),NNlayersQ[0][j].counterW.end(),update_w(&NNlayersQ[0][j].weightsMatrix[0],&NNlayersQ[0][j].outerDeltaW[0],toDivideRMS));
				for_each(NNlayersQ[0][ii].counterN.begin(),NNlayersQ[0][ii].counterN.end(),update_b(&NNlayersQ[0][ii].biases[0],&NNlayersQ[0][ii].outerDeltaB[0],toDivideRMS));
			}//*/
			for(int i=0;i<numThreads;++i) {
				for(int j=0;j<outputsIndex;++j) {
					ii=j+1;
					for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(&NNlayersQ[i][j].weightsMatrix[0],&NNlayersQ[0][j].innerDeltaW[0],toDivideRMS));
					for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(&NNlayersQ[i][ii].biases[0],&NNlayersQ[0][ii].innerDeltaB[0],toDivideRMS));
					//for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(&NNlayersQ[i][j].weightsMatrix[0],&tempDeltaW[j][0],toDivideRMS));
					//for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(&NNlayersQ[i][ii].biases[0],&tempDeltaB[j][0],toDivideRMS));
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
				MPI_Barrier(MPI_COMM_WORLD);
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					//MPI_Allreduce(&NNlayersQ[0][i].outerDeltaW[0],&NNlayersQ[1][i].outerDeltaW[0],NNlayersQ[0][i].allW,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][ii].outerDeltaB[0],&NNlayersQ[1][ii].outerDeltaB[0],NNlayersQ[0][ii].allN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][i].innerDeltaW[0],&tempDeltaW[i][0],NNlayersQ[0][i].allW,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][i+1].innerDeltaB[0],&tempDeltaB[i][0],NNlayersQ[0][i+1].allN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					transform(NNlayersQ[0][i].weightsMatrix.begin(),NNlayersQ[0][i].weightsMatrix.end(),NNlayersQ[2][i].weightsMatrix.begin(),hTimes);
					transform(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[2][ii].biases.begin(),hTimes);
					MPI_Allreduce(&NNlayersQ[2][i].weightsMatrix[0],&NNlayersQ[1][i].weightsMatrix[0],NNlayersQ[0][i].allW,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					MPI_Allreduce(&NNlayersQ[2][ii].biases[0],&NNlayersQ[1][ii].biases[0],NNlayersQ[0][ii].allN,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
					transform(NNlayersQ[1][i].weightsMatrix.begin(),NNlayersQ[1][i].weightsMatrix.end(),NNlayersQ[0][i].weightsMatrix.begin(),dThreads);
					transform(NNlayersQ[1][ii].biases.begin(),NNlayersQ[1][ii].biases.end(),NNlayersQ[0][ii].biases.begin(),dThreads);
				}
				for(int i=0;i<numThreads;++i) {
					for(int j=0;j<outputsIndex;++j) {
						ii=j+1;
						//for_each(NNlayersQ[i][j].counterW.begin(),NNlayersQ[i][j].counterW.end(),update_w(&NNlayersQ[i][j].weightsMatrix[0],&NNlayersQ[1][j].outerDeltaW[0],toDivideRMS));
						//for_each(NNlayersQ[i][ii].counterN.begin(),NNlayersQ[i][ii].counterN.end(),update_b(&NNlayersQ[i][ii].biases[0],&NNlayersQ[1][ii].outerDeltaB[0],toDivideRMS));
						copy(NNlayersQ[0][j].weightsMatrix.begin(),NNlayersQ[0][j].weightsMatrix.end(),NNlayersQ[i][j].weightsMatrix.begin());
						copy(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[i][ii].biases.begin());
					}
				}
			}
			if(!my_rank) {
				for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {

					//forward propagation
					which=&data[whichBatch];
					for(int i=0;i<outputsIndex;++i) {
						ii=i+1;
						thisSize=hiddenMatrix[i];
						nextSize=hiddenMatrix[ii];
						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayersQ[0][i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &NNlayersQ[0][ii].atNeuronInputs[0], nextSize);
						transform(NNlayersQ[0][ii].counterN.begin(),NNlayersQ[0][ii].counterN.end(),NNlayersQ[0][ii].atNeuronOutputs.begin(),forwardFeed_helper(&NNlayersQ[0][ii].atNeuronInputs[0],&NNlayersQ[0][ii].biases[0]));
						which=&NNlayersQ[0][ii].atNeuronOutputs;
					}

					batchStart=0;
					batchEnd=numOutputs;
					//printf("\nbatch starting at: %d\n",itemNum);
					for(int b=0;b<batchSize;++b) {
						iter = max_element(NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchEnd);
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
				printf("Training epoch: %d -- Got %d of %d -- max right: %d -- lRate: %.5f -- ",epochNum,gotRight,dataSetSize,maxGotRight,learningRate);
				gotRight=0;		
				whichBatch=0;
				for(int t=0;t<testSetSize;t+=testBatchSize) {
					which=&testData[whichBatch];
					for(int i=0;i<outputsIndex;++i) {
						ii=i+1;
						thisSize=hiddenMatrix[i];
						nextSize=hiddenMatrix[ii];
						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, testBatchSize, thisSize, 1.0, &NNlayersQ[0][i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &NNlayersQ[0][ii].atNeuronInputs[0], nextSize);
						transform(NNlayersQ[0][ii].counterN.begin(),NNlayersQ[0][ii].counterN.end(),NNlayersQ[0][ii].atNeuronOutputs.begin(),forwardFeed_helper(&NNlayersQ[0][ii].atNeuronInputs[0],&NNlayersQ[0][ii].biases[0]));
						which=&NNlayersQ[0][ii].atNeuronOutputs;
					}

					batchStart=0;
					batchEnd=numOutputs;
					//printf("\nbatch starting at: %d\n",t);
					for(int b=0;b<testBatchSize;++b) {
						iter = max_element(NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayersQ[0][outputsIndex].atNeuronOutputs.begin()+batchEnd);
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
				printf("Testing -- Got %d of %d -- max right: %d -- totalTime: %.5f -- errRate:%.5f percent\n",gotRight,testSetSize,maxTestRight,totalTime,errRate);
				if(testSetSize!=gotRight) {
					for(int i=1;i<numThreads;++i) {
						for(int j=0;j<outputsIndex;++j) {
							ii=j+1;
							copy(NNlayersQ[0][j].weightsMatrix.begin(),NNlayersQ[0][j].weightsMatrix.end(),NNlayersQ[i][j].weightsMatrix.begin());
							copy(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[i][ii].biases.begin());
						}
					}
				} else {
					threadExit=true;
				}
				MPI_Barrier(MPI_COMM_WORLD);
				pthread_barrier_wait(&barrier2);
			} else {
				MPI_Barrier(MPI_COMM_WORLD);
				pthread_barrier_wait(&barrier2);
			}
		}
    	int status;
		void * result;
    	for (int i=0; i < numThreads; ++i) {
        	if ((status = pthread_join(threads.at(i), &result)) != 0) {
            	fprintf (stderr, "join error %d: %s\n", status, strerror(status));
	        }
    	}
		saveStateQ("MPIv2-");
	}

	void saveStateQ(string outFile) {
		outFile+="noCuda-"+to_string(dataSetSize);
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

	neuralNet(string _inFile) : inFile(_inFile) {
		cout << "Setting up network...\n";
		loadState();
		cout << "Layers: ";
		for(auto h:hiddenMatrix) {
			cout << h << " ";
		}
		cout << "Batch size: " << batchSize << endl << endl;
	}
	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize) : 
		hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize) {

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

	void train_MatMul(vector<vector<double>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, double lRate, vector<vector<double>> &pTestData, vector<vector<double>> &pTestLabels, bool vlRate) {

		vector<int> bLabels;
		for(auto p:pLabels) {
			bLabels.push_back(max_element(p.begin(), p.end())-p.begin());
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

		int batchStart,batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		//dataSetSize=pData.size();
		int testSetSize=pTestData.size();
		vector<int> btLabels;
		vector<double> testData[testSetSize/batchSize];
		//vector<double> testLabels[testSetSize];		
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back(max_element(p.begin(), p.end())-p.begin());
			}
		}

		//toDivideRMS=((double)dataSetSize)*(double)numOutputs;
		vector<double> data[dataSetSize/batchSize];
		//vector<float> dataTemp;//[dataSetSize];
		//vector<double> dataTransposeTemp(itemSize*batchSize);
		vector<double> labels[dataSetSize/batchSize];
		//vector<double> labelsTemp;//[dataSetSize];
		//vector<double> batchLabels(numOutputs*batchSize,0.0);
		//vector<double> outputsTemp(batchSize*numOutputs,0.0);

		//Creating pre-made batches so I can simply copy them to layer[0]
		cout << "Making batches in memory...\n";
		int whichBatch=0;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			data[whichBatch]=vector<double>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			labels[whichBatch]=vector<double>(batchSize*numOutputs);
			memTracker(numOutputs*batchSize*8,false);
			for(int b=0;b<batchSize;++b) {
				copy(pData[itemNum+b].begin(),pData[itemNum+b].end(),data[whichBatch].begin()+batchStart);
				copy(pLabels[itemNum+b].begin(),pLabels[itemNum+b].end(),labels[whichBatch].begin()+batchEnd);
				batchStart+=itemSize;
				batchEnd+=numOutputs;
			}
			++whichBatch;
		}
		whichBatch=0;
		for(int i=0;i<testSetSize;i+=batchSize) {
			testData[whichBatch]=vector<double>(itemSize*batchSize);
			memTracker(itemSize*batchSize*8,false);
			batchStart=0;
			for(int j=0;j<batchSize;++j) {
				copy(pTestData[i+j].begin(),pTestData[i+j].end(),testData[whichBatch].begin()+batchStart);
				batchStart+=itemSize;
			}
			++whichBatch;
		}

		int mOut=outputsIndex-2;

		cout << "Starting training...\n";
		//memTracker(0,true);

		vector<double>::iterator iter;
		int position;
		int gotRight=0, prevSize;
		int numBatches=dataSetSize/batchSize;
		toDivideRMS=learningRate/((double)numBatches*(double)batchSize);
		//toDivideRMS=learningRate/(double)batchSize;
		//cout << "toDivideRMS: " << toDivideRMS << endl;
		int maxGotRight=0, maxTestRight=-1, ii;
		vector<double> *which;
		double origLearningRate=learningRate, seconds, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		double lastNoShowTime=0.0;
		int timeEstCountDown=10, mPlus;

		for(int epochNum=0;epochNum<maxEpochs && maxGotRight!=dataSetSize && maxTestRight!=testSetSize;++epochNum) {
			whichBatch=0;
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				fill(NNlayers[ii].outerDeltaB.begin(),NNlayers[ii].outerDeltaB.end(),0.0);
				fill(NNlayers[i].outerDeltaW.begin(),NNlayers[i].outerDeltaW.end(),0.0);
			}//*/
			gotRight=0;
			startTime=high_resolution_clock::now();
			for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {

				//forward propagation
				which=&data[whichBatch];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayers[i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &NNlayers[ii].atNeuronInputs[0], nextSize);
					transform(NNlayers[ii].counterN.begin(),NNlayers[ii].counterN.end(),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(&NNlayers[ii].atNeuronInputs[0],&NNlayers[ii].biases[0]));
					which=&NNlayers[ii].atNeuronOutputs;
				}

				//first check how many we got right
				if(!showIntervalCountDown) {
					batchStart=0;
					batchEnd=numOutputs;
					//printf("\nbatch starting at: %d\n",itemNum);
					for(int b=0;b<batchSize;++b) {
						iter = max_element(NNlayers[outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayers[outputsIndex].atNeuronOutputs.begin()+batchEnd);
						position = iter - NNlayers[outputsIndex].atNeuronOutputs.begin();
						position -= batchStart;
						/*printf("output: %d expected: %d\n",position,bLabels[itemNum+b]);
						for(int ot=batchStart;ot<batchEnd;++ot) {
							double oo=NNlayers[outputsIndex].atNeuronOutputs[ot];
							printf("%.5f ",oo);
						}
						printf("\n");//*/
						if(position==bLabels[itemNum+b]) {
							++gotRight;
						}
						batchStart=batchEnd;
						batchEnd+=numOutputs;					
					}
					//sleep(5);
				}

				//Backward propagation
				mOut=outputsIndex-1;
				mPlus=outputsIndex;
				prevSize=hiddenMatrix[mOut];
				transform(NNlayers[outputsIndex].counterN.begin(),NNlayers[outputsIndex].counterN.end(),NNlayers[outputsIndex].innerDeltaB.begin(),output_helper(&NNlayers[outputsIndex].atNeuronOutputs[0],&NNlayers[outputsIndex].atNeuronInputs[0],&NNlayers[outputsIndex].innerDeltaB[0],&labels[whichBatch][0]));
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, numOutputs, prevSize, batchSize, 1.0, &NNlayers[outputsIndex].innerDeltaB[0], numOutputs, &NNlayers[mOut].atNeuronOutputs[0], prevSize, 0.0, &NNlayers[mOut].innerDeltaW[0], numOutputs);

				--mOut;
				for(int i=outputsIndex-1;i;--i) {
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[i+1];
					prevSize=hiddenMatrix[i-1];
					cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, thisSize, batchSize, nextSize, 1.0, &NNlayers[i].weightsMatrix[0], nextSize, &NNlayers[i+1].innerDeltaB[0], nextSize, 0.0, &NNlayers[i].innerDeltaB[0], thisSize);
					if(i!=1) {
						transform(NNlayers[i].counterN.begin(),NNlayers[i].counterN.end(),NNlayers[i].innerDeltaB.begin(),backProp_helper2(&NNlayers[i].atNeuronOutputs[0],&NNlayers[i].innerDeltaB[0]));
						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &NNlayers[i].innerDeltaB[0], thisSize, &NNlayers[i-1].atNeuronOutputs[0], prevSize, 0.0, &NNlayers[mOut].innerDeltaW[0], thisSize);
					} else {
						transform(NNlayers[i].counterN.begin(),NNlayers[i].counterN.end(),NNlayers[i].innerDeltaB.begin(),backProp_helper(&NNlayers[i].innerDeltaB[0],&NNlayers[i].atNeuronInputs[0]));
						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &NNlayers[i].innerDeltaB[0], thisSize, &data[whichBatch][0], prevSize, 0.0, &NNlayers[mOut].innerDeltaW[0], thisSize);
					}
					--mOut;
					--mPlus;
				}
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					transform(NNlayers[ii].innerDeltaB.begin(),NNlayers[ii].innerDeltaB.end(),NNlayers[ii].outerDeltaB.begin(),NNlayers[ii].outerDeltaB.begin(),plus<double>());
					transform(NNlayers[i].innerDeltaW.begin(),NNlayers[i].innerDeltaW.end(),NNlayers[i].outerDeltaW.begin(),NNlayers[i].outerDeltaW.begin(),plus<double>());
				}//*/
				/*for(int i=0;i<outputsIndex;++i) {
					for_each(NNlayers[i].counterW.begin(),NNlayers[i].counterW.end(),update_w(&NNlayers[i].weightsMatrix[0],&innerDeltaW[i][0],toDivideRMS));
					for_each(NNlayers[i+1].counterN.begin(),NNlayers[i+1].counterN.end(),update_b(&NNlayers[i+1].biases[0],&innerDeltaB[i][0],toDivideRMS));
				}//*/
				++whichBatch;
			}
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				for_each(NNlayers[i].counterW.begin(),NNlayers[i].counterW.end(),update_w(&NNlayers[i].weightsMatrix[0],&NNlayers[i].outerDeltaW[0],toDivideRMS));
				for_each(NNlayers[ii].counterN.begin(),NNlayers[ii].counterN.end(),update_b(&NNlayers[ii].biases[0],&NNlayers[ii].outerDeltaB[0],toDivideRMS));
			}//*/
			if(!showIntervalCountDown) {
				if(gotRight>maxGotRight){maxGotRight=gotRight;}
				printf("Training epoch: %d -- Got %d of %d -- max right: %d -- lRate: %.5f -- ",epochNum,gotRight,dataSetSize,maxGotRight,learningRate);
				gotRight=0;
				//sleep(5);
			
				whichBatch=0;
				for(int t=0;t<testSetSize;t+=batchSize) {
					which=&testData[whichBatch];
					for(int i=0;i<outputsIndex;++i) {
						ii=i+1;
						thisSize=hiddenMatrix[i];
						nextSize=hiddenMatrix[ii];
						cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayers[i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &NNlayers[ii].atNeuronInputs[0], nextSize);
						transform(NNlayers[ii].counterN.begin(),NNlayers[ii].counterN.end(),NNlayers[ii].atNeuronOutputs.begin(),forwardFeed_helper(&NNlayers[ii].atNeuronInputs[0],&NNlayers[ii].biases[0]));
						which=&NNlayers[ii].atNeuronOutputs;
					}

					batchStart=0;
					batchEnd=numOutputs;
					//printf("\nbatch starting at: %d\n",t);
					for(int b=0;b<batchSize;++b) {
						iter = max_element(NNlayers[outputsIndex].atNeuronOutputs.begin()+batchStart, NNlayers[outputsIndex].atNeuronOutputs.begin()+batchEnd);
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
				if(gotRight>maxTestRight){maxTestRight=gotRight;}
			}
			if(vlRate) {
				double cutOff=0.92;
				double percLearned=(double)gotRight/(double)testSetSize;
				if(percLearned<0.99 && percLearned>cutOff) {
					percLearned=1.0-percLearned;
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
			}
			endTime=high_resolution_clock::now();
			seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
			totalTime+=seconds;
			if(!showIntervalCountDown) {
				printf("Testing -- Got %d of %d -- max right: %d -- sec: %.5f -- totalTime: %.5f\n",gotRight,testSetSize,maxTestRight,lastNoShowTime,totalTime);
				showIntervalCountDown=showInterval;
			} else {
				lastNoShowTime=seconds;
				--showIntervalCountDown;
				if(timeEstCountDown) {
					--timeEstCountDown;
					if(!timeEstCountDown) {
						printf("Update time interval approximately %.5f seconds apart\n",(lastNoShowTime*(double)showInterval)+1.0);
					}
				}
			}
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
	vector<vector<NN_layer>> NNlayersQ;

private:
	ULLI epoch, maxElement, layers, maxEpochs;//, maxWeightsMatrix, maxDeltaMatrix;
	int outputsIndex, dataSetSize, numInputs, numOutputs, batchSize;
	double RMS, minRMS, toDivideRMS, RMSwanted, learningRate;
	vector<int> hiddenMatrix;
	//---cublasHandle_t handle;
	ULLI itemSize;
	string inFile;
	ULLI neededEpochs;

	vector<vector<double>> neuralNet_weights_host;
	//vector<double> weightsTemp;
	//vector<double> deltaTemp;
};

void doMain(vector<int> &inputHiddenLayers, int batchSize, int doDataSetSize, double lRate, string inFile, string outFile, bool vlRate) {

	
	vector<int> hiddenMatrix;
	if(!inputHiddenLayers.size()) {
		hiddenMatrix.push_back(200);
		hiddenMatrix.push_back(100);
	} else {
		for(auto h:inputHiddenLayers) {
			hiddenMatrix.push_back(h);
		}
	}

	//vector<vector<float>> testData(10000);
	vector<vector<double>> testData(10000);
	ReadMNIST_float("t10k-images.idx3-ubyte",10000,784,testData);
	//vector<vector<float>> trainData(60000);
	vector<vector<double>> trainData(60000);
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

	neuralNet go;
	if(inFile=="") {
		//go=neuralNet(784,10,hiddenMatrix,batchSize);
		go=neuralNet(784,10,hiddenMatrix,batchSize,3);
	} else {
		go=neuralNet(inFile);
	}
	auto start = high_resolution_clock::now();
	//go.train_floats(trainData,trainLabels,1000000,0.0001,trainLabels2);
	//go.train(trainData,trainLabels,1000000,0.0001,trainLabels2, doDataSetSize);//*/
	//go.train_MatMul(trainData,trainLabels, 1000000, 0.0001, doDataSetSize, lRate, testData, testLabels, vlRate);//*/
	go.train_Quad(trainData,trainLabels, 1000000, 0.0001, doDataSetSize, lRate, testData, testLabels, vlRate);//*/
	//go.evaluate(testData,testLabels,testLabels2, doDataSetSize);
	auto endTime = high_resolution_clock::now();
	printTime(start,endTime);

}

int main(int argc, char *argv[]) {

	/*for(int i=0;i<(21*4);++i) {
		double core=300.0+(double)i;
		cout << (60000.0/core) << " " << core << endl;
	}
	return 0;*/

	struct sigaction ctrlc;
	ctrlc.sa_handler=ctrlchandler;
	ctrlc.sa_flags=0;
	sigemptyset(&ctrlc.sa_mask);
	sigaction(SIGTSTP,&ctrlc,NULL);

	numberOfProcessors = sysconf(_SC_NPROCESSORS_ONLN);
    //printf("Number of processors: %d\n", numberOfProcessors);

	string inFile="";
	string outFile="";
	int doDataSetSize=0;
	int batchSize=5;
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

	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    char my_host[100];
    gethostname(my_host, 100);
 	hostname=string(my_host);
    //printf("%s\n",hostname.c_str());
    if(hostname=="quattro.cis.gvsu.edu") {
    	numberOfProcessors=3;
    }
    /*vector<double> x(10);
    vector<double> y(10);
    for(int i=0;i<10;++i) {
    	x[i]=(double)my_rank+(double)i;
    }
    MPI_Allreduce(&x[0],&y[0],10,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    for(int i=0;i<10;++i) {
    	cout << hostname << ": " << i << ": " << y[i] << " x: " << x[i] << endl;
    }
    MPI_Finalize();
    return 0;*/

    int placeHolder=13;
    if (my_rank != MASTER) {
    	MPI_Sendrecv(&numberOfProcessors, 1, MPI_INT, MASTER, TAG, &placeHolder, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	//used for recieving the total cpus on the job and which parts this computer is doing
    	int getThree[3];
    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, getThree, 3, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	printf("%s got total: %d and which %d thru %d\n",hostname.c_str(),getThree[0],getThree[1],getThree[2]);
    	totalNum=getThree[0];
    	startBatch=getThree[1];
    	endBatch=getThree[2];
    } else {
    	totalNum=numberOfProcessors;
    	int tempNum=0;
    	vector<int> sourceCores;
    	sourceCores.push_back(totalNum);
    	for(int source = 1; source < num_nodes; source++) {
    		MPI_Sendrecv(&placeHolder, 1, MPI_INT, source, TAG, &tempNum, 1, MPI_INT, source, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    		totalNum+=tempNum;
    		sourceCores.push_back(tempNum);
    	}
    	int who=0, index=0;
    	int sendThree[3];
    	sendThree[0]=totalNum;
    	for(auto s:sourceCores) {
    		if(who) {
    			sendThree[1]=index;
    			sendThree[2]=index+s;
    			MPI_Sendrecv(sendThree, 3, MPI_INT, who, TAG, &placeHolder, 1, MPI_INT, who, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    		}
    		index+=s;
    		++who;
    	}
    }

    pthread_barrier_init(&barrier, NULL, numberOfProcessors+1);
	pthread_barrier_init(&barrier2, NULL, numberOfProcessors+1);

	doMain(inputHiddenLayers, batchSize, doDataSetSize, lRate, inFile, outFile, vlRate);

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

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr) {
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

void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end) {
	double seconds=duration_cast<microseconds>(end-start).count()/1000000.0;
	cout << "Processing time (milliseconds): " << duration_cast<milliseconds>(end - start).count() << endl;
	cout << "Processing time (microseconds): " << duration_cast<microseconds>(end - start).count() << endl;
	cout << "Processing time (nanoseconds): " << duration_cast<nanoseconds>(end - start).count() << endl;
	printf("Processing time (seconds): %.04f\n",seconds);
}

void print_matrix(vector<double> &A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i){
        for(int j = 0; j < nr_cols_A; ++j){
            //cout << A[j * nr_rows_A + i] << " ";
            double o=A[j*nr_rows_A+i];
            printf("%.4f ",o);
            //printf("%.10f ",A[j*nr_rows_A+i]);
        }
        cout << endl;
    }
    //co