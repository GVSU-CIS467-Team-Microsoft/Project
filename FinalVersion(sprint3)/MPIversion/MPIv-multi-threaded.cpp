//Artifical Neural Network with Cuda and Cublas matrix version

//Ron Patrick - Capstone GVSU - Winter 2017 - Multi-threaded

#include <iostream>
#include <fstream>
#include <string>
#include <experimental/optional>
#include <experimental/algorithm>
//#include <experimental/numeric>
//#include <experimental/execution_policy>
#include <parallel/algorithm>
#include <parallel/settings.h>
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
#include <math.h>
#include "imebra/imebra.h"
#include <pthread.h>
#include <iomanip>
//#include <openblas/cblas.h>
//#include "cblas.h"
#include <mkl.h>
#include <signal.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>
#define GetCurrentDir getcwd
using namespace std;
//using namespace __parallel;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define ULLI unsigned long long int
#define UNCHAR unsigned char
#define UINT unsigned int
#define INPUT 0
#define OUTPUT 1
#define HIDDEN 2
#define MASTER  0
#define TAG     0

#define GETMINMAX 3
#define DOPREPROC 2
#define REMOVEZEROS 1
#define DONEPREPROC 0

#ifndef ZERO_THRESHOLD
#define ZERO_THRESHOLD 10
#endif
#ifndef OUTLIER_MIN_COUNT
#define OUTLIER_MIN_COUNT 30
#endif

ULLI memoryTracker=0;
int showInterval=0;
pthread_mutex_t crmutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t crmutex2 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t preprocMaxMinMutex = PTHREAD_MUTEX_INITIALIZER;
bool threadExit=false;
static pthread_barrier_t barrier;
static pthread_barrier_t barrier2;
int numberOfProcessors;
int my_rank, num_nodes, totalNum;
string hostname="";
char CurrentPath[1000];// FILENAME_MAX];
string slash = "/";
vector<UNCHAR> success;
vector<float> oneFile;
vector<UNCHAR> inVector;
unordered_map<int,UNCHAR> toRemove;
vector<string> patients;
ULLI fsize=0;
vector<int> toDoList;
vector<vector<int>> threadedToDoList;
vector<int> toTrainList;
vector<int> toTestList;
vector<pair<string,float>> setLabeledPatients;
int toDo=0;
bool doingMNIST=false;
vector<string> patientFolders;
float maxFloat, minFloat=FLT_MAX, avgSpacing=0.0f;
int maxDim=INT_MAX;
bool reshuffle=false;

vector<vector<vector<float>>> threadInnerDeltaW;
vector<vector<vector<float>>> threadInnerDeltaB;
vector<vector<vector<float>>> threadOuterDeltaW;
vector<vector<vector<float>>> threadOuterDeltaB;
vector<vector<vector<float>>> threadInputs;
vector<vector<vector<float>>> threadOutputs;
vector<int> hiddenMatrix;

vector<pair<string,int>> trainingSet;
vector<pair<float,int>> trainingLabels;
vector<pair<string,int>> testSet;
vector<pair<float, int>> testLabels;
int zerosInTrainSet=0;
int zerosInTestSet=0;

void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr);
void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end);
void print_matrix(vector<float> &A, int nr_rows_A, int nr_cols_A);
void GetFilesInDirectory(vector<string> &out, vector<string> &types);
bool ChangeDir(string dir);
vector<string> split(string s, string any_of);
void getFile2(string filename, float *array);
void getFile(string filename);
int getFile(string filename, UNCHAR *input);
void *preprocThread(void *thread_parm);
void *preprocThread2(void *thread_parm);
void *preprocMaxMinThread(void *thread_parm);
struct idLink2 {
    int whichPatient;
    int my_id;
};
vector<bool> threadsDone;

void ctrlchandler(int sig) {
	printf("\nTrying to exit...\n");
	threadExit=true;
}

void memTracker(ULLI in, bool printIt) {
	memoryTracker+=in;
	if(printIt) {
	//if(!my_rank) {
		cout << hostname << ": memory tracker: Using(bytes): " << memoryTracker << " ";
		cout << "(Kb): " << (memoryTracker/1024) << " ";
		cout << "(Mb): " << ((memoryTracker/1024)/1024) << endl;
	}
}

struct fix_random_numbers : public unary_function<float, float> {
	float operator()(float t) {
		float tt=(float)random()/(float)RAND_MAX;
		return (tt*2.0)-1.0;
	}
};

struct update_w : public unary_function<int, void> {
	float *weights;
	float *newW;
	float lRate;
	update_w(float *w, float *_newW, float lr) : weights(w), newW(_newW), lRate(lr){}
	void operator()(int t) {
		float local=weights[t];
		float local2=lRate;
		float local3=newW[t];
		float local4=local-local2*local3;
		weights[t]=local4;
	}
};
struct update_b : public unary_function<int, void> {
	float *biases;
	float *newB;
	float lRate;
	update_b(float *b, float *_newB, float lr) : biases(b), newB(_newB), lRate(lr){}
	void operator()(int t) {
		float local=biases[t];
		float local2=lRate;
		float local3=newB[t];
		float local4=local-local2*local3;
		biases[t]=local4;
	}
};

template<typename T>
struct square {
	T operator()(const T& x) const { 
		return x * x;
	}
};
struct sigmoid_devrivative : public unary_function<float, float> {
	float operator()(float t) {
		float tt=1.0f/(1.0f+exp(-t));
		return tt*(1.0f-tt);
	}
};
struct sigmoid : public unary_function<float, float> {
	sigmoid(){}
	float operator()(float t) {
		return 1.0f / (1.0f + exp(-t));
	}
};
struct exp_float : public unary_function<float, float> {
	float operator()(float t) {
		return exp(t);
	}
};

struct forwardFeed_helper : public unary_function<int, float> {
	float *inputs;
	float *biases;
	forwardFeed_helper(){}
	forwardFeed_helper(float *_inputs, float* _biases) : inputs(_inputs), biases(_biases){}
	float operator()(int t) {
		float local=inputs[t];
		local+=biases[t];
		inputs[t]=local;
		//return tanh(local);
		return 1.0/(1.0+exp(-local));
	}
};
struct backProp_helper : public unary_function<int, float> {
	float *innerDelta;
	float *inputs;
	backProp_helper(){}
	backProp_helper(float* _innerDelta, float *_inputs) : innerDelta(_innerDelta), inputs(_inputs){}
	float operator()(int t) {
		float local=1.0/(1.0+exp(-inputs[t]));
		local=local*(1.0-local);
		return innerDelta[t]*local;//*/
		/*float local = tanh(inputs[t]);
		local*=local;
		local=1.0f-local;
    	return innerDelta[t]*local;//*/
	}
};
//not used now
struct backProp_helper2 : public unary_function<int, float> {
	float *outputs;
	float *innerDelta;
	backProp_helper2(){}
	backProp_helper2(float *_outputs, float* _innerDelta) : innerDelta(_innerDelta), outputs(_outputs){}
	float operator()(int t) {
		float local=outputs[t];
		local=local*(1.0-local);
		return innerDelta[t]*local;
	}
};
struct output_helper : public unary_function<int, float> {
	float *inputs;
	float *outputs;
	float *labels;
	float *innerDelta;
	output_helper(float *_outputs, float *_inputs, float* _innerDelta, float* _labels) : outputs(_outputs), inputs(_inputs), innerDelta(_innerDelta), labels(_labels){}
	float operator()(int t) {
		float local=outputs[t]-labels[t];
		float local2=1.0f/(1.0f+exp(-inputs[t]));
		local2=local2*(1.0f-local2);
		return local2*local;//*/
		/*float local2=tanh(inputs[t]);
		local2*=local2;
		local2=1.0f-local2;
		return local2*local;//*/
	}
};
struct output_helper2 : public unary_function<int, float> {
	float *inputs;
	float *outputs;
	float label;
	float *innerDelta;
	output_helper2(float *_outputs, float *_inputs, float* _innerDelta, float _label) : outputs(_outputs), inputs(_inputs), innerDelta(_innerDelta), label(_label){}
	float operator()(int t) {
		float local=outputs[t]-label;
		float local2=1.0f/(1.0f+exp(-inputs[t]));
		local2=local2*(1.0f-local2);
		return local2*local;//*/
		/*float local2=tanh(inputs[t]);
		local2*=local2;
		local2=1.0f-local2;
		return local2*local;//*/
	}
};
struct divThreads : public unary_function<float, float> {
	float what;
	divThreads(float _what) : what(_what){}
	float operator()(float t) {
		return t/what;
	}
};
struct multi_helper : public unary_function<float, float> {
	float what;
	multi_helper(float _what) : what(_what){}
	float operator()(float t) {
		return t*what;
	}
};

class NN_layer {
public:

	//vector<float> atNeuronOutputs;
	//vector<float> atNeuronInputs;
	vector<float> weightsMatrix;
	vector<float> biases;
	//vector<float> outerDeltaB;
	//vector<float> outerDeltaW;
	//vector<float> innerDeltaB;
	//vector<float> innerDeltaW;

	NN_layer(){}
	NN_layer(int sizeThis, int sizeNext, int pType, int _batchSize, bool newLayer) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext), batchSize(_batchSize) {

		allW=thisSize*nextSize;
		allN=thisSize*batchSize;

		ULLI two=2;
		ULLI four=4;
		ULLI uallW=(ULLI)allW;
		ULLI uallN=(ULLI)allN;
		
		if(type!=INPUT) {
			biases=vector<float>(allN);
			memTracker(uallN*four*two,false);
		}
		if(type!=OUTPUT) {
			weightsMatrix=vector<float>(allW);
			memTracker(uallW*four*two,false);
		}
	}
	NN_layer(int sizeThis, int sizeNext, int pBatchSize, int pType) : 
			type(pType), thisSize(sizeThis), nextSize(sizeNext), batchSize(pBatchSize) {

		setupLayer(true);
	}

	void setupLayer(bool newLayer) {
		allW=thisSize*nextSize;
		allN=thisSize*batchSize;
		//atNeuronOutputs=vector<float>(allN);

		ULLI two=2;
		ULLI three=3;
		ULLI four=4;
		ULLI uallW=(ULLI)allW;
		ULLI uallN=(ULLI)allN;

		memTracker(uallN*four,false);
		if(newLayer) {
			if(type!=INPUT) {
				//atNeuronInputs=vector<float>(allN);
				//innerDeltaB=vector<float>(allN);
				//outerDeltaB=vector<float>(allN);
				memTracker(uallN*four*three,false);
				biases=vector<float>(allN);
				if(!my_rank) {
					transform(biases.begin(),biases.end(),biases.begin(),fix_random_numbers());
				}
			}
			if(type!=OUTPUT) {
				weightsMatrix=vector<float>(allW);
				//innerDeltaW=vector<float>(allW);
				//outerDeltaW=vector<float>(allW);
				memTracker(uallW*four*two,false);
				if(!my_rank) {
					transform(weightsMatrix.begin(),weightsMatrix.end(),weightsMatrix.begin(),fix_random_numbers());
				}
				if(!my_rank) {
					cout << "thisSize: " << thisSize << " nextSize: " << nextSize << " thisSize*nextSize: " << (thisSize*nextSize) << endl;
				}
			}
		} else {
			if(type!=INPUT) {
				//atNeuronInputs=vector<float>(allN);
				//outerDeltaB=vector<float>(allN);
				//innerDeltaB=vector<float>(allN);
			}
			if(type!=OUTPUT) {
				//outerDeltaW=vector<float>(allW);
				//innerDeltaW=vector<float>(allW);
			}
		}
	}

	bool failed=false;
	int type, batchSize;
	int thisSize, nextSize, allW, allN;
};
vector<NN_layer> NNlayersQ;

void *threadedANN(void *thread_parm) {
    idLink2 data=*((idLink2*) thread_parm);
	int whichPatient=data.whichPatient;
	int my_id=data.my_id;
	int trainSetSize=threadedToDoList[my_id].size();
	int mOut, prevSize, nextSize, thisSize, mPlus, ii;
	int batchSize=1, numOutputs=1;
	float local, local2;
	float *which;
	vector<float> twoFile(fsize);
	int outputsIndex=hiddenMatrix.size()-1;
	int oMinusOne=outputsIndex-1;
	while(!threadExit) {
		for(int i=0;i<outputsIndex;++i) {
			fill(threadOuterDeltaW[my_id][i].begin(),threadOuterDeltaW[my_id][i].end(),0.0f);
			fill(threadOuterDeltaB[my_id][i].begin(),threadOuterDeltaB[my_id][i].end(),0.0f);
		}
		for(int j=0;j<trainSetSize;++j) {
			getFile2(setLabeledPatients[threadedToDoList[my_id][j]].first, &twoFile[0]);

			//forward propagation
			which=&oneFile[0];
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[ii];
				cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayersQ[i].weightsMatrix[0], nextSize, which, thisSize, 0.0, &threadInputs[my_id][i][0], nextSize);
				transform(NNlayersQ[ii].biases.begin(),NNlayersQ[ii].biases.end(),threadInputs[my_id][i].begin(),threadInputs[my_id][i].begin(),plus<float>());
				transform(threadInputs[my_id][i].begin(),threadInputs[my_id][i].end(),threadOutputs[my_id][i].begin(),sigmoid());
				which=&threadOutputs[my_id][i][0];
			}

			mOut=outputsIndex-1;
			if(!doingMNIST && !my_rank && !my_id) {//} && pred) {
				cout << hostname+" backprop file: "+setLabeledPatients[threadedToDoList[my_id][j]].first << " should be " << (int)setLabeledPatients[threadedToDoList[my_id][j]].second << " pred: " << (int)(threadOutputs[my_id][oMinusOne][0]+0.5f) << " is: " << threadOutputs[my_id][oMinusOne][0] << "\r";//endl;
			}

			//Backward propagation
			mPlus=outputsIndex;
			prevSize=hiddenMatrix[mOut];

			local=threadOutputs[my_id][oMinusOne][0]-setLabeledPatients[threadedToDoList[my_id][j]].second;
			local2=1.0f/(1.0f+exp(-threadInputs[my_id][oMinusOne][0]));
			local2=local2*(1.0f-local2);
			threadInnerDeltaB[my_id][oMinusOne][0]=local2*local;
			//transform(NNlayersQ[outputsIndex].counterN.begin(),NNlayersQ[outputsIndex].counterN.end(),NNlayersQ[outputsIndex].innerDeltaB.begin(),output_helper2(&NNlayersQ[outputsIndex].atNeuronOutputs[0],&NNlayersQ[outputsIndex].atNeuronInputs[0],&NNlayersQ[outputsIndex].innerDeltaB[0],setLabeledPatients[toTrainList[j]].second));
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, numOutputs, prevSize, batchSize, 1.0, &threadInnerDeltaB[my_id][oMinusOne][0], numOutputs, &threadOutputs[my_id][oMinusOne][0], prevSize, 0.0, &threadInnerDeltaW[my_id][oMinusOne][0], numOutputs);

			--mOut;
			for(int i=outputsIndex-1;i;--i) {
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[i+1];
				prevSize=hiddenMatrix[i-1];
				cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, thisSize, batchSize, nextSize, 1.0, &NNlayersQ[i].weightsMatrix[0], nextSize, &threadInnerDeltaB[my_id][i][0], nextSize, 0.0, &threadInnerDeltaB[my_id][mOut][0], thisSize);
				transform(threadInputs[my_id][mOut].begin(),threadInputs[my_id][mOut].end(),threadInputs[my_id][mOut].begin(),sigmoid_devrivative());
				transform(threadInputs[my_id][mOut].begin(),threadInputs[my_id][mOut].end(),threadInnerDeltaB[my_id][mOut].begin(),threadInnerDeltaB[my_id][mOut].begin(),multiplies<float>());
				if(i!=1) {
					cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &threadInnerDeltaB[my_id][mOut][0], thisSize, &threadOutputs[my_id][mOut-1][0], prevSize, 0.0, &threadInnerDeltaW[my_id][mOut][0], thisSize);
				} else {
					cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &threadInnerDeltaB[my_id][mOut][0], thisSize, &twoFile[0], prevSize, 0.0, &threadInnerDeltaW[my_id][mOut][0], thisSize);
				}
				--mOut;
				--mPlus;
			}
			for(int i=0;i<outputsIndex;++i) {
				transform(threadInnerDeltaB[my_id][i].begin(),threadInnerDeltaB[my_id][i].end(),threadOuterDeltaB[my_id][i].begin(),threadOuterDeltaB[my_id][i].begin(),plus<float>());
				transform(threadInnerDeltaW[my_id][i].begin(),threadInnerDeltaW[my_id][i].end(),threadOuterDeltaW[my_id][i].begin(),threadOuterDeltaW[my_id][i].begin(),plus<float>());
			}

		}
		pthread_barrier_wait(&barrier);
		pthread_barrier_wait(&barrier2);

	}
    pthread_exit(0);
}

class neuralNet {
public:
	neuralNet(){}

	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix, int pBatchSize) : 
		RMS(DBL_MAX), minRMS(DBL_MAX), batchSize(pBatchSize) {
		mkl_set_num_threads(1);
		hiddenMatrix=_hiddenMatrix;
		/*for(auto h:hiddenMatrix) {
			cout << hostname << ": " << h << endl;
		}*/
		//openblas_set_num_threads(1);
		//numThreads=numberOfProcessors;
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);
		epoch=0;

		layers=hiddenMatrix.size();
		NNlayersQ=vector<NN_layer>(layers);
		int outputsIndex=layers-1;
		if(!my_rank) {
			cout << "Setting up network...\n";
			cout << "Layers: ";
			for(auto h:hiddenMatrix) {
				cout << h << " ";
			}
			cout << "Batch size: " << batchSize << endl << endl;
		}

		NNlayersQ[0]=NN_layer(hiddenMatrix[0],hiddenMatrix[1],batchSize,INPUT);
		if(NNlayersQ[0].failed) {
			cout << hostname << " failed creating input array\n";
			return;
		}
		for(int i=1;i<outputsIndex;++i) {
			NNlayersQ[i]=NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],batchSize,HIDDEN);
			if(NNlayersQ[i].failed) {
				cout << hostname << " failed creating a hidden array\n";
				return;
			}
		}
		NNlayersQ[outputsIndex]=NN_layer(hiddenMatrix[outputsIndex],0,batchSize,OUTPUT);
		if(NNlayersQ[outputsIndex].failed) {
			cout << hostname << " failed creating output array\n";
			return;
		}

		threadOuterDeltaB=vector<vector<vector<float>>>(numberOfProcessors);
		threadOuterDeltaW=vector<vector<vector<float>>>(numberOfProcessors);
		threadInnerDeltaB=vector<vector<vector<float>>>(numberOfProcessors);
		threadInnerDeltaW=vector<vector<vector<float>>>(numberOfProcessors);
		threadOutputs=vector<vector<vector<float>>>(numberOfProcessors);
		threadInputs=vector<vector<vector<float>>>(numberOfProcessors);
		for(int i=0;i<numberOfProcessors;++i) {
			threadOutputs[i]=vector<vector<float>>(layers-1);
			threadInputs[i]=vector<vector<float>>(layers-1);
			threadOuterDeltaB[i]=vector<vector<float>>(layers-1);
			threadOuterDeltaW[i]=vector<vector<float>>(layers-1);
			threadInnerDeltaB[i]=vector<vector<float>>(layers-1);
			threadInnerDeltaW[i]=vector<vector<float>>(layers-1);
			for(int j=0;j<outputsIndex;++j) {
				threadOutputs[i][j]=vector<float>(NNlayersQ[j+1].allN);
				threadInputs[i][j]=vector<float>(NNlayersQ[j+1].allN);
				threadOuterDeltaW[i][j]=vector<float>(NNlayersQ[j].allW);
				threadOuterDeltaB[i][j]=vector<float>(NNlayersQ[j+1].allN);
				threadInnerDeltaW[i][j]=vector<float>(NNlayersQ[j].allW);
				threadInnerDeltaB[i][j]=vector<float>(NNlayersQ[j+1].allN);
			}
		}

		int ii;
		for(int i=0;i<outputsIndex;++i) {
			ii=i+1;
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&NNlayersQ[i].weightsMatrix[0],NNlayersQ[i].allW, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&NNlayersQ[ii].biases[0],NNlayersQ[ii].allN, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
		}//*/
	}

	void evaluate() {
		vector<string> evalFiles;
		vector<string> testFiles;
		vector<string> submission;
		submission.push_back("id,cancer");
		int outputsIndex=hiddenMatrix.size()-1;
		mkl_set_num_threads(numberOfProcessors);
		bool once=true;
		char temp[500];
		ifstream iFile("MPIversion/stage1_labels.csv",ios::in|ios::binary);
		if(iFile.is_open()) {
			while(iFile.getline(temp,500)) {
				if(once) {
					once=false;
				} else {
					string temp2=string(temp);
					vector<string> splitTemp=split(temp2,",");
					evalFiles.push_back(splitTemp[0]);
				}
			}
			iFile.close();
		}
		unordered_map<string,char> patientsSearch;
		for(auto p:evalFiles) {
			patientsSearch.insert(pair<string,char>(p+".dat",0));
		}
		string p;
		for(int i=0;i<patients.size();++i) {
			p=patients[i];
			if(patientsSearch.find(p)==patientsSearch.end()) {
				p.erase(p.begin()+p.length()-1);
				p.erase(p.begin()+p.length()-1);
				p.erase(p.begin()+p.length()-1);
				p.erase(p.begin()+p.length()-1);
				testFiles.push_back(p);
			}
		}
		sort(testFiles.begin(),testFiles.end());
		int totalTestSize=testFiles.size();
		float *which;
		int thisSize,nextSize;
		int ii;
		for(int j=0;j<totalTestSize;++j) {
			getFile(testFiles[j]);
			cout << "Processing file: " << testFiles[j] << " prediction: ";
			which=&oneFile[0];
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[ii];
				cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayersQ[i].weightsMatrix[0], nextSize, which, thisSize, 0.0, &threadInputs[0][i][0], nextSize);
				transform(NNlayersQ[ii].biases.begin(),NNlayersQ[ii].biases.end(),threadInputs[0][i].begin(),threadInputs[0][i].begin(),plus<float>());
				transform(threadInputs[0][i].begin(),threadInputs[0][i].end(),threadOutputs[0][i].begin(),sigmoid());
				//transform(NNlayersQ[ii].counterN.begin(),NNlayersQ[ii].counterN.end(),NNlayersQ[ii].atNeuronOutputs.begin(),forwardFeed_helper(&NNlayersQ[ii].atNeuronInputs[0],&NNlayersQ[ii].biases[0]));
				which=&threadOutputs[0][i][0];
			}
			cout << setprecision(15);
			cout << fixed << threadOutputs[0][outputsIndex][0] << endl;
			float out=threadOutputs[0][outputsIndex][0];//+0.5;
			//int outi=(int)out;
			//out=(float)outi;
			sprintf(temp,"%s,%.10f",testFiles[j].c_str(),out);
			submission.push_back(string(temp));
		}
		fstream ofile("MPIversion/submissionFile.csv",ios::out|ios::binary);
		if(ofile.is_open()) {
			for(auto s:submission) {
				ofile << s << endl;
			}
			ofile.close();
		}
	}

	void train_kaggle(ULLI maxIter, float lRate, bool vlRate, float RMSwant, int totalTestSize, int totalTrainSize) {
		cout << setprecision(20);
		if(!showInterval) {
			showInterval=1;
		}
		if(lRate<=0.0f) {
			learningRate=0.05f;
			lRate=learningRate;	
		} else {
			learningRate=lRate;
		}
		int testSetSize=toTestList.size();
		int trainSetSize=toTrainList.size();
		int mOut, ii;
		int maxTrainRight=-1, maxTestRight=-1, testRight, trainRight, tempInt;
		float seconds=0.0f, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		int mPlus;

		int layers=hiddenMatrix.size();
		int outputsIndex=layers-1, imax;
		int numOutputs=hiddenMatrix[outputsIndex];
		int prevSize, nextSize, thisSize;//, partial, umax=(ULLI)INT_MAX, ustart;
		float *which;
		bool gotTime=false;
		int timeCountDown=10;
		float trainRMS, minTrainRMS=FLT_MAX, tempRMS;
		float trainLogLoss, minTrainLogLoss=FLT_MAX, testLogLoss, minTestLogLoss=FLT_MAX;
		float yi, yhat, local, local2;
		int printWrongOnes=0;
		minRMS=FLT_MAX;
		if(!my_rank) {
			memTracker(0,true);
		}
		//float toDivide=lRate/((float)batchSize*(float)trainSetSize*(float)num_nodes);
		//float toDivide=lRate/((float)batchSize*(float)trainSetSize*(float)showInterval);
		float toDivide=lRate/((float)batchSize*(float)trainSetSize);
		divThreads dThreads(lRate);
		int otherTest=0;
		string temp;
		bool doPred=false;
		int pred;
		//float toDivide=lRate/((float)batchSize*(float)num_nodes);
		bool once=true;
		vector<pthread_t> threads;
		pthread_attr_t attr;
    	cpu_set_t cpus;
    	pthread_attr_init(&attr);
    	int oMinusOne=outputsIndex-1;

    	int index=0;
    	threadedToDoList=vector<vector<int>>(numberOfProcessors);
    	while(index<trainSetSize) {
    		for(int i=0;i<numberOfProcessors && index<trainSetSize;++i) {
	    		threadedToDoList[i].push_back(toTrainList[index++]);
    		}
    	}
    	MPI_Barrier(MPI_COMM_WORLD);

		for(;!threadExit && epoch<maxIter;) {
			startTime=high_resolution_clock::now();

			if(once) {
			    for(int j=0;j<numberOfProcessors;++j) {

     				CPU_ZERO(&cpus);
       				CPU_SET(j, &cpus);
       				pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

			        threads.push_back(pthread_t());
			        idLink2 *arg = (idLink2*)malloc(sizeof(*arg));
			        (*arg).whichPatient=j;
			        (*arg).my_id=j;
			        pthread_create(&threads.at(j), &attr,  threadedANN, arg);
			    }
			    once=false;
			}

			pthread_barrier_wait(&barrier);
			mkl_set_num_threads(numberOfProcessors);

			++epoch;

			for(int j=1;j<numberOfProcessors;++j) {
				for(int i=0;i<outputsIndex;++i) {
					__gnu_parallel::transform(threadOuterDeltaW[0][i].begin(),threadOuterDeltaW[0][i].end(),threadOuterDeltaW[j][i].begin(),threadOuterDeltaW[0][i].begin(),plus<float>());
					__gnu_parallel::transform(threadOuterDeltaB[0][i].begin(),threadOuterDeltaB[0][i].end(),threadOuterDeltaB[j][i].begin(),threadOuterDeltaB[0][i].begin(),plus<float>());
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			for(int i=0;i<outputsIndex;++i) {
				ii=i+1;
				MPI_Allreduce(&threadOuterDeltaW[0][i][0],&threadOuterDeltaW[1][i][0],NNlayersQ[i].allW,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				MPI_Allreduce(&threadOuterDeltaB[0][i][0],&threadOuterDeltaB[1][i][0],NNlayersQ[ii].allN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
				//if(!my_rank) {
				//	cout << hostname << " lrate: " << lRate << " totalTrainSize: " << totalTrainSize << endl;
				//}
				__gnu_parallel::transform(threadOuterDeltaW[1][i].begin(), threadOuterDeltaW[1][i].end(), threadOuterDeltaW[1][i].begin(), multi_helper(lRate/(float)totalTrainSize));
				__gnu_parallel::transform(NNlayersQ[i].weightsMatrix.begin(), NNlayersQ[i].weightsMatrix.end(), threadOuterDeltaW[1][i].begin(), NNlayersQ[i].weightsMatrix.begin(), minus<float>());
				__gnu_parallel::transform(threadOuterDeltaB[1][i].begin(), threadOuterDeltaB[1][i].end(), threadOuterDeltaB[1][i].begin(), multi_helper(lRate/(float)totalTrainSize));
				__gnu_parallel::transform(NNlayersQ[ii].biases.begin(), NNlayersQ[ii].biases.end(), threadOuterDeltaB[1][i].begin(), NNlayersQ[ii].biases.begin(), minus<float>());
			}

			showIntervalCountDown=showInterval;
			RMS=0.0f;
			trainRMS=0.0f;
			trainLogLoss=0.0f;
			testLogLoss=0.0f;
			trainRight=0;
			otherTest=0;
			for(int j=0;j<trainSetSize;++j) {
				getFile(setLabeledPatients[toTrainList[j]].first);

				//forward propagation
				which=&oneFile[0];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayersQ[i].weightsMatrix[0], nextSize, which, thisSize, 0.0, &threadInputs[0][i][0], nextSize);
					__gnu_parallel::transform(NNlayersQ[ii].biases.begin(),NNlayersQ[ii].biases.end(),threadInputs[0][i].begin(),threadInputs[0][i].begin(),plus<float>());
					__gnu_parallel::transform(threadInputs[0][i].begin(),threadInputs[0][i].end(),threadOutputs[0][i].begin(),sigmoid());
					which=&threadOutputs[0][i][0];
				}
				if(!printWrongOnes) {
					if(!my_rank && !doingMNIST) {
						cout << hostname+" train set file: "+setLabeledPatients[toTrainList[j]].first << " should be " << (int)setLabeledPatients[toTrainList[j]].second << " pred: " << (int)(threadOutputs[0][oMinusOne][0]+0.5f) << " is: " << threadOutputs[0][oMinusOne][0] << "\r";
					}
				} else {
					if(!doingMNIST) {
						if((int)setLabeledPatients[toTrainList[j]].second!=(int)(threadOutputs[0][oMinusOne][0]+0.5f)) {
							cout << hostname+" train set file: "+setLabeledPatients[toTrainList[j]].first << " should be " << (int)setLabeledPatients[toTrainList[j]].second << " pred: " << (int)(threadOutputs[0][oMinusOne][0]+0.5f) << " is: " << threadOutputs[0][oMinusOne][0] << "\n";
						}
					}
				}
				yi=setLabeledPatients[toTrainList[j]].second;
				yhat=threadOutputs[0][oMinusOne][0];
				trainLogLoss+=(yi*log(yhat)+(1.0f-yi)*(log(1.0f-yhat)));
				trainRMS+=pow(setLabeledPatients[toTrainList[j]].second-threadOutputs[0][oMinusOne][0],2.0f);
				if(doingMNIST) {
					yhat*=10.0f;
					yi*=10.0f;
					yhat+=0.5f;
					int yhatt=(int)yhat;
					int yii=(int)yi;
					if(yhatt==yii) {
						++trainRight;
					}
				} else {
					yhat+=0.5f;
					int yhatt=(int)yhat;
					int yii=(int)yi;
					if(yhatt==yii) {
						++trainRight;
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&trainRight,&tempInt,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
			trainRight=tempInt;
			if(trainRight>maxTrainRight){maxTrainRight=trainRight;}

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&trainRMS,&tempRMS,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			trainRMS=sqrt(tempRMS/(float)totalTrainSize);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&trainLogLoss,&tempRMS,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			trainLogLoss=tempRMS*(-1.0f/(float)totalTrainSize);
			if(trainLogLoss<minTrainLogLoss){minTrainLogLoss=trainLogLoss;}
			if(trainRMS<minTrainRMS){minTrainRMS=trainRMS;}

			testRight=0;
			for(int j=0;j<testSetSize;++j) {
				getFile(setLabeledPatients[toTestList[j]].first);

				//forward propagation
				which=&oneFile[0];
				for(int i=0;i<outputsIndex;++i) {
					ii=i+1;
					thisSize=hiddenMatrix[i];
					nextSize=hiddenMatrix[ii];
					cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayersQ[i].weightsMatrix[0], nextSize, which, thisSize, 0.0, &threadInputs[0][i][0], nextSize);
					__gnu_parallel::transform(NNlayersQ[ii].biases.begin(),NNlayersQ[ii].biases.end(),threadInputs[0][i].begin(),threadInputs[0][i].begin(),plus<float>());
					__gnu_parallel::transform(threadInputs[0][i].begin(),threadInputs[0][i].end(),threadOutputs[0][i].begin(),sigmoid());
					which=&threadOutputs[0][i][0];
				}
				if(!printWrongOnes) {
					if(!my_rank && !doingMNIST) {
						cout << hostname+" test set file: "+setLabeledPatients[toTestList[j]].first << " should be " << (int)setLabeledPatients[toTestList[j]].second << " pred: " << (int)(threadOutputs[0][oMinusOne][0]+0.5f) << " is: " << threadOutputs[0][oMinusOne][0] << "\r";
					}
				} else {
					if(!doingMNIST) {
						if((int)setLabeledPatients[toTestList[j]].second!=(int)(threadOutputs[0][oMinusOne][0]+0.5f)) {
							cout << hostname+" test set file: "+setLabeledPatients[toTestList[j]].first << " should be " << (int)setLabeledPatients[toTestList[j]].second << " pred: " << (int)(threadOutputs[0][oMinusOne][0]+0.5f) << " is: " << threadOutputs[0][oMinusOne][0] << "\n";
						}
					}
				}
				yi=setLabeledPatients[toTestList[j]].second;
				yhat=threadOutputs[0][oMinusOne][0];
				testLogLoss+=(yi*log(yhat)+(1.0f-yi)*(log(1.0f-yhat)));
				RMS+=pow(setLabeledPatients[toTestList[j]].second-threadOutputs[0][oMinusOne][0],2.0f);
				if(doingMNIST) {
					yhat*=10.0f;
					yi*=10.0f;
					yhat+=0.5f;
					int yhatt=(int)yhat;
					int yii=(int)yi;
					if(yhatt==yii) {
						++testRight;
					}
				} else {
					if(yhat>0.4f) {
						if(yi>0.9f) {
							++otherTest;
						}
					} else {
						if(yi<0.1f) {
							++otherTest;
						}
					}
					yhat+=0.5f;
					int yhatt=(int)yhat;
					int yii=(int)yi;
					//if(!my_rank) {
					//	cout << "yhat: " << yhat << " -- yi: " << yi << " -- yhatt: " << yhatt << " -- yii: " << yii << endl;
					//}
					if(yhatt==yii) {
						++testRight;
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&otherTest,&tempInt,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
			otherTest=tempInt;

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&testRight,&tempInt,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
			testRight=tempInt;
			if(testRight>maxTestRight){maxTestRight=testRight;}

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&RMS,&tempRMS,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			RMS=sqrt(tempRMS/(float)totalTestSize);
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(&testLogLoss,&tempRMS,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
			testLogLoss=tempRMS*(-1.0f/(float)totalTestSize);
			if(testLogLoss<minTestLogLoss){minTestLogLoss=testLogLoss;}
			if(RMS<minRMS){minRMS=RMS;}
			if(!my_rank) {
				endTime=high_resolution_clock::now();
				seconds+=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
				cout << endl;
				if((totalTrainSize-trainRight)<100) {
					printWrongOnes=1;
				} else {
					printWrongOnes=0;
				}
				cout << "Epoch: " << epoch << "\t-- Total time: " << seconds << endl << "TestLogLoss: " << testLogLoss << "\t-- minTestLogLoss: " << minTestLogLoss << endl;
				cout << "TrainLogLoss: " << trainLogLoss << "\t-- minTrainLogLoss: " << minTrainLogLoss << endl;
				cout << "TrainRMS: " << trainRMS << "\t-- minTrainRMS: " << minTrainRMS << "\t-- RMS: " << RMS << "\t-- minRMS: " << minRMS << endl;
				cout << "trainRight: " << trainRight << "\t-- maxTrainRight: " << maxTrainRight << " (out of " << totalTrainSize << ")\t-- testRight: " << testRight << "\t-- maxTestRight: " << maxTestRight << " (out of " << totalTestSize << ")" << endl;
				if(!doingMNIST) {
					cout << "otherTest: " << otherTest << " -- zerosInTestSet: " << zerosInTestSet << " -- zerosInTrainSet: " << zerosInTrainSet << endl;
				}
				saveState();
			}
			MPI_Barrier(MPI_COMM_WORLD);
			int tempB=0;
			MPI_Bcast(&printWrongOnes, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			mkl_set_num_threads(1);
			pthread_barrier_wait(&barrier2);	
		}

	}

	neuralNet(string _inFile) : inputFile(_inFile) {
		mkl_set_num_threads(1);
		if(!my_rank) {
			cout << "Setting up network...\n";
		}
		loadState();
		if(!my_rank) {
			cout << "Layers: ";
			for(auto h:hiddenMatrix) {
				cout << h << " ";
			}
			cout << "Batch size: " << batchSize << endl << endl;
		}
		int outputsIndex=hiddenMatrix.size()-1;
		threadOuterDeltaB=vector<vector<vector<float>>>(numberOfProcessors);
		threadOuterDeltaW=vector<vector<vector<float>>>(numberOfProcessors);
		threadInnerDeltaB=vector<vector<vector<float>>>(numberOfProcessors);
		threadInnerDeltaW=vector<vector<vector<float>>>(numberOfProcessors);
		threadOutputs=vector<vector<vector<float>>>(numberOfProcessors);
		threadInputs=vector<vector<vector<float>>>(numberOfProcessors);
		for(int i=0;i<numberOfProcessors;++i) {
			threadOutputs[i]=vector<vector<float>>(layers-1);
			threadInputs[i]=vector<vector<float>>(layers-1);
			threadOuterDeltaB[i]=vector<vector<float>>(layers-1);
			threadOuterDeltaW[i]=vector<vector<float>>(layers-1);
			threadInnerDeltaB[i]=vector<vector<float>>(layers-1);
			threadInnerDeltaW[i]=vector<vector<float>>(layers-1);
			for(int j=0;j<outputsIndex;++j) {
				threadOutputs[i][j]=vector<float>(NNlayersQ[j+1].allN);
				threadInputs[i][j]=vector<float>(NNlayersQ[j+1].allN);
				threadOuterDeltaW[i][j]=vector<float>(NNlayersQ[j].allW);
				threadOuterDeltaB[i][j]=vector<float>(NNlayersQ[j+1].allN);
				threadInnerDeltaW[i][j]=vector<float>(NNlayersQ[j].allW);
				threadInnerDeltaB[i][j]=vector<float>(NNlayersQ[j+1].allN);
			}
		}
	}

	void saveState() {
		if(!doingMNIST) {
			cout << "Saving state to file\n";
			ChangeDir("MPIversion");
		}
		int outputsIndex=hiddenMatrix.size()-1;
		fstream oFile("latestState.mat",ios::binary|ios::out);
		if(oFile.is_open()) {
			oFile.write((char*)&epoch,sizeof(ULLI));
			oFile.write((char*)&layers,sizeof(int));
			for(int i=0;i<layers;++i) {
				int l=hiddenMatrix[i];
				oFile.write((char*)&l,sizeof(int));
			}
			oFile.write((char*)&batchSize,sizeof(int));
			for(int i=0;i<outputsIndex;++i) {
				oFile.write((char*)&NNlayersQ[i].weightsMatrix[0],NNlayersQ[i].allW*sizeof(float));
				oFile.write((char*)&NNlayersQ[i+1].biases[0],NNlayersQ[i+1].allN*sizeof(float));
			}
			int l=trainingSet.size();
			oFile.write((char*)&l,sizeof(int));
			for(auto t:trainingSet) {
				l=t.first.size();
				oFile.write((char*)&l,sizeof(int));
				oFile.write((char*)&t.first[0],l);
				oFile.write((char*)&t.second,sizeof(int));
			}
			l=trainingLabels.size();
			oFile.write((char*)&l,sizeof(int));
			for(auto t:trainingLabels) {
				oFile.write((char*)&t.first,sizeof(float));
				oFile.write((char*)&t.second,sizeof(int));
			}
			l=testSet.size();
			oFile.write((char*)&l,sizeof(int));
			for(auto t:testSet) {
				l=t.first.size();
				oFile.write((char*)&l,sizeof(int));
				oFile.write((char*)&t.first[0],l);
				oFile.write((char*)&t.second,sizeof(int));
			}
			l=testLabels.size();
			oFile.write((char*)&l,sizeof(int));
			for(auto t:testLabels) {
				oFile.write((char*)&t.first,sizeof(float));
				oFile.write((char*)&t.second,sizeof(int));
			}
			oFile.close();
		}
		if(!doingMNIST) {
			cout << "Done saving state\n";
			ChangeDir("..");
		}
	}

	void loadState() {
		if(!my_rank) {
			cout << hostname << " reading weights from file: " << inputFile << endl;
		}
		ifstream oFile(inputFile, ios::binary|ios::in);
		if(oFile.is_open()) {
			oFile.read((char*)&epoch,sizeof(ULLI));
			oFile.read((char*)&layers,sizeof(int));
			int outputsIndex=layers-1;
			hiddenMatrix.clear();
			for(int i=0;i<layers;++i) {
				int l=0;
				oFile.read((char*)&l,sizeof(int));
				hiddenMatrix.push_back(l);
			}
			oFile.read((char*)&batchSize,sizeof(int));
			numInputs=hiddenMatrix[0];
			numOutputs=hiddenMatrix[outputsIndex];

			NNlayersQ.clear();
			int type=INPUT;
			for(int i=0;i<outputsIndex;++i) {
				if(i){type=HIDDEN;}
				NNlayersQ.push_back(NN_layer(hiddenMatrix[i],hiddenMatrix[i+1],type,batchSize,false));
			}
			NNlayersQ.push_back(NN_layer(hiddenMatrix[outputsIndex],0,OUTPUT,batchSize,false));
			for(int i=0;i<outputsIndex;++i) {
				oFile.read((char*)&NNlayersQ[i].weightsMatrix[0],NNlayersQ[i].allW*sizeof(float));
				oFile.read((char*)&NNlayersQ[i+1].biases[0],NNlayersQ[i+1].allN*sizeof(float));
			}
			for(int i=0;i<layers;++i) {
				NNlayersQ[i].setupLayer(false);
			}
			int ll=0,l=0;
			oFile.read((char*)&ll,sizeof(int));
			if(!reshuffle) {
				trainingSet=vector<pair<string,int>>(ll);
			}
			for(int i=0;i<ll;++i) {
				oFile.read((char*)&l,sizeof(int));
				string temps(l,' ');
				int tempi=0;
				oFile.read((char*)&temps[0],l);
				oFile.read((char*)&tempi,sizeof(int));
				if(!reshuffle) {
					trainingSet[i].first=temps;
					trainingSet[i].second=tempi;
				}
			}
			oFile.read((char*)&ll,sizeof(int));
			if(!reshuffle) {
				trainingLabels=vector<pair<float,int>>(ll);
			}
			for(int i=0;i<ll;++i) {
				float tempf=0.0f;
				float tempi=0;
				oFile.read((char*)&tempf,sizeof(float));
				oFile.read((char*)&tempi,sizeof(int));
				if(!reshuffle) {
					trainingLabels[i].first=tempf;
					trainingLabels[i].second=tempi;
				}
			}
			oFile.read((char*)&ll,sizeof(int));
			if(!reshuffle) {
				testSet=vector<pair<string,int>>(ll);
			}
			for(int i=0;i<ll;++i) {
				oFile.read((char*)&l,sizeof(int));
				string temps(l,' ');
				int tempi=0;
				oFile.read((char*)&temps[0],l);
				oFile.read((char*)&tempi,sizeof(int));
				if(!reshuffle) {
					testSet[i].first=temps;
					testSet[i].second=tempi;
				}
			}
			oFile.read((char*)&ll,sizeof(int));
			if(!reshuffle) {
				testLabels=vector<pair<float,int>>(ll);
			}
			for(int i=0;i<ll;++i) {
				float tempf=0.0f;
				int tempi=0;
				oFile.read((char*)&tempf,sizeof(float));
				oFile.read((char*)&tempi,sizeof(int));
				if(!reshuffle) {
					testLabels[i].first=tempf;
					testLabels[i].second=tempi;
				}
			}
			oFile.close();
		}
		cout << hostname << " done loading state\n";
	}

private:
	ULLI epoch, itemSize;
	int dataSetSize, numInputs, numOutputs, batchSize;
	float RMS, minRMS, toDivideRMS, learningRate;
	int layers;
	string inputFile;
};

void doMain(vector<int> &inputHiddenLayers, int batchSize, int doDataSetSize, float lRate, string inputFile, string outFile, bool vlRate, bool eval, float RMSwant, int totalTestSize, int totalTrainSize) {
	
	vector<int> hiddenMatrix;
	if(!inputHiddenLayers.size()) {
		hiddenMatrix.push_back(200);
		hiddenMatrix.push_back(100);
	} else {
		for(auto h:inputHiddenLayers) {
			hiddenMatrix.push_back(h);
		}
	}

	neuralNet NNmpi(fsize,1,hiddenMatrix,batchSize);
	NNmpi.train_kaggle(100000000,lRate,vlRate,RMSwant,totalTestSize,totalTrainSize);

	return;

}

void fixMNIST_function() {

	auto start = high_resolution_clock::now();
	vector<vector<float>> testData(10000);
	ReadMNIST_float("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<float>> trainData(60000);
	ReadMNIST_float("train-images.idx3-ubyte",60000,784,trainData);
	vector<vector<float>> testLabels(10000);
	vector<vector<float>> trainLabels(60000);
	vector<float> testL(10000);
	ifstream file("t10k-labels.idx1-ubyte",ios::binary);
	if(file.is_open()) {
		int placeHolder=0;
		file.read((char*)&placeHolder,sizeof(placeHolder));
		file.read((char*)&placeHolder,sizeof(placeHolder));
		for(int i=0;i<10000;++i) {
			testLabels[i]=vector<float>(10,0.0);
			UNCHAR temp=0;
			file.read((char*)&temp,1);
			for(UNCHAR j=0;j<10;++j) {
				if(j==temp) {
					testLabels[i][j]=1.0;
					//cout << "test label: " << (int)j << " is now: " << ((float)j/10.0f) << endl;
					testL[i]=((float)j/10.0f);
				}
			}
		}
		file.close();
	}
	ifstream file2("train-labels.idx1-ubyte",ios::binary);
	vector<float> trainL(60000);
	if(file2.is_open()) {
		int placeHolder=0;
		file2.read((char*)&placeHolder,sizeof(placeHolder));
		file2.read((char*)&placeHolder,sizeof(placeHolder));
		for(int i=0;i<60000;++i) {
			trainLabels[i]=vector<float>(10,0.0);
			UNCHAR temp=0;
			file2.read((char*)&temp,1);
			for(UNCHAR j=0;j<10;++j) {
				if(j==temp) {
					trainLabels[i][j]=1.0;
					trainL[i]=((float)j/10.0f);
				}
			}
		}
		file2.close();
	}

	ChangeDir("mnist");
	vector<string> lFile;
	lFile.push_back("id,label");
	string filen;
	char temp[500];
	for(int i=0;i<10000;++i) {
		sprintf(temp,"%05d,%.1f",i,testL[i]);
		lFile.push_back(string(temp));
		sprintf(temp,"%05d.dat",i);
		filen=string(temp);
		fstream ofile(filen,ios::out|ios::binary);
		if(ofile.is_open()) {
			ofile.write((char*)&testData[i][0],testData[i].size()*sizeof(float));
			ofile.close();
		} else {
			cout << " couldn't make a new file\n";
		}
	}
	int index=10000;
	for(int i=0;i<60000;++i) {
		sprintf(temp,"%05d,%.1f",index,trainL[i]);
		lFile.push_back(string(temp));
		sprintf(temp,"%05d.dat",index++);
		filen=string(temp);
		fstream ofile(filen,ios::out|ios::binary);
		if(ofile.is_open()) {
			ofile.write((char*)&trainData[i][0],trainData[i].size()*sizeof(float));
			ofile.close();
		} else {
			cout << " couldn't make a new file\n";
		}
	}
	ChangeDir("..");
	fstream oFile("MNIST_labels.csv",ios::out|ios::binary);
	if(oFile.is_open()) {
		for(auto s:lFile) {
			oFile << s << endl;
		}
		oFile.close();
	}

	auto endTime = high_resolution_clock::now();
	printTime(start,endTime);
}

int main(int argc, char *argv[]) {

	/*cout << sizeof(short int) << endl;
	short int t=-10;
	short int r=22;
	cout << t << " " << r << endl;
	return 0;*/

	/*for(int i=0;i<(21*4);++i) {
		float core=300.0+(float)i;
		cout << (60000.0/core) << " " << core << endl;
	}
	return 0;*/
	if (!GetCurrentDir(CurrentPath, sizeof(CurrentPath))) {
		cout << "Couldn't retrieve current directory\n";
		return 0;
	}

	numberOfProcessors = sysconf(_SC_NPROCESSORS_ONLN);
	pthread_barrier_init(&barrier, NULL, numberOfProcessors+1);
	pthread_barrier_init(&barrier2, NULL, numberOfProcessors+1);
    //printf("Number of processors: %d\n", numberOfProcessors);

	string inputFile="";
	string outFile="";
	int doDataSetSize=0;
	int batchSize=1;
	float lRate=-1.0;
	bool vlRate=false;
	float percentTestSet=10.0f;
	bool doPreproc=false;
	bool remZeros=false;
	float RMSwant=0.0001f;
	if(!vlRate){}
	vector<int> inputHiddenLayers;
	showInterval=0;
	bool fixMNIST=false;
	bool eval=false;
	bool andTrain=false;
	reshuffle=false;
    for(int i=1;i<argc;++i) {
        string temp=string(argv[i]);
        if(temp.find("showInterval=")!=string::npos) {
        	sscanf(argv[i],"showInterval=%d",&showInterval);
        }
        if(temp.find("andTrain")!=string::npos) {
        	andTrain=true;
        }
        if(temp.find("reshuffle")!=string::npos) {
        	reshuffle=true;
        }
        if(temp.find("removeZeros")!=string::npos) {
        	remZeros=true;
        }
        if(temp.find("eval")!=string::npos) {
        	eval=true;
        }
        if(temp.find("fixMNIST")!=string::npos) {
        	fixMNIST=true;
        }
        if(temp.find("mnist")!=string::npos) {
        	doingMNIST=true;
        }
        if(temp.find("vlRate")!=string::npos) {
        	vlRate=true;
        }
        if(temp.find("preproc")!=string::npos) {
        	doPreproc=true;
        }
        if(temp.find("outWeights=")!=string::npos) {
        	outFile=temp.substr(11,temp.size());
        }
        if(temp.find("inWeights=")!=string::npos) {
        	inputFile=temp.substr(10,temp.size());
        }
        if(temp.find("setSize=")!=string::npos) {
        	sscanf(argv[i],"setSize=%d",&doDataSetSize);
        }
        if(temp.find("batchSize=")!=string::npos) {
        	sscanf(argv[i],"batchSize=%d",&batchSize);
        }
        if(temp.find("learningRate=")!=string::npos) {
        	sscanf(argv[i],"learningRate=%f",&lRate);
        }
        if(temp.find("RMSwant=")!=string::npos) {
        	sscanf(argv[i],"RMSwant=%f",&RMSwant);
        }
        if(temp.find("percTest=")!=string::npos) {
        	sscanf(argv[i],"percTest=%f",&percentTestSet);
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

    vector<string> types;

    if(fixMNIST) {
    	fixMNIST_function();
    	return 0;
    }

    //do preprocessing instead and then exit
    if(doPreproc) {
    	ChangeDir("..");
    	ChangeDir("stage1");
    	GetFilesInDirectory(patientFolders,types);
    	sort(patientFolders.begin(),patientFolders.end());
    	//ChangeDir("..");
    	//ChangeDir("..");
    	//ChangeDir("KagglePre");
    	GetFilesInDirectory(patients,types);
    	int numPatients=patients.size();
    	//numPatients=20;
    	sort(patients.begin(), patients.end());
    	for(int i=0;i<patients.size();++i) {
    		patients[i]+=".dat";
    	}
    	ChangeDir("..");
    	ChangeDir("MPIversion");
    	if(remZeros) {
			ifstream inFile("indexesThatCanBeRemoved.dat",ios::in | ios::binary | ios::ate);
		    if (inFile.is_open()) {
			   	ULLI trem = inFile.tellg();
			   	vector<int> toRemoveIn(trem/sizeof(int),0);
			   	inFile.seekg(0,inFile.beg);
		       	inFile.read((char*)&toRemoveIn[0],trem*sizeof(int));
		       	for(auto t:toRemoveIn) {
		       		toRemove.insert(pair<int,UNCHAR>(t,0));
		       	}
		       	inFile.close();
		   	} else {
		   		cout << "Error opening index file\n";
		   		MPI_Finalize();
		   		return 0;
			}
		}

		MPI_Init(&argc, &argv);
	    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
	    char my_host[100];
	    gethostname(my_host, 100);
	    hostname=string(my_host);
	    if(!my_rank) {
	    	cout << "Doing preprocessing...\n";
	    	cout << "Number of patient files: " << numPatients << endl;
	    }
	    for(int i=0;i<numPatients;++i) {
	    	success.push_back(0);
	    }
	    vector<UNCHAR> recvSuccess;
	    //int staging=GETMINMAX;
	    int staging;
	    if(remZeros) {
	    	staging=REMOVEZEROS;
	    } else {
	    	staging=DOPREPROC;
	    }
    	maxFloat=0.0f;
    	minFloat=FLT_MAX;
    	maxDim=0;
    	avgSpacing=0.0f;

	    int placeHolder=my_rank;
	    if (my_rank != MASTER) {
	    	MPI_Sendrecv(&numberOfProcessors, 1, MPI_INT, MASTER, TAG, &placeHolder, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	//used for recieving the total cpus on the job and which parts this computer is doing
	    	int getOne;
	    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, &getOne, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	int *doWhich=(int*)calloc(getOne,sizeof(int));
	    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, doWhich, getOne, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	for(int i=0;i<getOne;++i) {
	    		toDoList.push_back(doWhich[i]);
	    	}
	    	free(doWhich);

	    	vector<pthread_t> threads;
			pthread_attr_t attr;
    		cpu_set_t cpus;
    		pthread_attr_init(&attr);

	    	int toDo=toDoList.size();
	    	if(toDo<numberOfProcessors) {
	    		numberOfProcessors=toDo;
	    	}
	    	for(int i=0;i<numberOfProcessors;++i) {
	    		threadsDone.push_back(true);
	    		threads.push_back(pthread_t());
	    	}
	    	int status;
			void * result;
			int whichPatient;
			while(staging) {
				whichPatient=0;
	    		for(int i=0;i<numberOfProcessors;++i) {
	    			threadsDone.push_back(true);
	    		}
				while(whichPatient<toDo) {
					for(int i=0;i<numberOfProcessors && whichPatient<toDo;++i) {
						pthread_mutex_lock(&crmutex);
						if(threadsDone[i]) {
							pthread_mutex_unlock(&crmutex);
							cout << hostname << " starting job: " << patients[toDoList[whichPatient]] << endl;

	     					CPU_ZERO(&cpus);
	       					CPU_SET(i, &cpus);
	       					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

		        			idLink2 *arg = (idLink2*)malloc(sizeof(*arg));
		        			(*arg).whichPatient=toDoList[whichPatient];
		        			++whichPatient;
		        			(*arg).my_id=i;
		        			threadsDone[i]=false;
		        			if(staging==REMOVEZEROS) {
		        				pthread_create(&threads[i], &attr, preprocThread2, arg);
		        			}
		        			if(staging==GETMINMAX) {
		        				pthread_create(&threads[i], &attr, preprocMaxMinThread, arg);
		        			}
		        			if(staging==DOPREPROC){
		        				pthread_create(&threads[i], &attr, preprocThread, arg);
		        			}
						} else {
							pthread_mutex_unlock(&crmutex);
						}
					}
				}
		    	for (int i=0; i < numberOfProcessors; ++i) {
		        	if ((status = pthread_join(threads[i], &result)) != 0) {
		            	fprintf (stderr, "join error %d: %s\n", status, strerror(status));
			        }
		    	}
		    	cout << hostname << " done with all work\n";
		    	recvSuccess=vector<UNCHAR>(numPatients,0);
		    	MPI_Barrier(MPI_COMM_WORLD);
		    	MPI_Allreduce(&success[0],&recvSuccess[0],numPatients,MPI_UNSIGNED_CHAR,MPI_SUM,MPI_COMM_WORLD);
		    	if(staging==GETMINMAX) {
		    		MPI_Barrier(MPI_COMM_WORLD);
		    		int tempIntt=0;
		    		MPI_Allreduce(&maxDim,&tempIntt,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
		    		maxDim=tempIntt;
		    		float tempFloat=0.0f;
		    		MPI_Allreduce(&maxFloat,&tempFloat,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
		    		maxFloat=tempFloat;
		    		MPI_Allreduce(&maxFloat,&tempFloat,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		    		float avgMaxFloat=tempFloat/1595.0f;
		    		MPI_Allreduce(&minFloat,&tempFloat,1,MPI_FLOAT,MPI_MIN,MPI_COMM_WORLD);
		    		minFloat=tempFloat;
		    		MPI_Allreduce(&minFloat,&tempFloat,1,MPI_FLOAT,MPI_MIN,MPI_COMM_WORLD);
		    		float avgMinFloat=tempFloat/1595.0f;
		    		//maxFloat-=minFloat;
		    		MPI_Allreduce(&avgSpacing,&tempFloat,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		    		avgSpacing=tempFloat/1595.0f;
		    		exit(0);
		    	}
		    	if(staging==DOPREPROC) {
		    		MPI_Barrier(MPI_COMM_WORLD);
		    		MPI_Barrier(MPI_COMM_WORLD);
					ifstream inFile("indexesThatCanBeRemoved.dat",ios::in | ios::binary | ios::ate);
				    if (inFile.is_open()) {
					   	ULLI trem = inFile.tellg();
					   	vector<int> toRemoveIn(trem/sizeof(int),0);
					   	inFile.seekg(0,inFile.beg);
				       	inFile.read((char*)&toRemoveIn[0],trem*sizeof(int));
				       	for(auto t:toRemoveIn) {
				       		toRemove.insert(pair<int,UNCHAR>(t,0));
				       	}
				       	inFile.close();
				   	} else {
				   		cout << "Error opening index file\n";
				   		MPI_Finalize();
				   		return 0;
				   	}
		    	}
		    	--staging;
		    }
	    } else {
	    	high_resolution_clock::time_point startTime, endTime;
	    	totalNum=numberOfProcessors;
	    	int tempNum=0;
	    	vector<int> sourceCores;
	    	sourceCores.push_back(totalNum);
	    	for(int source = 1; source < num_nodes; source++) {
	    		MPI_Sendrecv(&placeHolder, 1, MPI_INT, source, TAG, &tempNum, 1, MPI_INT, source, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    		totalNum+=tempNum;
	    		sourceCores.push_back(tempNum);
	    	}
	    	int whichPatient=0, who=0;
	    	vector<int> toSend[sourceCores.size()];
	    	while(whichPatient<numPatients) {
	    		for(int i=0;i<sourceCores[who] && whichPatient<numPatients;++i) {
	    			toSend[who].push_back(whichPatient++);
	    		}
	    		who=(who+1)%sourceCores.size();
	    	}
	    	int sizeToSend;
	    	for(int i=1;i<sourceCores.size();++i) {
	    		sizeToSend=toSend[i].size();
				MPI_Sendrecv(&sizeToSend, 1, MPI_INT, i, TAG, &placeHolder, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Sendrecv(&toSend[i][0], sizeToSend, MPI_INT, i, TAG, &placeHolder, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    	}
	    	for(auto d:toSend[0]) {
	    		toDoList.push_back(d);
	    	}

	    	vector<pthread_t> threads;
			pthread_attr_t attr;
    		cpu_set_t cpus;
    		pthread_attr_init(&attr);

	    	int toDo=toDoList.size();
	    	if(toDo<numberOfProcessors) {
	    		numberOfProcessors=toDo;
	    	}
	    	for(int i=0;i<numberOfProcessors;++i) {
	    		threads.push_back(pthread_t());
	    	}
	    	int status;
			void * result;
			while(staging) {
				startTime=high_resolution_clock::now();
				whichPatient=0;
	    		for(int i=0;i<numberOfProcessors;++i) {
	    			threadsDone.push_back(true);
	    		}
				while(whichPatient<toDo) {
					for(int i=0;i<numberOfProcessors && whichPatient<toDo;++i) {
						pthread_mutex_lock(&crmutex);
						if(threadsDone[i]) {
							pthread_mutex_unlock(&crmutex);
							cout << hostname << " starting job: " << patients[toDoList[whichPatient]] << endl;

	     					CPU_ZERO(&cpus);
	       					CPU_SET(i, &cpus);
	       					pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);

		        			idLink2 *arg = (idLink2*)malloc(sizeof(*arg));
		        			(*arg).whichPatient=toDoList[whichPatient];
		        			++whichPatient;
		        			(*arg).my_id=i;
		        			threadsDone[i]=false;
		        			if(staging==REMOVEZEROS) {
		        				pthread_create(&threads[i], &attr, preprocThread2, arg);
		        			}
		        			if(staging==GETMINMAX) {
		        				pthread_create(&threads[i], &attr, preprocMaxMinThread, arg);
		        			}
		        			if(staging==DOPREPROC){
		        				pthread_create(&threads[i], &attr, preprocThread, arg);
		        			}
						} else {
							pthread_mutex_unlock(&crmutex);
						}
					}
				}
		    	for (int i=0; i < numberOfProcessors; ++i) {
		        	if ((status = pthread_join(threads[i], &result)) != 0) {
		            	fprintf (stderr, "join error %d: %s\n", status, strerror(status));
			        }
		    	}
		    	recvSuccess=vector<UNCHAR>(numPatients,0);
		    	MPI_Barrier(MPI_COMM_WORLD);
		    	MPI_Allreduce(&success[0],&recvSuccess[0],numPatients,MPI_UNSIGNED_CHAR,MPI_SUM,MPI_COMM_WORLD);
		    	cout << "Master done with all\n";
		    	endTime=high_resolution_clock::now();
		    	printTime(startTime,endTime);
		    	cout << endl;
		    	if(staging==GETMINMAX) {
		    		MPI_Barrier(MPI_COMM_WORLD);
		    		int tempIntt=0;
		    		MPI_Allreduce(&maxDim,&tempIntt,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
		    		maxDim=tempIntt;
		    		float tempFloat=0.0f;
		    		MPI_Allreduce(&maxFloat,&tempFloat,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
		    		maxFloat=tempFloat;
		    		MPI_Allreduce(&maxFloat,&tempFloat,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		    		float avgMaxFloat=tempFloat/1595.0f;
		    		MPI_Allreduce(&minFloat,&tempFloat,1,MPI_FLOAT,MPI_MIN,MPI_COMM_WORLD);
		    		minFloat=tempFloat;
		    		MPI_Allreduce(&minFloat,&tempFloat,1,MPI_FLOAT,MPI_MIN,MPI_COMM_WORLD);
		    		float avgMinFloat=tempFloat/1595.0f;
		    		maxFloat-=minFloat;
		    		MPI_Allreduce(&avgSpacing,&tempFloat,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
		    		avgSpacing=tempFloat/1595.0f;
		    		cout << "maxFloat: " << maxFloat << " minFloat: " << minFloat << " avgMaxFloat: " << avgMaxFloat << " avgMinFloat: " << avgMinFloat << " maxDim: " << maxDim << endl;
		    		cout << "avgSpacing: " << avgSpacing << endl;
		    		exit(0);
		    	}
		    	if(staging==DOPREPROC) {
		    		MPI_Barrier(MPI_COMM_WORLD);

					ChangeDir("KagglePre");
					vector<string> patients2;
					GetFilesInDirectory(patients2,types);
					int numPatients2=patients2.size();
					sort(patients2.begin(), patients2.end());
					ChangeDir("..");
					ULLI ffsize;

					ifstream iFile("KagglePre/"+patients2[0],ios::in | ios::binary | ios::ate);
					if (iFile.is_open()) {
						ffsize = iFile.tellg();
					  	iFile.close();
					} else {
						cout << "what?\n";
						return 0;
					}

					vector<UNCHAR> inVector2(ffsize);
					vector<UNCHAR> zeros2(ffsize,0);
					vector<int> outliers(ffsize,0);
					int count2=0;
					int index=0;
					for(int i=0;i<patients2.size();++i) {
						getFile(patients2[i],&inVector2[0]);
						for(int j=0;j<ffsize;++j) {
							if(inVector2[j]>ZERO_THRESHOLD) {
								outliers[j]++;
								zeros2[j]=1;
							}
						}
						count2=0;
						for(int vv=0;vv<ffsize;++vv) {
							if(!zeros2[vv]) {
								++count2;
							}
						}
						cout << "Zero count now: " << count2 << " file size can be reduced to: " << (ffsize-count2) << " -- filenumber: " << index << endl;
						++index;
					}
					for(int i=0;i<ffsize;++i) {
						if(outliers[i]<OUTLIER_MIN_COUNT) {
							toRemove.insert(pair<int,UNCHAR>(i,0));
						}
						if(!zeros2[i]) {
							toRemove.insert(pair<int,UNCHAR>(i,0));
						}
					}
					cout << "toRemove size: " << toRemove.size() << " number of indexes" << endl;
					int newSize=toRemove.size();
					cout << "new file size: " << (ffsize-newSize) << endl;

					vector<int> output;
					for(int i=0;i<ffsize;++i) {
						if(toRemove.find(i)!=toRemove.end()) {
							output.push_back(i);
						}
					}
					fstream oFile("indexesThatCanBeRemoved.dat",ios::out | ios::binary);
				   	if (oFile.is_open()) {
				       	oFile.write((char*)&output[0],newSize*sizeof(int));
				       	oFile.close();
				   	}

		    		MPI_Barrier(MPI_COMM_WORLD);

		    	}
		    	--staging;
		    }
	    	bool allGood=true;
	    	for(int i=0;i<numPatients;++i) {
	    		if(!recvSuccess[i]) {
	    			allGood=false;
	    			cout << "Patient# " << i << " failed.  Patient ID: " << patients[i] << endl;
	    		}
	    	}
	    	if(allGood) {
	    		cout << "All files preprocessed successfully\n";
	    	}
	    }

	    if(!andTrain) {
			MPI_Finalize();
			return 0;
		}
    }

	//*/
	srandom(time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());
	//srand((unsigned int)time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());

	if(!doingMNIST) {
		ChangeDir("..");
		ChangeDir("KagglePre2");
	} else {
		ChangeDir("mnist");
	}
	//ChangeDir("mnist");
	GetFilesInDirectory(patients,types);
	int numPatients=patients.size();
	//numPatients=1595;
	sort(patients.begin(), patients.end());
	ChangeDir("..");
	if(!doingMNIST) {
		ChangeDir("MPIversion");
	}

	string labelsFile;
	if(doingMNIST) {
		labelsFile="MNIST_labels.csv";
	} else {
		labelsFile="stage1_labels.csv";
	}
	ifstream lfile(labelsFile,ios::in|ios::binary);
	char temp[500];
	bool once=false;
	if(lfile.is_open()) {
		while(lfile.getline(temp,500)) {
			if(!once) {
				once=true;
			} else {
				string temp2=string(temp);
				vector<string> splitTemp=split(temp2,",");
				if(doingMNIST) {
					float t;
					sprintf(temp,"%s",splitTemp[1].c_str());
					sscanf(temp,"%f",&t);
					setLabeledPatients.push_back(pair<string,float>(splitTemp[0],t));
				} else {
					setLabeledPatients.push_back(pair<string,float>(splitTemp[0],(float)(splitTemp[1][0]-48)));
					//cout << hostname << ": " << splitTemp[0] << " " << ((float)splitTemp[1][0]-48.0f) << endl;
				}
			}
		}
		lfile.close();
	} else {
		cout << "what?\n";
	}
	if(!doingMNIST) {
		ChangeDir("..");
	}

	sort(setLabeledPatients.begin(),setLabeledPatients.end());
	int totalSize=setLabeledPatients.size();
	float tempPerc=((float)totalSize)*(percentTestSet/100.0f);
	int trainingSize=totalSize-(int)tempPerc;
	int testSize=totalSize-trainingSize;
	if(doingMNIST) {
		testSize=10000;
		trainingSize=60000;
		percentTestSet=1000000.0f/70000.0f;
	}

	if(!andTrain) {
		MPI_Init(&argc, &argv);
    	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
    }
    char my_host[100];
    gethostname(my_host, 100);
    hostname=string(my_host);//*/
    neuralNet NNmpi;

	if(eval) {
		if(!my_rank) {
			ChangeDir("MPIversion");
			if(inputFile=="") {
				cout << "Need a weights file in order to evaluate a neural network\n";
				exit(1);
			}

			NNmpi=neuralNet(inputFile);

			ChangeDir("..");
	    	ifstream inFileA("KagglePre2/"+patients[my_rank],ios::in | ios::binary | ios::ate);
		    if (inFileA.is_open()) {
			   	fsize = inFileA.tellg();
			   	inFileA.close();
			} else {
				cout << "couldn't find data\n";
				MPI_Finalize();
				exit(1);
			}
			inVector=vector<UNCHAR>(fsize);
			oneFile=vector<float>(fsize);

			NNmpi.evaluate();
		}
		MPI_Finalize();
		return 0;
	}

    if(inputFile!="") {
    	if(!doingMNIST) {
    		ChangeDir("MPIversion");
    	}
    	NNmpi=neuralNet(inputFile);
    	if(!doingMNIST) {
    		ChangeDir("..");
    	}
    }
    
    if(!my_rank) {

		struct sigaction ctrlc;
		ctrlc.sa_handler=ctrlchandler;
		ctrlc.sa_flags=0;
		sigemptyset(&ctrlc.sa_mask);
		sigaction(SIGTSTP,&ctrlc,NULL);//ctrl-z
		//sigaction(SIGINT,&ctrlc,NULL);//ctrl-c
		//sigaction(SIGQUIT,&ctrlc,NULL);//ctrl-\

		if(inputFile=="" || reshuffle) {
			if(!doingMNIST) {
		    	vector<pair<string, float>> tempVect;
		    	tempVect=setLabeledPatients;
		    	unordered_map<string, UNCHAR> tempHash;
				for(int i=0;i<testSize;++i) {
					int pick=(random()%tempVect.size());
					testSet.push_back(pair<string, int>(tempVect[pick].first,pick));
					testLabels.push_back(pair<float, int>(tempVect[pick].second,pick));
					if(((int)tempVect[pick].second)==0) {
						++zerosInTestSet;
					}
					tempHash.insert(pair<string,UNCHAR>(tempVect[pick].first,0));
					tempVect.erase(tempVect.begin()+pick);
				}
				for(int i=0;i<totalSize;++i) {
					auto search=tempHash.find(setLabeledPatients[i].first);
					if(search==tempHash.end()) {
						trainingSet.push_back(pair<string, int>(setLabeledPatients[i].first,i));
						trainingLabels.push_back(pair<float, int>(setLabeledPatients[i].second,i));
						if(((int)setLabeledPatients[i].second)==0) {
							++zerosInTrainSet;
						}
					}
				}
			} else {
				for(int i=0;i<10000;++i) {
					testSet.push_back(pair<string, int>(setLabeledPatients[i].first,i));
					testLabels.push_back(pair<float, int>(setLabeledPatients[i].second,i));
				}
				int index=10000;
				for(int i=0;i<60000;++i) {
					trainingSet.push_back(pair<string, int>(setLabeledPatients[index].first,index));
					trainingLabels.push_back(pair<float, int>(setLabeledPatients[index].second,index++));
				}
			}
		} else {
			percentTestSet=((float)testSet.size()/(float)totalSize)*100.0f;
			trainingSize=trainingSet.size();
			testSize=testSet.size();
			if(!doingMNIST) {
				for(auto t:testLabels) {
					if(((int)t.first)==0) {
						++zerosInTestSet;
					}
				}
				for(auto t:trainingLabels) {
					if(((int)t.first)==0) {
						++zerosInTrainSet;
					}
				}
			}
		}
    	cout << "Starting ANN...\n";
    	cout << "Percent of data to use as a test set: " << percentTestSet << "%" << endl;
    	cout << "Number of training set size: " << trainingSet.size() << endl;
    	cout << "Number of test set size: " << testSet.size() << endl;
    	cout << "Total patients with given labels: " << totalSize << endl;
    	cout << "Leaving " << (numPatients-totalSize) << " patients for evaluation and submission" << endl;
    }

    if(doPreproc) {
    	for(int i=0;i<numPatients;++i) {
	    	success.push_back(0);
	    }
	}

    int placeHolder=my_rank;
    if (my_rank != MASTER) {
    	MPI_Sendrecv(&numberOfProcessors, 1, MPI_INT, MASTER, TAG, &placeHolder, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	//used for recieving the total cpus on the job and which parts this computer is doing
    	int getOne;
    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, &getOne, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	int *doWhich=(int*)calloc(getOne,sizeof(int));
    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, doWhich, getOne, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	for(int i=0;i<getOne;++i) {
    		toTrainList.push_back(doWhich[i]);
    		//cout << hostname << " toTrain: " << toTrainList.back() << " who is: " << setLabeledPatients[toTrainList.back()].first << " labeled: " << setLabeledPatients[toTrainList.back()].second << endl;
    	}
    	free(doWhich);

    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, &getOne, 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	int *doWhich2=(int*)calloc(getOne,sizeof(int));
    	MPI_Sendrecv(&placeHolder, 1, MPI_INT, MASTER, TAG, doWhich2, getOne, MPI_INT, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	for(int i=0;i<getOne;++i) {
    		toTestList.push_back(doWhich2[i]);
    		//cout << hostname << " toTrain: " << toTestList.back() << " who is: " << setLabeledPatients[toTestList.back()].first << " labeled: " << setLabeledPatients[toTestList.back()].second << endl;
    	}
    	free(doWhich2);

    	toDo=toTrainList.size();
    	if(toDo) {

    		cout << hostname << " is in\n";

    		string tempName;
    		if(!doingMNIST) {
    			tempName="KagglePre2/"+patients[my_rank];
    		} else {
    			tempName="mnist/"+patients[my_rank];
    		}
	    	ifstream inFile(tempName,ios::in | ios::binary | ios::ate);
		    if (inFile.is_open()) {
			   	fsize = inFile.tellg();
			   	if(doingMNIST) {
			   		fsize/=sizeof(float);
			   	}
			   	inFile.close();
			} else {
				cout << "couldn't find data\n";
				MPI_Finalize();
				exit(1);
			}
			inVector=vector<UNCHAR>(fsize);
			oneFile=vector<float>(fsize);

			if(inputFile=="") {
				doMain(inputHiddenLayers, batchSize, doDataSetSize, lRate, inputFile, outFile, vlRate, false, RMSwant, testSize, trainingSize);
			} else {
				NNmpi.train_kaggle(100000000,lRate,vlRate,RMSwant,testSize,trainingSize);
			}
		}

    } else {//*/
    	high_resolution_clock::time_point startTime, endTime;
    	startTime=high_resolution_clock::now();
    	totalNum=numberOfProcessors;
    	int tempNum=0;
    	vector<int> sourceCores;
    	sourceCores.push_back(totalNum);
    	for(int source = 1; source < num_nodes; source++) {
    		MPI_Sendrecv(&placeHolder, 1, MPI_INT, source, TAG, &tempNum, 1, MPI_INT, source, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    		totalNum+=tempNum;
    		sourceCores.push_back(tempNum);
    	}//*/
    	cout << "Master recieved " << totalNum << " number of cores from cluster\n";
    	int whichPatient=0, who=0;
    	vector<int> toSendTrain[sourceCores.size()];
    	vector<int> toSendTest[sourceCores.size()];
    	while(whichPatient<trainingSize) {
    		for(int i=0;i<sourceCores[who] && whichPatient<trainingSize;++i) {
	   			toSendTrain[who].push_back(trainingSet[whichPatient++].second);
	   		}
    		who=(who+1)%sourceCores.size();
    	}
    	int sizeToSend;
    	for(int i=1;i<sourceCores.size();++i) {
    		sizeToSend=toSendTrain[i].size();
			MPI_Sendrecv(&sizeToSend, 1, MPI_INT, i, TAG, &placeHolder, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Sendrecv(&toSendTrain[i][0], sizeToSend, MPI_INT, i, TAG, &placeHolder, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	}//*/
    	whichPatient=0;
    	who=0;
    	while(whichPatient<testSize) {
    		for(int i=0;i<sourceCores[who] && whichPatient<testSize;++i) {
	   			toSendTest[who].push_back(testSet[whichPatient++].second);
	   		}
    		who=(who+1)%sourceCores.size();
    	}
    	for(int i=1;i<sourceCores.size();++i) {
    		sizeToSend=toSendTest[i].size();
			MPI_Sendrecv(&sizeToSend, 1, MPI_INT, i, TAG, &placeHolder, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Sendrecv(&toSendTest[i][0], sizeToSend, MPI_INT, i, TAG, &placeHolder, 1, MPI_INT, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	}//*/
    	for(auto d:toSendTrain[0]) {
    		toTrainList.push_back(d);
    		//cout << hostname << " toTrain: " << toTrainList.back() << " who is: " << setLabeledPatients[toTrainList.back()].first << " labeled: " << setLabeledPatients[toTrainList.back()].second << endl;
    	}
    	for(auto d:toSendTest[0]) {
    		toTestList.push_back(d);
    		//cout << hostname << " toTrain: " << toTestList.back() << " who is: " << setLabeledPatients[toTestList.back()].first << " labeled: " << setLabeledPatients[toTestList.back()].second << endl;
    	}

    	toDo=toTrainList.size();
    	if(toDo<numberOfProcessors) {
    		numberOfProcessors=toDo;
    	}

		string tempName;
		if(!doingMNIST) {
			tempName="KagglePre2/"+patients[my_rank];
		} else {
			tempName="mnist/"+patients[my_rank];
		}
    	ifstream inFile(tempName,ios::in | ios::binary | ios::ate);
	    if (inFile.is_open()) {
		   	fsize = inFile.tellg();
			if(doingMNIST) {
				fsize/=sizeof(float);
			}
		   	inFile.close();
		} else {
			cout << "couldn't find data\n";
			MPI_Finalize();
			exit(1);
		}
		inVector=vector<UNCHAR>(fsize);
		oneFile=vector<float>(fsize);

		cout << hostname << " is in\n";

		if(inputFile=="") {
			doMain(inputHiddenLayers, batchSize, doDataSetSize, lRate, inputFile, outFile, vlRate, false, RMSwant, testSize, trainingSize);
		} else {
			NNmpi.train_kaggle(100000000,lRate,vlRate,RMSwant,testSize,trainingSize);
		}

    	cout << "Master done with all\n";
    	endTime=high_resolution_clock::now();
    	printTime(startTime,endTime);
    	cout << endl;
    }

	MPI_Finalize();//*/

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
	float seconds=duration_cast<microseconds>(end-start).count()/1000000.0;
	cout << "Processing time (milliseconds): " << duration_cast<milliseconds>(end - start).count() << endl;
	cout << "Processing time (microseconds): " << duration_cast<microseconds>(end - start).count() << endl;
	cout << "Processing time (nanoseconds): " << duration_cast<nanoseconds>(end - start).count() << endl;
	printf("Processing time (seconds): %.04f\n",seconds);
	printf("Processing time (minutes): %.04f\n",seconds/60.0f);
	printf("Processing time (hours): %.04f\n",seconds/3600.0f);
}

void print_matrix(vector<float> &A, int nr_rows_A, int nr_cols_A) {
    for(int i = 0; i < nr_rows_A; ++i) {
        for(int j = 0; j < nr_cols_A; ++j) {
            //cout << A[j * nr_rows_A + i] << " ";
            float o=A[j*nr_rows_A+i];
            printf("%.4f ",o);
            //printf("%.10f ",A[j*nr_rows_A+i]);
        }
        cout << endl;
    }
}

void GetFilesInDirectory(vector<string> &out, vector<string> &types) {
	DIR *dir;
	class dirent *ent;
	class stat st;

	dir = opendir(CurrentPath);
	while ((ent = readdir(dir)) != NULL) {
		const string file_name = ent->d_name;
		//const string full_file_name = directory + "/" + file_name;
		const string full_file_name = file_name;

		if (file_name[0] == '.')
			continue;

		if (stat(full_file_name.c_str(), &st) == -1)
			continue;

		const bool is_directory = (st.st_mode & S_IFDIR) != 0;

		out.push_back(full_file_name);
		if (is_directory) {
			types.push_back("Directory\t");
		}
		else {
			types.push_back("File\t\t");
		}
	}
	closedir(dir);
}

bool ChangeDir(string dir) {
	if (dir == "..") {
		string temp = string(CurrentPath);
		if (temp.length()>1) {
			vector<string> folders = split(temp, slash);
			string temp2 = folders.at(0);
			temp2 += slash;
			for (int i = 1; i < (int)folders.size() - 1; i++) {
				temp2 += folders.at(i);
				temp2 += slash;
			}
			if(string(CurrentPath)!="/") {
				temp2.pop_back();
			}
			sprintf(CurrentPath, "%s", temp2.c_str());
			int test=chdir(CurrentPath);
			//SetCurrentDirectory(s2lps(string(gCurrentPath)));
		}
		return true;
	}
	else {
		if(dir.front()!='/') {
			vector<string> FileNames2;
			vector<string> DirTypes2;
			GetFilesInDirectory(FileNames2, DirTypes2);
			string temp = dir;
			bool found = false;
			for (auto f : FileNames2) {
				if (f == temp) {
					found = true;
				}
			}
			if (found) {
				temp = string(CurrentPath) + slash + dir + slash;
				sprintf(CurrentPath, "%s", temp.c_str());
				int test=chdir(CurrentPath);
				//SetCurrentDirectory(s2lps(string(gCurrentPath)));
				return true;
			}
			return false;
		}
		else {
			if(chdir(dir.c_str())==0) {
				sprintf(CurrentPath, "%s", dir.c_str());
				return true;
			}
			return false;
		}
	}
}

vector<string> split(string s, string any_of) {
	vector<string> seps;
	string temp = s;
	int foundat = temp.find(any_of);
	while (foundat != -1 && temp.length()>0) {
		seps.push_back(temp.substr(0, foundat));
		temp = temp.substr(foundat + 1, temp.length());
		foundat = temp.find(any_of);
	}
	if (temp.length()>0) {
		seps.push_back(temp);
	}
	return seps;
}

void getFile2(string filename, float *array) {
	if(doingMNIST) {
		filename="mnist/"+filename+".dat";
		ifstream inFile(filename,ios::in | ios::binary);
	   	if (inFile.is_open()) {
	       	inFile.read((char*)&array,fsize*sizeof(float));
	       	inFile.close();
	   	} else {
	   		cout << hostname << " couldn't find file: " << filename << endl;
	   	}
	} else {
		vector<UNCHAR> inVect(fsize);
		filename="KagglePre2/"+filename+".dat";
		ifstream inFile(filename,ios::in | ios::binary);
	   	if (inFile.is_open()) {
	       	inFile.read((char*)&inVect[0],fsize);
	       	inFile.close();
	   	} else {
	   		cout << hostname << " couldn't find file: " << filename << endl;
	   	}
	   	int index=0;
	   	bool something=false;
	   	for(auto i:inVect) {
	   		if(i) {
	   			something=true;
	   		}
	   		array[index++]=((float)i)/255.0f;
	   	}
	   	if(!something) {
	   		cout << "\n\nGot a file that's not reading correctly: " << filename << endl;
	   		exit(1);
	   	}
	}
}

void getFile(string filename) {
	if(doingMNIST) {
		filename="mnist/"+filename+".dat";
		ifstream inFile(filename,ios::in | ios::binary);
	   	if (inFile.is_open()) {
	       	inFile.read((char*)&oneFile[0],fsize*sizeof(float));
	       	inFile.close();
	   	} else {
	   		cout << hostname << " couldn't find file: " << filename << endl;
	   	}
	} else {
		filename="KagglePre2/"+filename+".dat";
		ifstream inFile(filename,ios::in | ios::binary);
	   	if (inFile.is_open()) {
	       	inFile.read((char*)&inVector[0],fsize);
	       	inFile.close();
	   	} else {
	   		cout << hostname << " couldn't find file: " << filename << endl;
	   	}
	   	int index=0;
	   	bool something=false;
	   	for(auto i:inVector) {
	   		if(i) {
	   			something=true;
	   		}
	   		oneFile[index++]=((float)i)/255.0f;
	   	}
	   	if(!something) {
	   		cout << "\n\nGot a file that's not reading correctly: " << filename << endl;
	   		exit(1);
	   	}
	}
}

int getFile(string filename, UNCHAR *input) {
	filename="/home/gvuser/MPIversion/KagglePre/"+filename;
	int size=0;
	ifstream inFile3(filename,ios::in | ios::binary | ios::ate);
	size=inFile3.tellg();
	inFile3.seekg(0, inFile3.beg);
   	if (inFile3.is_open()) {
       	inFile3.read((char*)input,size);
       	/*if(my_rank==1) {
       		for(int i=0;i<size;++i) {
	       		cout << input[i];
       		}
       		cout << endl;
       	}*/
       	inFile3.close();
   	} else {
   		cout << "couldn't find file\n";
   		return size;
   	}
   	return size;
}

void putZeroedFile(string filename, UNCHAR *input, int newSize) {
	filename="/home/gvuser/KagglePre2/"+filename;
	fstream inFile3(filename,ios::out | ios::binary);
	if(inFile3.is_open()) {
       	inFile3.write((char*)input,newSize);
      	inFile3.close();
	}
}

void *preprocThread2(void *thread_parm) {
	idLink2 data=*((idLink2*) thread_parm);
	int whichPatient=data.whichPatient;
	int my_id=data.my_id;

	int ssize=0;
	string filename=patients[whichPatient];
	ifstream inFile3("/home/gvuser/MPIversion/KagglePre/"+filename,ios::in | ios::binary | ios::ate);
   	if (inFile3.is_open()) {
		ssize=inFile3.tellg();
		inFile3.close();
	} else {
		cout << "Couldn't find input file\n";
		success[whichPatient]=0;
		pthread_exit(0);
	}
	vector<UNCHAR> newFile;
	/*int percDone=0;
	int magNumber=(int)size;
	if(!my_rank && !my_id) {
		cout << "percent done " << percDone << "%\r";
	}
	magNumber/=100;
	int index=0;*/
	vector<UNCHAR> input(ssize,0);
	getFile(filename,&input[0]);
	for(int i=0;i<ssize;++i) {
		/*if(!my_rank && !my_id) {
			++index;
			if(index>magNumber) {
				++percDone;
				index=0;
				cout << "percent done " << percDone << "%\r";
			}
		}*/
		if(toRemove.find(i)==toRemove.end()) {
			newFile.push_back(input[i]);
		}
		//input.erase(input.begin()+toRemove[i]);
	}
	putZeroedFile(filename, &newFile[0], newFile.size());

	pthread_mutex_lock(&crmutex);
	threadsDone[my_id]=true;
	pthread_mutex_unlock(&crmutex);
	success[whichPatient]=1;
	pthread_exit(0);
}

void *preprocMaxMinThread(void *thread_parm) {
	idLink2 data=*((idLink2*) thread_parm);
	int whichPatient=data.whichPatient;
	int my_id=data.my_id;

	pid_t pid;
	int p[2];
	int test=pipe(p);
	pid = fork();
	string fromOtherApp="";
	string wp=to_string(whichPatient);

	if (pid == 0) {
		cout << setprecision(10);
		int argSize=4;
		char *argv[argSize];
		vector<string> argvV;
		argvV.push_back("python3");
		argvV.push_back("getMaxDim.py");
		argvV.push_back(wp);
		for(int i=0;i<argSize-1;i++) {
			argv[i]=(char*)malloc(argvV.at(i).length()+1);
			strncpy(argv[i],argvV.at(i).c_str(),
					(unsigned long)argvV.at(i).length());
			argv[i][argvV.at(i).length()]='\0';
		}
		argv[argSize-1]=0;
		dup2(p[1], 1);
		close(p[0]);
		execvp(argv[0], argv);
		exit(EXIT_FAILURE);
	}
	else {
		close(p[1]);
		fd_set rfds;
		char buffer[10] = {0};
		pid_t waitres;
		int status;
		bool endPipe=false;
		while (!endPipe) {
			FD_ZERO(&rfds);
			FD_SET(p[0], &rfds);
			select(p[0] + 1, &rfds, NULL, NULL, NULL);
			if(FD_ISSET(p[0], &rfds)) {
				int ret = 0;
				while ((ret = read(p[0], buffer, 10)) > 0) {
					fromOtherApp+=string(buffer,ret);
					//write(1, buffer, ret);
					memset(buffer, 0, 10);
				}
			}
			waitres=waitpid(pid, &status, WNOHANG);
			if (waitres==pid) {
				endPipe=true;
			}
		}
		close(p[0]);
		vector<string> splitTemp=split(fromOtherApp,"maxDim=");
		float xFloat=0.0f;
		float mFloat=0.0f;
		float aSpacing=0.0f;
		int mDim=0;
		char temp[1000];
		sprintf(temp,"%s",splitTemp[1].c_str());
		sscanf(temp,"axDim=%d",&mDim);
		splitTemp=split(fromOtherApp,"maxFloat=");
		sprintf(temp,"%s",splitTemp[1].c_str());
		sscanf(temp,"axFloat=%f",&xFloat);
		splitTemp=split(fromOtherApp,"minFloat=");
		sprintf(temp,"%s",splitTemp[1].c_str());
		sscanf(temp,"inFloat=%f",&mFloat);
		splitTemp=split(fromOtherApp,"avgSpacing=");
		sprintf(temp,"%s",splitTemp[1].c_str());
		sscanf(temp,"vgSpacing=%f",&aSpacing);
		cout << hostname << " got output text from python script: '" << fromOtherApp << "' Completed: " << patients[whichPatient] << endl;//" -- mDim: " << mDim << " -- xFloat: " << xFloat << " -- mFloat: " << mFloat << " -- aSpacing: " << aSpacing << endl;

		pthread_mutex_lock(&preprocMaxMinMutex);
		if(xFloat>maxFloat) {maxFloat=xFloat;}
		if(mFloat<minFloat) {minFloat=mFloat;}
		if(mDim>maxDim) {maxDim=mDim;}
		avgSpacing+=aSpacing;
		pthread_mutex_unlock(&preprocMaxMinMutex);

		pthread_mutex_lock(&crmutex);
		threadsDone[my_id]=true;
		pthread_mutex_unlock(&crmutex);
		success[whichPatient]=1;
		//free((void*)&data);
		pthread_exit(0);
	}
}

void *preprocThread(void *thread_parm) {
	idLink2 data=*((idLink2*) thread_parm);
	int whichPatient=data.whichPatient;
	int my_id=data.my_id;

	pid_t pid;
	int p[2];
	int test=pipe(p);
	pid = fork();
	string fromOtherApp="";
	string wp=to_string(whichPatient);
	string mf=to_string(minFloat);
	string xf=to_string(maxFloat);
	string xd=to_string(maxDim);

	if (pid == 0) {
		int argSize=7;
		char *argv[argSize];
		vector<string> argvV;
		argvV.push_back("python3");
		argvV.push_back("preproc.py");
		argvV.push_back(wp);
		argvV.push_back(mf);
		argvV.push_back(xf);
		argvV.push_back(xd);
		for(int i=0;i<argSize-1;i++) {
			argv[i]=(char*)malloc(argvV.at(i).length()+1);
			strncpy(argv[i],argvV.at(i).c_str(),
					(unsigned long)argvV.at(i).length());
			argv[i][argvV.at(i).length()]='\0';
		}
		argv[argSize-1]=0;
		dup2(p[1], 1);
		close(p[0]);
		execvp(argv[0], argv);
		exit(EXIT_FAILURE);
	}
	else {
		close(p[1]);
		fd_set rfds;
		char buffer[10] = {0};
		pid_t waitres;
		int status;
		bool endPipe=false;
		while (!endPipe) {
			FD_ZERO(&rfds);
			FD_SET(p[0], &rfds);
			select(p[0] + 1, &rfds, NULL, NULL, NULL);
			if(FD_ISSET(p[0], &rfds)) {
				int ret = 0;
				while ((ret = read(p[0], buffer, 10)) > 0) {
					fromOtherApp+=string(buffer,ret);
					//write(1, buffer, ret);
					memset(buffer, 0, 10);
				}
			}
			waitres=waitpid(pid, &status, WNOHANG);
			if (waitres==pid) {
				endPipe=true;
			}
		}
		close(p[0]);
		//cout << hostname << " got output text from python script: '" << fromOtherApp << "' Completed: " << patients[whichPatient] << endl;
		pthread_mutex_lock(&crmutex);
		threadsDone[my_id]=true;
		pthread_mutex_unlock(&crmutex);
		success[whichPatient]=1;
		//free((void*)&data);
		pthread_exit(0);
	}
}
