//ANN proof of concept Cuda


//Ron Patrick - Capstone GVSU - Winter 2017

//MNIST here http://yann.lecun.com/exdb/mnist/

//If you don't want to install OpenMPI and Cuda toolkit 8.0,
//simply compile with g++ pConcept.cpp -o pConcept -std=c++11
//since the Makefile only works on my machine with Cuda and OpenMPI
//Also you'll need to comment out all 'thrust' header includes
//as well as <mpi.h>, and <imebra/imebra.h> both these require
//separate libraries in order for the program to run correctly.

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
struct doRandomFloats {
	__device__ float operator()(ULLI t) {
		thrust::default_random_engine defRandEngine;
		thrust::uniform_real_distribution<float> uniRealDist;
		defRandEngine.discard(t);
		return (uniRealDist(defRandEngine)*2.0f)-1.0f;
	}
};

//index into a connWeights layer is fromNode*NumberOfNodesInNextLayer+toNode
// for example 3 nodes in first layer and 5 nodes in second layer
//	0 to 0	0
// 	0 to 1	1
//	0 to 2	2
//	0 to 3	3
//	0 to 4	4
//	1 to 0	5
//	1 to 1	6
//	1 to 2	7
//	1 to 3	8
//	1 to 4	9
//	2 to 0	10
//	2 to 1	11
//	2 to 2	12
//	2 to 3	13
//	2 to 4	14
struct forwardProp_functor : public thrust::unary_function<int, void> {
	double *connWeights;
	double *netPrev;
	double *netThis;
	double *lDeriv;
	int sizePrev;
	int sizeThis;
	forwardProp_functor(double *_lDeriv, double *_connWeights, double *_netPrev, double *_netThis, int _sizePrev, int _sizeThis) : 
		lDeriv(_lDeriv), connWeights(_connWeights), netPrev(_netPrev), netThis(_netThis), sizePrev(_sizePrev), sizeThis(_sizeThis) {}
	__device__ void operator()(int t) {
		double local=0.0;
		for(int i=0;i<sizePrev;++i) {
			local+=(connWeights[i*sizeThis+t]*netPrev[i]);
		}
		local=1.0 / (1.0 + exp(-local));
		netThis[t]=local;
		lDeriv[t]=local*(1.0-local);
	}
};
struct backwardProp_functor : public thrust::unary_function<int, void> {
	double *connWeights;
	double *netErrThis;
	double *netErrNext;
	double *netThis;
	double *lDeriv;
	int sizeNext;
	int sizeThis;
	double learningRate;
	backwardProp_functor(double *_lDeriv, double *_connWeights, double *_netErrNext, double *_netErrThis, double *_netThis, int _sizeNext, int _sizeThis, double lRate) : 
		lDeriv(_lDeriv), connWeights(_connWeights), netErrNext(_netErrNext), netErrThis(_netErrThis), netThis(_netThis), sizeNext(_sizeNext), sizeThis(_sizeThis),
		learningRate(lRate) {}
	__device__ void operator()(int t) {
		double local=0.0;
		double local2=netThis[t];
		double local3;
		int target=t*sizeNext;
		for(int i=0;i<sizeNext;++i) {
			local3=netErrNext[i];
			local+=(connWeights[target]*local3);
			connWeights[target]-=(learningRate*local3*local2);
			++target;
		}
		netErrThis[t]=local*lDeriv[t];
	}
};
struct forwardProp_functor_float : public thrust::unary_function<int, void> {
	float *connWeights;
	float *netPrev;
	float *netThis;
	float *lDeriv;
	int sizePrev;
	int sizeThis;
	forwardProp_functor_float(float *_lDeriv, float *_connWeights, float *_netPrev, float *_netThis, int _sizePrev, int _sizeThis) : 
		lDeriv(_lDeriv), connWeights(_connWeights), netPrev(_netPrev), netThis(_netThis), sizePrev(_sizePrev), sizeThis(_sizeThis) {}
	__device__ void operator()(int t) {
		float local=0.0f;
		for(int i=0;i<sizePrev;++i) {
			//printf("t: %d (i*sizeThis+t): %d i: %d, sizePrev: %d, sizeThis: %d\n",t,i*sizeThis+t,i,sizePrev,sizeThis);
			local+=(connWeights[i*sizeThis+t]*netPrev[i]);
		}
		local=1.0f/(1.0f+exp(-local));
		netThis[t]=local;
		lDeriv[t]=local*(1.0f-local);
	}
};
struct backwardProp_functor_float : public thrust::unary_function<int, void> {
	float *connWeights;
	float *netErrThis;
	float *netErrNext;
	float *netThis;
	float *lDeriv;
	int sizeNext;
	int sizeThis;
	float learningRate;
	backwardProp_functor_float(float *_lDeriv, float *_connWeights, float *_netErrNext, float *_netErrThis, float *_netThis, int _sizeNext, int _sizeThis, float lRate) : 
		lDeriv(_lDeriv), connWeights(_connWeights), netErrNext(_netErrNext), netErrThis(_netErrThis), netThis(_netThis), sizeNext(_sizeNext), sizeThis(_sizeThis),
		learningRate(lRate) {}
	__device__ void operator()(int t) {
		//netErrThis[t]=0.0;
		float local=0.0f;
		int target=t*sizeNext;
		float local2=netThis[t];
		float local3;
		for(int i=0;i<sizeNext;++i) {
			local3=netErrNext[i];
			local+=(connWeights[target]*local3);
			connWeights[target]-=(learningRate*local3*local2);
			++target;
		}
		netErrThis[t]=local*lDeriv[t];
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
	neuralNet(int _numInputs, int _numOutputs, vector<int> &_hiddenMatrix) : hiddenMatrix(_hiddenMatrix), RMS(DBL_MAX), minRMS(DBL_MAX), epoch(0), learningRate(0.99) {
		numInputs=_numInputs;
		numOutputs=_numOutputs;
		hiddenMatrix.insert(hiddenMatrix.begin(),numInputs);
		hiddenMatrix.push_back(numOutputs);
		RMS_f=FLT_MAX;
		minRMS_f=FLT_MAX;
		learningRate_f=0.99f;
	}

	void train(vector<vector<float>> &pData, vector<vector<double>> &pLabels, ULLI maxIter, float RMSwant, vector<UNCHAR> &bLabels) {

		cout << "Setting up network...\n";
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;

		dataSetSize=pData.size();
		//dataSetSize=1000;
		cout << "dataSetSize: " << dataSetSize << endl;
		cout << "size of item: " << pData[0].size() << endl;
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
		ULLI maxElement=*std::max_element(hiddenMatrix.begin(),hiddenMatrix.end());
		device_vector<double> net[layers];
		device_vector<double> netLocalDerivatives[layers];
		device_vector<double> netErrorSignals[layers];
		//device_vector<double> connWeights[layers-1][maxElement];
		device_vector<double> connWeights[layers-1];
		//device_vector<device_vector<double>> connWeights[layers-1];;

		int index=0;
		for(auto h:hiddenMatrix) {
			net[index]=device_vector<double>(h);
			netLocalDerivatives[index]=device_vector<double>(h);
			netErrorSignals[index++]=device_vector<double>(h);
		}

		for(ULLI i=0;i<layers-1;++i) {
			int lSize=hiddenMatrix[i]*hiddenMatrix[i+1];
			connWeights[i]=device_vector<double>(lSize);
			thrust::transform(thrust::make_counting_iterator<ULLI>(0),thrust::make_counting_iterator<ULLI>(lSize),connWeights[i].begin(),doRandomDoubles());
		}

		cout << "Starting training...\n";

		thrust::device_vector<double>::iterator iter;
		UNCHAR position;
		int gotRight=0;
		for(int ii=0;ii<maxEpochs && RMSwanted<RMS && gotRight!=dataSetSize;++ii) {
			RMS=0.0;
			gotRight=0;
			for(int iii=0;iii<dataSetSize;++iii) {
				printf("Item# %d\r",iii);

				//forward
				thrust::transform(data[iii].begin(),data[iii].end(),net[0].begin(),floatToDoubleFunctor());
				for(int i=1;i<layers;++i) {

					thrust::for_each(thrust::make_counting_iterator<int>(0),
						thrust::make_counting_iterator<int>(hiddenMatrix[i]),
						forwardProp_functor(netLocalDerivatives[i].data().get(),
							connWeights[i-1].data().get(),
							net[i-1].data().get(),
							net[i].data().get(),
							hiddenMatrix[i-1],
							hiddenMatrix[i]));
				}
				iter = thrust::max_element(net[outputsIndex].begin(), net[outputsIndex].end());
				position = iter - net[outputsIndex].begin();
				if(position==bLabels[iii]) {
					++gotRight;
				}

				//backProp
				thrust::transform(net[outputsIndex].begin(),net[outputsIndex].end(),labels[iii].begin(),netErrorSignals[outputsIndex].begin(),thrust::minus<double>());

				RMS=thrust::transform_reduce(netErrorSignals[outputsIndex].begin(),netErrorSignals[outputsIndex].end(),square<double>(),RMS,thrust::plus<double>());
				for(int i=layers-2;i>-1;--i) {

					thrust::for_each(thrust::make_counting_iterator<int>(0),
						thrust::make_counting_iterator<int>(hiddenMatrix[i]),
						backwardProp_functor(netLocalDerivatives[i].data().get(),
							connWeights[i].data().get(),
							netErrorSignals[i+1].data().get(),
							netErrorSignals[i].data().get(),
							net[i].data().get(),
							hiddenMatrix[i+1],
							hiddenMatrix[i],
							learningRate));
				}
			}
			RMS=sqrt(RMS/toDivideRMS);
			if(RMS<minRMS) {
				minRMS=RMS;
				//printf("minRMS Error: %.15f Iteration: %d of %llu -- Got %d of %d correct\n",minRMS,ii,maxEpochs,gotRight,dataSetSize);
			}

			printf("\ncurrent RMS: %.15f minRMS Error: %.15f Iteration: %d of %llu -- Got %d of %d correct\n",RMS,minRMS,ii,maxEpochs,gotRight,dataSetSize);
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
			/*cout << "Inputs: ";
			for(auto i:net[0]) {
				double temp=i;
				printf("%.15f ",temp);
			}//*/
			for(int i=1;i<layers;++i) {
				thrust::for_each(thrust::make_counting_iterator<int>(0),
					thrust::make_counting_iterator<int>(hiddenMatrix[i]),
					forwardProp_functor(netLocalDerivatives[i].data().get(),
						connWeights[i-1].data().get(),
						net[i-1].data().get(),
						net[i].data().get(),
						hiddenMatrix[i-1],
						hiddenMatrix[i]));
			}
			iter = thrust::max_element(net[outputsIndex].begin(), net[outputsIndex].end());
			position = iter - net[outputsIndex].begin();
			cout << "Item# " << ii << " predicted: " << (int)position << " expected: " << (int)bLabels[ii] << endl;

			/*cout << endl << "Output: ";
			for(auto o:net[outputsIndex]) {
				double temp=o;
				printf("%.15f ",temp);
			}
			cout << endl;*/
		}
	}

	void train_floats(vector<vector<float>> &pData, vector<vector<float>> &pLabels, ULLI maxIter, float RMSwant, vector<UNCHAR> &bLabels) {

		cout << "Setting up network...\n";
		RMSwanted_f=RMSwant;
		maxEpochs=maxIter;
		layers=hiddenMatrix.size();
		outputsIndex=layers-1;

		dataSetSize=pData.size();
		dataSetSize=1000;
		cout << "dataSetSize: " << dataSetSize << endl;
		cout << "size of item: " << pData[0].size() << endl;
		toDivideRMS_f=((float)dataSetSize)*(float)numOutputs;
		device_vector<float> data[dataSetSize];
		device_vector<float> labels[dataSetSize];
		float *temp;
		float *tempd;
		ULLI len=pData[0].size();
		ULLI llen=pLabels[0].size();
		for(int i=0;i<dataSetSize;++i) {
			temp=&pData[i][0];
			data[i]=device_vector<float>(temp, temp+len);
			tempd=&pLabels[i][0];
			labels[i]=device_vector<float>(tempd, tempd+llen);
		}
		ULLI maxElement=*std::max_element(hiddenMatrix.begin(),hiddenMatrix.end());
		device_vector<float> net[layers];
		device_vector<float> netLocalDerivatives[layers];
		device_vector<float> netErrorSignals[layers];
		device_vector<float> connWeights[layers-1];

		int index=0;
		for(auto h:hiddenMatrix) {
			net[index]=device_vector<float>(h);
			netLocalDerivatives[index]=device_vector<float>(h);
			netErrorSignals[index++]=device_vector<float>(h);
		}

		for(ULLI i=0;i<layers-1;++i) {
			int lSize=hiddenMatrix[i]*hiddenMatrix[i+1];
			connWeights[i]=device_vector<float>(lSize);
			thrust::transform(thrust::make_counting_iterator<ULLI>(0),thrust::make_counting_iterator<ULLI>(lSize),connWeights[i].begin(),doRandomFloats());
		}

		cout << "Starting training...\n";

		thrust::device_vector<float>::iterator iter;
		UNCHAR position;
		int gotRight=0;
		for(int ii=0;ii<maxEpochs && RMSwanted_f<RMS_f && gotRight!=dataSetSize;++ii) {
			RMS_f=0.0;
			gotRight=0;
			for(int iii=0;iii<dataSetSize;++iii) {
				//printf("Item# %d\r",iii);

				//forward
				thrust::copy(data[iii].begin(),data[iii].end(),net[0].begin());
				for(int i=1;i<layers;++i) {
					thrust::for_each(thrust::make_counting_iterator<int>(0),
						thrust::make_counting_iterator<int>(hiddenMatrix[i]),
						forwardProp_functor_float(netLocalDerivatives[i].data().get(),
							connWeights[i-1].data().get(),
							net[i-1].data().get(),
							net[i].data().get(),
							hiddenMatrix[i-1],
							hiddenMatrix[i]));
				}
				iter = thrust::max_element(net[outputsIndex].begin(), net[outputsIndex].end());
				position = iter - net[outputsIndex].begin();
				if(position==bLabels[iii]) {
					++gotRight;
				}

				//backProp
				thrust::transform(net[outputsIndex].begin(),net[outputsIndex].end(),labels[iii].begin(),netErrorSignals[outputsIndex].begin(),thrust::minus<float>());

				RMS_f=thrust::transform_reduce(netErrorSignals[outputsIndex].begin(),netErrorSignals[outputsIndex].end(),square<float>(),RMS_f,thrust::plus<float>());
				for(int i=layers-2;i>-1;--i) {
					thrust::for_each(thrust::make_counting_iterator<int>(0),
						thrust::make_counting_iterator<int>(hiddenMatrix[i]),
						backwardProp_functor_float(netLocalDerivatives[i].data().get(),
							connWeights[i].data().get(),
							netErrorSignals[i+1].data().get(),
							netErrorSignals[i].data().get(),
							net[i].data().get(),
							hiddenMatrix[i+1],
							hiddenMatrix[i],
							learningRate));
				}
			}
			RMS_f=sqrt(RMS_f/toDivideRMS_f);
			if(RMS_f<minRMS_f) {
				minRMS_f=RMS_f;
				//printf("\nminRMS Error: %.15f Iteration: %d",RMS,ii);
			}

			printf("\ncurrent RMS: %.15f minRMS Error: %.15f Iteration: %d of %llu -- Got %d of %d correct\n",RMS_f,minRMS_f,ii,maxEpochs,gotRight,dataSetSize);
			/*cout << endl << "Output: ";
			for(auto o:net[outputsIndex]) {
				float temp=o;
				printf("%.5f ",temp);
			}
			cout << endl;*/
		}
		cout << endl;
		for(int ii=0;ii<dataSetSize;++ii) {
			thrust::copy(data[ii].begin(),data[ii].end(),net[0].begin());
			/*cout << "Inputs: ";
			for(auto i:net[0]) {
				float temp=i;
				printf("%.15f ",temp);
			}//*/
			for(int i=1;i<layers;++i) {
				thrust::for_each(thrust::make_counting_iterator<int>(0),
					thrust::make_counting_iterator<int>(hiddenMatrix[i]),
					forwardProp_functor_float(netLocalDerivatives[i].data().get(),
						connWeights[i-1].data().get(),
						net[i-1].data().get(),
						net[i].data().get(),
						hiddenMatrix[i-1],
						hiddenMatrix[i]));
			}
			iter = thrust::max_element(net[outputsIndex].begin(), net[outputsIndex].end());
			position = iter - net[outputsIndex].begin();
			cout << "Item# " << ii << " predicted: " << (int)position << " expected: " << (int)bLabels[ii] << endl;

			/*cout << endl << "Output: ";
			for(auto o:net[outputsIndex]) {
				float temp=o;
				printf("%.15f ",temp);
			}
			cout << endl;*/
		}

	}		

private:
	ULLI epoch, maxElement, layers, maxEpochs;
	int outputsIndex, dataSetSize, numInputs, numOutputs;
	double RMS, minRMS, toDivideRMS, RMSwanted, learningRate;
	float RMS_f, minRMS_f, toDivideRMS_f, RMSwanted_f, learningRate_f;
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

	/*vector<int> hiddenMatrix;
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
	test.train(countingTest,countingLabels,1000000,0.00001);*/

	vector<int> hiddenMatrix;
	hiddenMatrix.push_back(128);
	//hiddenMatrix.push_back(784+(784/2));
	//hiddenMatrix.push_back(784+(784/2));
	//hiddenMatrix.push_back(784);
	//hiddenMatrix.push_back(784);

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

	neuralNet go(784,10,hiddenMatrix);
	//go.train_floats(trainData,trainLabels,1000000,0.0001,trainLabels2);
	go.train(trainData,trainLabels,1000000,0.0001,trainLabels2);//*/
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
			//cout << "Total Cuda Memory: " << totalCudaMem << endl;
		}
	}

	doMain(my_rank, hostname, num_nodes);

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
                    //cout << "from arr: " << arr[i].back() << endl;
                }
            }
        }
    }
    file.close();
}