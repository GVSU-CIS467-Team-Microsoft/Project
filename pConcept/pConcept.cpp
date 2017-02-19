//ANN proof of concept

//Ron Patrick
//credit for help from these sources
//credit to http://rimstar.org/science_electronics_projects/backpropagation_neural_network_software_3_layer.htm
//credit to https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
//credit to J. R. Chen and P. Mars, "Stepsize Variation Methods for Accelerating the Back-Propagation Algorithm", IJCNN-90-WASH-DC volume 1, pp 601-604, Lawrence Erlbaum, 1990.

//The intarray2bmp.hpp was simply used to test if I was reading the training data
//correctly or not  (MNIST number recognition training and test set)
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
//#include <cudnn.h>
using namespace std;
using namespace thrust;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define ULLI unsigned long long int
#define UNCHAR unsigned char
//#define MAX 25
#ifndef BITS
#define BITS 4
#endif

// A separate struct in case this needs to be
// more complex in the future
//(Actually I found that this needs to be less complex in order
//to be easier to work with in Cuda. All these arrays should be separated)
struct neuron_t {
	double output=0.0;
	double input=0.0;
	double biasWeight=1.0;
	double beta=0.0;
	double prevBiasDelta=0.0;
	double errorSignal=0.0;
	double localDerivative=0.0;
};


void ReadMNIST_double(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr);
void ReadMNIST_UNCHAR(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<UNCHAR>> &arr);
void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<float> &arr);
void ReadMNIST_neuron_t(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<neuron_t>> &arr);

struct connectionWeights_t {
	double weight;
	double prevDelta=0.0;
	//int from=-1;
	//int to=-1;
};

class neuralNet {
public:
	neuralNet(int in, int out, vector<int> &sizeHiddenMatrix, bool Cuda) : Momentum(0.9), StepSize(0.25), divergingCheck(0), RMS(0.0), epoch(0) {
		if(!Cuda) {
			net=vector<vector<neuron_t>>(sizeHiddenMatrix.size()+2);
			net[0]=vector<neuron_t>(in);
			for(size_t i=0;i<sizeHiddenMatrix.size();++i) {
				net[i+1]=vector<neuron_t>(sizeHiddenMatrix[i]);
			}
			net[sizeHiddenMatrix.size()+1]=vector<neuron_t>(out);

			//Make this # larger for a wider variety of random numbers between -1.0 and 1.0
			ULLI randomPrecision=100000;

			connectionWeights=vector<vector<connectionWeights_t>>(sizeHiddenMatrix.size()+1);
			for(size_t i=0;i<net.size()-1;++i) {
				connectionWeights[i]=vector<connectionWeights_t>(net[i].size()*net[i+1].size());
				for(size_t j=0;j<connectionWeights[i].size();++j) {
					connectionWeights[i][j].weight=doRandom(randomPrecision);
				}
				for(size_t j=0;j<net[i].size();++j) {
					net[i][j].biasWeight=doRandom(randomPrecision);
				}
			}
			outputsIndex=net.size()-1;
			numOutputs_int=net[net.size()-1].size();
			for(size_t i=0;i<numOutputs_int;++i) {
				net[outputsIndex][i].biasWeight=doRandom(randomPrecision);
			}
			StepSizeAcc=0.1*StepSize;
			numOutputs=(double)out;
			//Setup from and to indexes.  Just for troubleshooting. May not be needed in the future
			/*for(size_t i=1;i<net.size();++i) {
				for(size_t j=0;j<net[i].size();++j) {
					ULLI JxN=j*net[i-1].size();
					for(size_t k=0;k<net[i-1].size();++k) {
						connectionWeights[i-1][JxN+k].from=k;
						connectionWeights[i-1][JxN+k].to=j;
					}
				}
			}*/
		} else {

		}
	};
	~neuralNet(){};
	void loadWeights(string filename);
	void saveWeights(string filename);

	void train_cuda(vector<float> &data, vector<float> &labels, float desiredError, ULLI max_cycles, ULLI pItem_size, ULLI setSize) {
		//LastRMS_cuda=99.9f;
		//ULLI outerEpoch=0;
		//minRMS_cuda=1000000.0f;
		//toDivideRMS_cuda=((float)setSize)*numOutputs_cuda;
		dataSetSize=setSize;
		item_size=pItem_size;
		label_size=numOutputs_int;
		//RMS_cuda=0.0f;

		float *temp=&data[0];
		ULLI len=data.size();
		//data_cuda=device_vector<float>(temp, temp+len);

		float *temp2=&labels[0];
		ULLI len2=labels.size();
		//labels_cuda=device_vector<float>(temp2,temp2+len2);
		
		//printOutputsForSet_cuda(data_cuda,labels_cuda);
		//cout << "Training done: Cycles: " << outerEpoch << " Error: " << LastRMS_cuda << " minRMS: " << minRMS_cuda << "\n";
	}

	void train(vector<vector<neuron_t>> &data, vector<vector<double>> &labels, double desiredError, ULLI max_cycles) {
		LastRMS=99.9;
		cost=1.0;
		ULLI outerEpoch=0;
		minRMS=1000000.0;
		toDivideRMS=((double)data.size())*numOutputs;
		dataSetSize=data.size();
		//int which;

		cout << "Started training..." << endl;
		//bool RMS_not_moving=false;
		//while(LastRMS > desiredError && outerEpoch<max_cycles && !RMS_not_moving) {
		while(LastRMS > desiredError && outerEpoch<max_cycles) {
			for(size_t i=0;i<dataSetSize;++i) {
				/*which=random()%dataSetSize;
				forwardProp(data[which],labels[which]);
				backwardProp(data[which],labels[which]);//*/
				forwardProp(data[i]);
				//printOutputsForSet(data,labels);
				backwardProp(labels[i]);
				//printNetworkState();
			}
			++epoch;
			LastRMS=sqrt(RMS/toDivideRMS);
			printf("Epoch: %llu\t Last RMS Error: %.15f\t min RMS: %.15f\r",epoch,LastRMS,minRMS);
			RMS=0.0;
			if(epoch>1) {
				if(PrevRMSError<LastRMS) {
					//cout << "Diverging\n";
					divergingCheck=0;
					StepSize*=0.95;
					StepSizeAcc=0.1*StepSize;
				} else if(PrevRMSError>LastRMS) {
					//cout << "Converging\n";
					++divergingCheck;
					if(divergingCheck==5) {
						StepSize+=0.04;
						StepSizeAcc=0.1*StepSize;
						divergingCheck=0;
					}
				} else {
					divergingCheck=0;
				}
			}//*/
			if(LastRMS<minRMS) {
				minRMS=LastRMS;
				printf("Epoch: %llu\t Last RMS Error: %.15f\t min RMS: %.15f\r",epoch,LastRMS,minRMS);
				//printOutputsForSet(data,labels);
			}
			PrevRMSError=LastRMS;
			++outerEpoch;
		}
		printOutputsForSet(data,labels);
		printf("Training done: Cycles: %llu Error: %.15f\n",outerEpoch,minRMS);
		//cout << "Training done: Cycles: " << outerEpoch << " Error: " << LastRMS << " minRMS: " << minRMS << "\n";
	}

	/*void printOutputsForSet_cuda(device_vector<float> &data, device_vector<float> &labels) {
		cout << endl;
		for(size_t i=0;i<dataSetSize;++i) {
			forwardProp_cuda(i);
			//cout << "Inputs: ";
			//for(auto n:net[0]) {
			//	cout << n.output << " ";
			//}
			//cout << endl;
			cout << "Output:\t\t";
			for(size_t j=0;j<numOutputs_int;++j) {
				//printf("%.6f ",net_cuda[outputsIndex+j]);
				cout << net_cuda[outputsIndex+j] << " ";
			}
			cout << endl;
			cout << "Expected:\t";
			for(size_t j=0;j<numOutputs_int;++j) {
				//printf("%.6f ",labels[i*label_size+j]);
				cout << labels_cuda[i*label_size+j] << " ";
			}
			cout << endl;
		}
		printf("Epoch: %llu\t Last RMS Error: %.15f\t min RMS: %.15f\n",epoch,LastRMS_cuda,minRMS_cuda);
		cout << "-------------------------------------\n";
	}*/

	void printOutputsForSet(vector<vector<neuron_t>> &data, vector<vector<double>> &labels) {
		cout << endl;
		for(size_t i=0;i<dataSetSize;++i) {
			forwardProp(data[i]);
			cout << "Inputs: ";
			for(auto n:net[0]) {
				cout << n.output << " ";
			}
			cout << endl;//*/
			cout << "Output:\t\t";
			for(size_t j=0;j<numOutputs_int;++j) {
				printf("%.10f ",net[outputsIndex][j].output);
				//cout << net[outputsIndex][j].output << " ";
			}
			cout << endl;
			cout << "Expected:\t";
			for(size_t j=0;j<numOutputs_int;++j) {
				printf("%.10f ",labels[i][j]);
				//cout << labels[i][j] << " ";
			}
			cout << endl;
		}
		printf("Epoch: %llu\t Last RMS Error: %.15f\t min RMS: %.15f\n",epoch,LastRMS,minRMS);
		cout << "-------------------------------------\n";
	}

	void printNetworkState() {
		int layer=0;
		cout << "-------------------------------------\n";
		cout << "Neuron layer 0 is the input layer\n";
		for(auto n:net) {
			cout << "Layer: " << layer++ << endl;
			int neuron=0;
			for(auto nn:n) {
				cout << "Neuron: " << neuron++ << " value: " << nn.output << " biasWeight: " << nn.biasWeight << " prevBiasDelta: " << nn.prevBiasDelta << " beta: " << nn.beta << endl;
			}
			cout << endl;
		}
		layer=0;
		for(auto c:connectionWeights) {
			cout << "Connection weights from layer: " << layer++ << endl;
			for(auto cc:c) {
				//cout << "From: " << cc.from << " To: " << cc.to << " weight: " << cc.weight << " prevDelta: " << cc.prevDelta << endl;
			}
		}
		cout << "-------------------------------------\n";
		sleep(1);
	}

private:
	vector<vector<neuron_t>> net;
	vector<vector<connectionWeights_t>> connectionWeights;

	ULLI item_size;
	ULLI label_size;

	double RMS, LastRMS, StepSize, StepSizeAcc, Momentum, PrevRMSError;
	double numOutputs, minRMS, toDivideRMS, cost;

	ULLI divergingCheck, epoch, outputsIndex, numOutputs_int;
	ULLI dataSetSize;
	ULLI layers;
	ULLI totalConnectionWeights;
	ULLI totalNeurons;

	//float numOutputs_cuda, minRMS_cuda, toDivideRMS_cuda;
	//float RMS_cuda, LastRMS_cuda, StepSize_cuda, StepSizeAcc_cuda, Momentum_cuda, PrevRMSError_cuda;

	__host__ __device__ double sigmoid(const double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	__host__ __device__ double sigmoid_derivative(const double x) {
		return x*(1.0-x);
	}
	double tanH_derivative(const double x) {
		double th = tanh(x); // tanh(x) \in (-1,1); cosh(x) \in (1,inf)
		return 1.0 - th*th; // sech^2(x) = 1 - tanh^2(x)
	}

	void forwardProp(vector<neuron_t> &item) {
		net[0]=item;
		for(size_t i=1;i<net.size();++i) {
			for(size_t j=0;j<net[i].size();++j) {
				net[i][j].output=0.0;
				net[i][j].beta=0.0;
				ULLI JxN=j*net[i-1].size();
				for(size_t k=0;k<net[i-1].size();++k) {
					net[i][j].output+=net[i-1][k].output*connectionWeights[i-1][JxN+k].weight;
				}
				net[i][j].output+=net[i][j].biasWeight;
				net[i][j].output=sigmoid(net[i][j].output);
			}
		}
	}

	void backwardProp(vector<double> &label) {
		double deltaweight, tempBeta, tempActivation;
		ULLI lastHiddenSize=net[net.size()-2].size();
		ULLI lastHiddenIndex=net.size()-2;
		ULLI connectionWeightsIndex=connectionWeights.size()-1;
		for(size_t i=0;i<numOutputs_int;++i) {
			tempBeta=label[i]-net[outputsIndex][i].output;
			net[outputsIndex][i].beta=tempBeta;
			RMS+=tempBeta*tempBeta;
			ULLI IxN=lastHiddenSize*i;
			tempActivation=sigmoid_derivative(net[outputsIndex][i].output);
			for(size_t j=0;j<lastHiddenSize;++j) {
				net[lastHiddenIndex][j].beta+=connectionWeights[connectionWeightsIndex][IxN+j].weight*tempActivation*tempBeta;
				deltaweight=net[lastHiddenIndex][j].output*tempBeta;
				//deltaweight=net[lastHiddenIndex][j].output*net[outputsIndex][i].beta*sigmoid_derivative(net[outputsIndex][i].output);
				//connectionWeights[connectionWeightsIndex][IxN+j].weight+=(StepSize*deltaweight)+(Momentum*connectionWeights[connectionWeightsIndex][IxN+j].prevDelta);
				connectionWeights[connectionWeightsIndex][IxN+j].weight+=StepSize*deltaweight+connectionWeights[connectionWeightsIndex][IxN+j].prevDelta;
				connectionWeights[connectionWeightsIndex][IxN+j].prevDelta=deltaweight;
			}
			//deltaweight=net[outputsIndex][i].beta+sigmoid_derivative(net[outputsIndex][i].output);
			//net[outputsIndex][i].biasWeight+=(StepSize*tempBeta)+(Momentum*net[outputsIndex][i].prevBiasDelta);
			net[outputsIndex][i].biasWeight+=StepSize*tempBeta+net[outputsIndex][i].prevBiasDelta;
			net[outputsIndex][i].prevBiasDelta=tempBeta;
			//yes,3,inf,yes//no,3,inf,yes
		}
		--connectionWeightsIndex;
		while(lastHiddenIndex>0) {
			for(size_t i=0;i<net[lastHiddenIndex].size();++i) {
				ULLI IxN=i*net[lastHiddenIndex-1].size();
				for(size_t j=0;j<net[lastHiddenIndex-1].size();++j) {
					deltaweight=net[lastHiddenIndex-1][j].output*sigmoid_derivative(net[lastHiddenIndex][i].output)*net[lastHiddenIndex][i].beta;
					//connectionWeights[connectionWeightsIndex][IxN+j].weight+=(StepSizeAcc*deltaweight)+(Momentum*connectionWeights[connectionWeightsIndex][IxN+j].prevDelta);
					//connectionWeights[connectionWeightsIndex][IxN+j].weight+=(StepSize*deltaweight)+(Momentum*connectionWeights[connectionWeightsIndex][IxN+j].prevDelta);
					connectionWeights[connectionWeightsIndex][IxN+j].weight+=StepSize*deltaweight+connectionWeights[connectionWeightsIndex][IxN+j].prevDelta;
					connectionWeights[connectionWeightsIndex][IxN+j].prevDelta=deltaweight;
					net[lastHiddenIndex-1][j].beta+=connectionWeights[connectionWeightsIndex][IxN+j].weight*sigmoid_derivative(net[lastHiddenIndex][i].output)*net[lastHiddenIndex][i].beta;
				}
				deltaweight=net[lastHiddenIndex][i].beta*sigmoid_derivative(net[lastHiddenIndex][i].output);
				//net[lastHiddenIndex][i].biasWeight+=(StepSizeAcc*deltaweight)+(Momentum*net[lastHiddenIndex][i].prevBiasDelta);
				//net[lastHiddenIndex][i].biasWeight+=(StepSize*deltaweight)+(Momentum*net[lastHiddenIndex][i].prevBiasDelta);
				net[lastHiddenIndex][i].biasWeight+=StepSize*deltaweight+net[lastHiddenIndex][i].prevBiasDelta;
				net[lastHiddenIndex][i].prevBiasDelta=deltaweight;
			}
			--connectionWeightsIndex;
			--lastHiddenIndex;
		}
	}

	double doRandom(ULLI randomPrecision) {
		//random() for -1 to +1
		//cout << ((((float)rand()/(RAND_MAX)) * 2.0) - 1.0) << endl;
		//cout << ULLONG_MAX << endl;//random() << endl;
		//cout << DBL_MAX << endl;
		//cout << ((double)random()/(double)ULLONG_MAX)*2.0 << endl;
		//cout << ((((double)(random()%randomPrecision)/(double)randomPrecision)*2.0)-1.0) << endl;
		//return ((((double)(random()%randomPrecision)/(double)randomPrecision)*2.0)-1.0);
		double what=(double)rand();
		what/=(double)RAND_MAX;
		what*=2.0;
		what-=1.0;
		return what;
		//return (((double)(random()%randomPrecision)/(double)randomPrecision)*2.0);
		//return (((double)(random()%randomPrecision)/(double)randomPrecision)-0.5);
		//return ((double)(random()%randomPrecision)/(double)randomPrecision);
	}
};

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
	return x*(1.0-x);
}
/*void printNet(vector<neuron_t> &net, vector<int> &hiddenMatrix) {
	ULLI maxElement=*max_element(hiddenMatrix.begin(),hiddenMatrix.end());
	ULLI layers=hiddenMatrix.size();
	for(ULLI i=0;i<maxElement;++i) {
		for(ULLI j=0;j<layers;++j) {
			if(i<hiddenMatrix[i]) {
				printf("");
			}
		}
	}
}*/

//Made a separate 'doMain' function (as opposed to 'int main' in order to not have to constantly work
//around the OpenMPI directives(among other things) in 'int main'  Just makes it easier to
//work with.
void doMain(int my_rank, string hostname, int num_nodes) {



	//Testing arrays for device_vectors
	/*int x=4,y=5;
	device_vector<float> connWeights[x][y];
	connWeights[0][0]=device_vector<float>(10);
	thrust::sequence(connWeights[0][0].begin(),connWeights[0][0].end(),0,1);
	for(auto a:connWeights[0][0]) {
		cout << a << endl;
	}
	return;*/
	////////////////////////////////////////////////////////////////////
	//This section isn't used currently.  Was used to test if I was reading
	//in the pictures from the MNIST dataset properly

	//cout << "sizeof float: " << sizeof(float) << endl;
	//cout << "sizeof double: " << sizeof(double) << endl;
	/*vector<vector<UNCHAR>> testData;
	ReadMNIST_UNCHAR("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<UNCHAR>> trainData;
	ReadMNIST_UNCHAR("train-images.idx3-ubyte",60000,784,trainData);
	vector<UNCHAR> testLabels;
	vector<UNCHAR> trainLabels;*/

	/*vector<vector<double>> testData;
	ReadMNIST_double("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<double>> trainData;
	ReadMNIST_double("train-images.idx3-ubyte",60000,784,trainData);*/


	////////////////////////////////////////////////////////////////////
	//Uncomment the following section in order to run MNIST data through the
	//sequential version of the algorithm
	//(be sure to comment out the other sections in 'doMain' before
	// trying to compile however)

	//MNIST picutres are 28x28 pixels.  So naturally we make the 
	//number of inputs to the nerual net as 784 (28x28).  I found
	//that the NN works the best when there is only one hidden layer
	//and that hidden layer has 1.5 times as many nodes as input layers
	//So 1.5*784=1176 hidden nodes in the middle.  Also the algorithm
	//works better when there is more than one output node.  So there 
	//is one output node for each decimal digit (10). 
	//This runs way too slow to be useful in the sequential version,
	//but the speed up I get from early trials with Cuda and MNIST
	//makes it run fast enough to run through the entire dataset once in less
	//than about 10 minutes.  Which is doable.  

	/*vector<vector<neuron_t>> testData;
	ReadMNIST_neuron_t("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<neuron_t>> trainData;
	ReadMNIST_neuron_t("train-images.idx3-ubyte",60000,784,trainData);
	vector<vector<double>> testLabels;
	vector<vector<double>> trainLabels;
	ifstream file("t10k-labels.idx1-ubyte",ios::binary);
	if(file.is_open()) {
		int placeHolder=0;
		file.read((char*)&placeHolder,sizeof(placeHolder));
		file.read((char*)&placeHolder,sizeof(placeHolder));
		for(int i=0;i<10000;++i) {
			UNCHAR temp=0;
			file.read((char*)&temp,1);
			testLabels.push_back(vector<double>(10,0.0));
			testLabels.back()[temp]=1.0;
			//testLabels.push_back((double)temp);
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
			trainLabels.push_back(vector<double>(10,0.0));
			trainLabels.back()[temp]=1.0;
			//trainLabels.push_back((double)temp);
		}
		file2.close();
	}//*/

	////////////////////////////////////////////////////////////////////
	//Uncomment out the following section in order to run the MNIST data through the
	//Cuda(parallel) version of the algorithm
	//(be sure to comment out the other sections in 'doMain' before
	// trying to compile however)

	/*vector<float> testData;
	ReadMNIST_float("t10k-images.idx3-ubyte",10000,784,testData);
	vector<float> trainData;
	ReadMNIST_float("train-images.idx3-ubyte",60000,784,trainData);
	vector<float> testLabels;
	vector<float> trainLabels;
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
					testLabels.push_back(1.0f);
				} else {
					testLabels.push_back(0.0f);
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
					trainLabels.push_back(1.0f);
				} else {
					trainLabels.push_back(0.0f);
				}
			}
		}
		file2.close();
	}
	vector<int> hiddenMatrix;
	hiddenMatrix.push_back(784+(784/2));
	//hiddenMatrix.push_back(7);
	neuralNet test(784,10,hiddenMatrix,true);
	test.train_cuda(trainData,trainLabels,0.0001,1000000,784,60000);	//*/

	//////////////////////////////////////////////////////////////////////
	//Uncomment out the following section in order to run the 'bit counting' problem thru
	//the NN thru Cuda(parallel) version of the algorithm
	//(be sure to comment out the other sections in 'doMain' before
	// trying to compile however)

	/*vector<int> hiddenMatrix;
	//hiddenMatrix.push_back(2000);
	//hiddenMatrix.push_back(200);
	//hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(BITS*2);
	hiddenMatrix.push_back(BITS+(BITS/2));
	//hiddenMatrix.push_back(4);
	//hiddenMatrix.push_back(4);
	//hiddenMatrix.push_back(4);
	//hiddenMatrix.push_back(4);
	//hiddenMatrix.push_back(4);
	//hiddenMatrix.push_back(4);
	
	neuralNet test(BITS,BITS,hiddenMatrix,true);
	vector<float> countingTest;
	vector<float> countingLabels;
	int size=pow(2,BITS);
	for(int i=0;i<size;++i) {
		//countingTest.push_back(vector<float>(BITS,0.0f));
		//countingLabels.push_back(vector<float>(BITS,0.0));
		for(int j=0;j<BITS;++j) {
			countingTest.push_back((float)bitset<BITS>(i)[(BITS-1)-j]);
			countingLabels.push_back((float)bitset<BITS>((i+1)%size)[(BITS-1)-j]);
		}
	}
	test.train_cuda(countingTest,countingLabels,0.005f,1000000,BITS,size);
	return;//*/

	////////////////////////////////////////////////////////////////////////////////
	//Uncomment out the following section of code the run the 'bit counting' problem thru
	//the sequential version of the algorithm
	//(be sure to comment out the other sections in 'doMain' before
	// trying to compile however)

	//This code just tries to see if I can get the neuralNet to
	//count in binary  input:  0 0 0 0 to output: 0 0 0 1

	vector<int> hiddenMatrix;
	//hiddenMatrix.push_back(BITS+(BITS/2));
	//hiddenMatrix.push_back(BITS+(BITS/2));
	for(int i=0;i<2;++i) {
		hiddenMatrix.push_back(BITS+(BITS/2));
		//hiddenMatrix.push_back(12);
	}
	//neuralNet test(BITS,BITS,hiddenMatrix,false);
	//vector<vector<neuron_t>> countingTest;
	//vector<vector<double>> countingLabels;
	int size=pow(2,BITS);
	double toDivideRMS=((double)size)*(double)BITS;
	double RMS=0.0;
	vector<neuron_t> countingTest[size];
	vector<double> countingLabels[size];
	for(int i=0;i<size;++i) {
		//countingTest.push_back(vector<neuron_t>(BITS));
		//countingLabels.push_back(vector<double>(BITS,0.0));
		countingLabels[i]=vector<double>(BITS,0.0);
		countingTest[i]=vector<neuron_t>(BITS);
		for(int j=0;j<BITS;++j) {
			//countingTest.back()[j].output=(double)bitset<BITS>(i)[(BITS-1)-j];
			//countingLabels.back()[j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
			countingTest[i][j].output=(double)bitset<BITS>(i)[(BITS-1)-j];
			countingLabels[i][j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
		}
	}
	//test.train(countingTest,countingLabels,0.00001,1000000);
	hiddenMatrix.insert(hiddenMatrix.begin(),BITS);
	hiddenMatrix.push_back(BITS);
	ULLI maxElement=*max_element(hiddenMatrix.begin(),hiddenMatrix.end());
	ULLI layers=hiddenMatrix.size();
	vector<neuron_t> net[layers];
	vector<double> connWeights[layers-1][maxElement];
	int index=0;
	for(auto h:hiddenMatrix) {
		net[index++]=vector<neuron_t>(h);
	}
	for(ULLI i=0;i<layers-1;++i) {
		for(ULLI j=0;j<maxElement;++j) {
			connWeights[i][j]=vector<double>(maxElement);
			for(ULLI k=0;k<maxElement;++k) {
				connWeights[i][j][k]=((((double)rand())/((double)RAND_MAX))*2.0)-1.0;
			}
		}
	}
	ULLI outputsIndex=layers-1;
	double learningRate=0.9;
	double RMSwanted=0.0001;
	double temp;
	RMS=100000.0;
	double minRMS=100000.0;
	int maxIter=10000000;
	for(int ii=0;ii<maxIter && RMSwanted<RMS;++ii) {
		RMS=0.0;
		for(int iii=0;iii<size;++iii) {

			//forward
			net[0]=countingTest[iii];
			for(int i=1;i<layers;++i) {
				for(int j=0;j<hiddenMatrix[i];++j) {
					net[i][j].input=0.0;
					for(int k=0;k<hiddenMatrix[i-1];++k) {
						net[i][j].input+=connWeights[i-1][k][j]*net[i-1][k].output;
					}
					net[i][j].output=sigmoid(net[i][j].input);
					net[i][j].localDerivative=sigmoid_derivative(net[j][i].output);
					//printf("net[%d][%d].output: %.15f derivative: %.15f\n",i,j,net[i][j].output,net[i][j].localDerivative);
				}
			}

			//backProp
			for(int i=0;i<BITS;++i) {
				temp=(net[outputsIndex][i].output-countingLabels[iii][i])*net[outputsIndex][i].localDerivative;
				//temp=net[outputsIndex][i].output-countingLabels[iii][i];
				net[outputsIndex][i].errorSignal=temp;
				RMS+=temp*temp;
			}
			for(int i=layers-2;i>-1;--i) {
				for(int j=0;j<hiddenMatrix[i];++j) {
					net[i][j].errorSignal=0.0;
					for(int k=0;k<hiddenMatrix[i+1];++k) {
						net[i][j].errorSignal+=connWeights[i][j][k]*net[i+1][k].errorSignal;
						connWeights[i][j][k]-=learningRate*net[i+1][k].errorSignal*net[i][j].output;
					}
					net[i][j].errorSignal*=net[i][j].output*(1.0-net[i][j].output);
				}
			}
		}
		RMS=sqrt(RMS/toDivideRMS);
		if(RMS<minRMS) {
			minRMS=RMS;
			//printf("minRMS Error: %.15f Iteration: %d",RMS,ii);
		}
		printf("current RMS: %.15f minRMS Error: %.15f Iteration: %d of %d\r",RMS,minRMS,ii,maxIter);
	}
	cout << endl;
	for(int ii=0;ii<size;++ii) {
		net[0]=countingTest[ii];
		cout << "Inputs: ";
		for(auto i:net[0]) {
			printf("%.15f ",i.output);
		}		
		for(int i=1;i<layers;++i) {
			for(int j=0;j<hiddenMatrix[i];++j) {
				net[i][j].input=0.0;
				for(int k=0;k<hiddenMatrix[i-1];++k) {
					net[i][j].input+=connWeights[i-1][k][j]*net[i-1][k].output;
				}
				net[i][j].output=sigmoid(net[i][j].input);
				net[i][j].localDerivative=sigmoid_derivative(net[j][i].output);
			}
		}
		cout << endl << "Output: ";
		for(auto o:net[outputsIndex]) {
			printf("%.15f ",o.output);
		}
		cout << endl;		
	}

	return;//*/

	////////////////////
	//I was outputting a bitmap
	//to disk in order to see if I was reading in the MNIST 28x28 pictures correctly
	//Since I'm sure I'm doing that correctly, I could probably remove this code.
	//....I suppose...whatever
	/*vector<int> hiddenMatrix;
	hiddenMatrix.push_back(784+(784/2));
	//hiddenMatrix.push_back(7);
	neuralNet test(784,10,hiddenMatrix,false);
	test.train(trainData,trainLabels,0.005,1000000);//*/
	//cout << (int)trainLabels[0] << " " << (int)trainLabels[1] << endl;
	/*vector<int> temp;
	for(auto p:trainData[1]) {
		temp.push_back((int)p);
	}
	int* t=&temp[0];*/
	//intarray2bmp::intarray2bmp("outputtest.bmp",&trainData[0][0],(UNCHAR)28,(UNCHAR)28,(UNCHAR)0,(UNCHAR)255);//*/
}

int main(int argc, char *argv[]) {
	srandom(time_point_cast<nanoSec>(high_resolution_clock::now()).time_since_epoch().count());
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

	MPI_Finalize();//*/
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

//Four versions of this load function.  I suppose a C++ template would be appropriate
//, but why?  It's probably more trouble than it's worth.  We're not using the MNIST
//data for our final problem anyway.  Also depending on the type, the code is different
//for each data type anyway.
void ReadMNIST_neuron_t(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<neuron_t>> &arr) {
    arr.resize(NumberOfImages,vector<neuron_t>(DataOfAnImage));
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
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    UNCHAR temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c].output = ((double)temp)/256.0;
                }
            }
        }
    }
    file.close();
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
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    UNCHAR temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= ((double)temp)/256.0;
                }
            }
        }
    }
    file.close();
}

//void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<float>> &arr) {
void ReadMNIST_float(string filename, int NumberOfImages, int DataOfAnImage, vector<float> &arr) {
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
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    UNCHAR temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    //arr[i][(n_rows*r)+c]= ((float)temp)/256.0f;
                    //cout << "from read: " << ((float)temp)/256.0f << endl;
                    arr.push_back(((float)temp)/256.0f);
                    //cout << "from read: " << arr.back() << endl;
                }
            }
        }
    }
    file.close();
}

void ReadMNIST_UNCHAR(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<UNCHAR>> &arr) {
    arr.resize(NumberOfImages,vector<UNCHAR>(DataOfAnImage));
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
            for(int r=0;r<n_rows;++r) {
                for(int c=0;c<n_cols;++c) {
                    UNCHAR temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= temp;
                }
            }
        }
    }
    file.close();
}
