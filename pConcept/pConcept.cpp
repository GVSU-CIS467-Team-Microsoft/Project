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
#include <mpi.h>
#include "helper_cuda.h"
#include "helper_string.h"
#include "intarray2bmp.hpp"
#include <cmath>
#include <limits.h>
#include <float.h>
using namespace std;
using namespace thrust;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define ULLI unsigned long long int
#define UNCHAR unsigned char
//#define MAX 25

void ReadMNIST_double(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr);
void ReadMNIST_UNCHAR(string filename, int NumberOfImages, int DataOfAnImage, vector<vector<UNCHAR>> &arr);

// A separate struct in case this needs to be
// more complex in the future
struct neuron_t {
	double value=0.0;
	double biasWeight=1.0;
	double beta=0.0;
	double prevBiasDelta=0.0;
};
struct connectionWeights_t {
	double weight;
	double prevDelta=0.0;
	int from=-1;
	int to=-1;
};

//This currently only handles 'double' values
class neuralNet {
public:
	neuralNet(int in, int out, vector<int> &sizeHiddenMatrix) : Momentum(0.9), StepSize(0.25), divergingCheck(0), RMS(0.0), epoch(0) {
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
		for(size_t i=1;i<net.size();++i) {
			for(size_t j=0;j<net[i].size();++j) {
				ULLI JxN=j*net[i-1].size();
				for(size_t k=0;k<net[i-1].size();++k) {
					connectionWeights[i-1][JxN+k].from=k;
					connectionWeights[i-1][JxN+k].to=j;
				}
			}
		}
	};
	~neuralNet(){};
	void loadWeights(string filename);
	void saveWeights(string filename);

	void train(vector<vector<double>> &data, vector<vector<double>> &labels, double desiredError, ULLI max_cycles) {
		LastRMS=99.9;
		cost=1.0;
		ULLI outerEpoch=0;
		minRMS=1000000.0;
		toDivideRMS=((double)data.size())*numOutputs;
		dataSetSize=data.size();

		cout << "Started training..." << endl;
		//bool RMS_not_moving=false;
		//while(LastRMS > desiredError && outerEpoch<max_cycles && !RMS_not_moving) {
		while(LastRMS > desiredError && outerEpoch<max_cycles) {
			for(size_t i=0;i<dataSetSize;++i) {
				/*int which=random()%data.size();
				forwardProp(data[which],labels[which]);
				backwardProp(data[which],labels[which]);*/
				forwardProp(data[i],labels[i]);
				backwardProp(data[i],labels[i]);
				//printNetworkState();
			}
			++epoch;
			LastRMS=sqrt(RMS/toDivideRMS);
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
			}
			if(LastRMS<minRMS) {
				minRMS=LastRMS;
				printOutputsForSet(data,labels);
			}
			PrevRMSError=LastRMS;
			++outerEpoch;
		}
		printOutputsForSet(data,labels);
		cout << "Training done: Cycles: " << outerEpoch << " Error: " << LastRMS << " minRMS: " << minRMS << "\n";
	}

	void printOutputsForSet(vector<vector<double>> &data, vector<vector<double>> &labels) {
		cout << endl;
		for(size_t i=0;i<data.size();++i) {
			forwardProp(data[i],labels[i]);
			cout << "Inputs: ";
			for(auto n:net[0]) {
				cout << n.value << " ";
			}
			cout << endl;//*/
			cout << "Output:\t\t";
			for(size_t j=0;j<numOutputs_int;++j) {
				printf("%.10f ",net[outputsIndex][j].value);
				//cout << net[outputsIndex][j].value << " ";
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
				cout << "Neuron: " << neuron++ << " value: " << nn.value << " biasWeight: " << nn.biasWeight << " prevBiasDelta: " << nn.prevBiasDelta << " beta: " << nn.beta << endl;
			}
			cout << endl;
		}
		layer=0;
		for(auto c:connectionWeights) {
			cout << "Connection weights from layer: " << layer++ << endl;
			for(auto cc:c) {
				cout << "From: " << cc.from << " To: " << cc.to << " weight: " << cc.weight << " prevDelta: " << cc.prevDelta << endl;
			}
		}
		cout << "-------------------------------------\n";
		sleep(1);
	}

private:
	vector<vector<neuron_t>> net;
	vector<vector<connectionWeights_t>> connectionWeights;
	double RMS, LastRMS, StepSize, StepSizeAcc, Momentum, PrevRMSError;
	double numOutputs, minRMS, toDivideRMS, cost;
	ULLI divergingCheck, epoch, outputsIndex, numOutputs_int;
	ULLI dataSetSize;

	double sigmoid(const double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	double sigmoid_derivative(const double x) {
		return x*(1.0-x);
	}
	double tanH_derivative(const double x) {
		double th = tanh(x); // tanh(x) \in (-1,1); cosh(x) \in (1,inf)
		return 1.0 - th*th; // sech^2(x) = 1 - tanh^2(x)
	}

	void forwardProp(vector<double> &item, vector<double> &label) {
		for(size_t i=0;i<net[0].size();++i) {
			net[0][i].value=item[i];
		}
		size_t outputIndex=net.size()-1;
		for(size_t i=1;i<net.size();++i) {
			for(size_t j=0;j<net[i].size();++j) {
				net[i][j].value=0.0;
				net[i][j].beta=0.0;
				ULLI JxN=j*net[i-1].size();
				for(size_t k=0;k<net[i-1].size();++k) {
					net[i][j].value+=net[i-1][k].value*connectionWeights[i-1][JxN+k].weight;
				}
				net[i][j].value+=net[i][j].biasWeight;
				net[i][j].value=sigmoid(net[i][j].value);
			}
		}
	}

	void backwardProp(vector<double> &item, vector<double> &label) {
		double deltaweight, tempBeta;
		ULLI lastHiddenSize=net[net.size()-2].size();
		ULLI lastHiddenIndex=net.size()-2;
		ULLI connectionWeightsIndex=connectionWeights.size()-1;
		for(size_t i=0;i<numOutputs_int;++i) {
			tempBeta=label[i]-net[outputsIndex][i].value;
			net[outputsIndex][i].beta=tempBeta;
			RMS+=tempBeta*tempBeta;
			ULLI IxN=lastHiddenSize*i;
			for(size_t j=0;j<lastHiddenSize;++j) {
				/*if(connectionWeights[connectionWeightsIndex][IxN+j].from!=j || connectionWeights[connectionWeightsIndex][IxN+j].to!=i) {
					cout << "Found bad connection: i: " << i << " j: " << j << endl;
				}*/
				net[lastHiddenIndex][j].beta+=connectionWeights[connectionWeightsIndex][IxN+j].weight*sigmoid_derivative(net[outputsIndex][i].value*net[outputsIndex][i].beta);
				deltaweight=net[lastHiddenIndex][j].value*net[outputsIndex][i].beta;
				//deltaweight=net[lastHiddenIndex][j].value*net[outputsIndex][i].beta*sigmoid_derivative(net[outputsIndex][i].value);
				connectionWeights[connectionWeightsIndex][IxN+j].weight+=(StepSize*deltaweight)+(Momentum*connectionWeights[connectionWeightsIndex][IxN+j].prevDelta);
				connectionWeights[connectionWeightsIndex][IxN+j].prevDelta=deltaweight;
			}
			deltaweight=net[outputsIndex][i].beta;
			//deltaweight=net[outputsIndex][i].beta+sigmoid_derivative(net[outputsIndex][i].value);
			net[outputsIndex][i].biasWeight+=(StepSize*deltaweight)+(Momentum*net[outputsIndex][i].prevBiasDelta);
			net[outputsIndex][i].prevBiasDelta=deltaweight;
		}
		--connectionWeightsIndex;
		while(lastHiddenIndex>0) {
			for(size_t i=0;i<net[lastHiddenIndex].size();++i) {
				ULLI IxN=i*net[lastHiddenIndex-1].size();
				for(size_t j=0;j<net[lastHiddenIndex-1].size();++j) {
					/*if(connectionWeights[connectionWeightsIndex][IxN+j].from!=j || connectionWeights[connectionWeightsIndex][IxN+j].to!=i) {
						cout << "Found bad connection: i: " << i << " j: " << j << endl;
					}*/				
					net[lastHiddenIndex-1][j].beta+=connectionWeights[connectionWeightsIndex][IxN+j].weight*sigmoid_derivative(net[lastHiddenIndex][i].value*net[lastHiddenIndex][i].beta);
					deltaweight=net[lastHiddenIndex-1][j].value*sigmoid_derivative(net[lastHiddenIndex][i].value*net[lastHiddenIndex][i].beta);
					connectionWeights[connectionWeightsIndex][IxN+j].weight+=(StepSizeAcc*deltaweight)+(Momentum*connectionWeights[connectionWeightsIndex][IxN+j].prevDelta);
					//connectionWeights[connectionWeightsIndex][IxN+j].weight+=(StepSize*deltaweight)+(Momentum*connectionWeights[connectionWeightsIndex][IxN+j].prevDelta);
					connectionWeights[connectionWeightsIndex][IxN+j].prevDelta=deltaweight;
				}
				deltaweight=net[lastHiddenIndex][i].beta*sigmoid_derivative(net[lastHiddenIndex][i].value);
				net[lastHiddenIndex][i].biasWeight+=(StepSizeAcc*deltaweight)+(Momentum*net[lastHiddenIndex][i].prevBiasDelta);
				//net[lastHiddenIndex][i].biasWeight+=(StepSize*deltaweight)+(Momentum*net[lastHiddenIndex][i].prevBiasDelta);
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
		return ((((double)(random()%randomPrecision)/(double)randomPrecision)*2.0)-1.0);
		//return (((double)(random()%randomPrecision)/(double)randomPrecision)*2.0);
		//return ((((double)(random()%randomPrecision)/(double)randomPrecision)*1.0)-0.5);
	}
};

#define BITS 4
void doMain(int my_rank, string hostname, int num_nodes) {
	/*vector<vector<UNCHAR>> testData;
	ReadMNIST_UNCHAR("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<UNCHAR>> trainData;
	ReadMNIST_UNCHAR("train-images.idx3-ubyte",60000,784,trainData);
	vector<UNCHAR> testLabels;
	vector<UNCHAR> trainLabels;*/
	/*vector<vector<double>> testData;
	ReadMNIST_double("t10k-images.idx3-ubyte",10000,784,testData);
	vector<vector<double>> trainData;
	ReadMNIST_double("train-images.idx3-ubyte",60000,784,trainData);
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
	vector<int> hiddenMatrix;
	//hiddenMatrix.push_back(2000);
	//hiddenMatrix.push_back(200);
	hiddenMatrix.push_back(pow(2,BITS+1)+1);
	//hiddenMatrix.push_back(3);
	//hiddenMatrix.push_back(3);
	//hiddenMatrix.push_back(3);
	//hiddenMatrix.push_back(6);
	//hiddenMatrix.push_back(3);

	//This code just tries to see if I can get the neuronNet to
	//count in binary  input:  0 0 0 0 to output: 0 0 0 1
	neuralNet test(BITS,BITS,hiddenMatrix);
	vector<vector<double>> countingTest;
	vector<vector<double>> countingLabels;
	int size=pow(2,BITS);
	for(int i=0;i<size;++i) {
		countingTest.push_back(vector<double>(BITS,0.0));
		countingLabels.push_back(vector<double>(BITS,0.0));
		for(int j=0;j<BITS;++j) {
			countingTest.back()[j]=(double)bitset<BITS>(i)[(BITS-1)-j];
			countingLabels.back()[j]=(double)bitset<BITS>((i+1)%size)[(BITS-1)-j];
		}
	}
	test.train(countingTest,countingLabels,0.005,10000000);
	return;

	/*vector<int> hiddenMatrix;
	hiddenMatrix.push_back(112);
	//hiddenMatrix.push_back(7);
	neuralNet test(784,10,hiddenMatrix);
	test.train(trainData,trainLabels,0.005,1000000);*/
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

	/*int my_rank, num_nodes;
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

	MPI_Finalize();*/
	doMain(0,"",0);

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
