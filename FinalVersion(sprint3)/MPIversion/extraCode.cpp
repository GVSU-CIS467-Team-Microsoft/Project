//extraneous code not being used

void train_Quad(vector<vector<float>> &pData, vector<vector<float>> &pLabels, ULLI maxIter, 
		float RMSwant, int doDataSetSize, float lRate, vector<vector<float>> &pTestData, vector<vector<float>> &pTestLabels, bool vlRate) {

		if(!showInterval) {
			showInterval=10;
		}
		vector<int> bLabels;
		for(auto p:pLabels) {
			bLabels.push_back(max_element(p.begin(), p.end())-p.begin());
		}

		if(lRate<0.0f) {
			learningRate=0.05f;
		} else {
			learningRate=lRate;
		}
		dataSetSize=doDataSetSize;
		int testBatchSize=batchSize;
		/*if(batchSize>10000) {
			testBatchSize=10000;
		} else {
			testBatchSize=100;
		}*/

		UINT batchStart, batchEnd, thisSize, nextSize;
		RMSwanted=RMSwant;
		maxEpochs=maxIter;
		itemSize=pData[0].size();

		int testSetSize=pTestData.size();
		vector<int> btLabels;
		vector<float> testData[testSetSize/testBatchSize];
	
		if(testSetSize) {
			for(auto p:pTestLabels) {
				btLabels.push_back(max_element(p.begin(), p.end())-p.begin());
			}
		}

		vector<float> data[totalNum];
		vector<float> labels[totalNum];

		//Creating pre-made batches so I can simply copy them to layer[0]
		if(!my_rank) {
			cout << "Making batches in memory...\n";
		}
		int whichBatch=0;
		for(int itemNum=0;itemNum<dataSetSize;itemNum+=batchSize) {
			batchStart=0;
			batchEnd=0;
			data[whichBatch]=vector<float>(itemSize*batchSize);
			memTracker((ULLI)(itemSize*batchSize*4),false);
			labels[whichBatch]=vector<float>(batchSize*numOutputs);
			memTracker((ULLI)(numOutputs*batchSize*4),false);
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
			testData[whichBatch]=vector<float>(itemSize*testBatchSize);
			memTracker((ULLI)(itemSize*testBatchSize*4),false);
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

		vector<float>::iterator iter;
		int position;
		int gotRight=0, prevSize;
		//int numBatches=dataSetSize/batchSize;
		//toDivideRMS=learningRate/((float)numBatches*(float)batchSize);
		//toDivideRMS=learningRate/((float)batchSize*(float)showInterval);
		//toDivideRMS=learningRate/((float)batchSize*(float)num_nodes);//*(float)showInterval);
		toDivideRMS=learningRate/(float)batchSize;
		//toDivideRMS=learningRate/(float)showInterval;
		int maxGotRight=0, maxTestRight=-1, ii;
		vector<float> *which;
		float seconds, totalTime=0.0;
		high_resolution_clock::time_point startTime, endTime;
		int showIntervalCountDown=showInterval;
		int mPlus, sInterval=showInterval;
		bool once=true;

		vector<pthread_t> threads;
		pthread_attr_t attr;
    	cpu_set_t cpus;
    	pthread_attr_init(&attr);

		//divThreads dThreads((float)num_nodes);
		divThreads dThreads((float)totalNum);
		multi_helper hTimes((float)numberOfProcessors);

		memTracker(0,true);

		for(ULLI epochNum=0;!threadExit && epochNum<maxEpochs && maxGotRight!=dataSetSize && maxTestRight!=testSetSize;++epochNum) {//epochNum+=sInterval) {
			whichBatch=0;
			gotRight=0;
			startTime=high_resolution_clock::now();
			if(once) {
			    //for(int j=0;j<numThreads;++j) {
				int j=1;

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
			    //}
			    once=false;
			}
			pthread_barrier_wait(&barrier);
			for(int i=1;i<numThreads;++i) {
				for(int j=0;j<outputsIndex;++j) {
					ii=j+1;
					//transform(NNlayersQ[0][j].outerDeltaW.begin(),NNlayersQ[0][j].outerDeltaW.end(),NNlayersQ[i][j].outerDeltaW.begin(),NNlayersQ[0][j].outerDeltaW.begin(),plus<float>());
					//transform(NNlayersQ[0][ii].outerDeltaB.begin(),NNlayersQ[0][ii].outerDeltaB.end(),NNlayersQ[i][ii].outerDeltaB.begin(),NNlayersQ[0][ii].outerDeltaB.begin(),plus<float>());
					transform(NNlayersQ[0][j].innerDeltaW.begin(),NNlayersQ[0][j].innerDeltaW.end(),NNlayersQ[i][j].innerDeltaW.begin(),NNlayersQ[0][j].innerDeltaW.begin(),plus<float>());
					transform(NNlayersQ[0][ii].innerDeltaB.begin(),NNlayersQ[0][ii].innerDeltaB.end(),NNlayersQ[i][ii].innerDeltaB.begin(),NNlayersQ[0][ii].innerDeltaB.begin(),plus<float>());
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
					//MPI_Allreduce(&NNlayersQ[0][i].outerDeltaW[0],&NNlayersQ[1][i].outerDeltaW[0],NNlayersQ[0][i].allW,MPI_float,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][ii].outerDeltaB[0],&NNlayersQ[1][ii].outerDeltaB[0],NNlayersQ[0][ii].allN,MPI_float,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][i].innerDeltaW[0],&tempDeltaW[i][0],NNlayersQ[0][i].allW,MPI_float,MPI_SUM,MPI_COMM_WORLD);
					//MPI_Allreduce(&NNlayersQ[0][i+1].innerDeltaB[0],&tempDeltaB[i][0],NNlayersQ[0][i+1].allN,MPI_float,MPI_SUM,MPI_COMM_WORLD);
					transform(NNlayersQ[0][i].weightsMatrix.begin(),NNlayersQ[0][i].weightsMatrix.end(),NNlayersQ[2][i].weightsMatrix.begin(),hTimes);
					transform(NNlayersQ[0][ii].biases.begin(),NNlayersQ[0][ii].biases.end(),NNlayersQ[2][ii].biases.begin(),hTimes);
					MPI_Allreduce(&NNlayersQ[2][i].weightsMatrix[0],&NNlayersQ[1][i].weightsMatrix[0],NNlayersQ[0][i].allW,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
					MPI_Allreduce(&NNlayersQ[2][ii].biases[0],&NNlayersQ[1][ii].biases[0],NNlayersQ[0][ii].allN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
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
						cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &NNlayersQ[0][i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &NNlayersQ[0][ii].atNeuronInputs[0], nextSize);
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
							float oo=NNlayersQ[0][outputsIndex].atNeuronOutputs[ot];
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
				printf("Training epoch: %llu -- Got %d of %d -- max right: %d -- lRate: %.5f -- ",epochNum,gotRight,dataSetSize,maxGotRight,learningRate);
				gotRight=0;		
				whichBatch=0;
				for(int t=0;t<testSetSize;t+=testBatchSize) {
					which=&testData[whichBatch];
					for(int i=0;i<outputsIndex;++i) {
						ii=i+1;
						thisSize=hiddenMatrix[i];
						nextSize=hiddenMatrix[ii];
						cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, testBatchSize, thisSize, 1.0, &NNlayersQ[0][i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &NNlayersQ[0][ii].atNeuronInputs[0], nextSize);
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
							float oo=NNlayersQ[0][outputsIndex].atNeuronOutputs[ot];
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
				float errRate=(1.0-((float)gotRight/(float)testSetSize))*100.0;
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
			oFile.write((char*)&layers,sizeof(int));
			for(int i=0;i<hiddenMatrix.size();++i) {
				oFile.write((char*)&hiddenMatrix[i],sizeof(int));
			}
			oFile.write((char*)&batchSize,sizeof(int));
			oFile.write((char*)&learningRate,sizeof(float));
			for(int i=0;i<outputsIndex;++i) {
				for(int j=0;j<NNlayersQ[0][i].allW;++j) {
					float o=NNlayersQ[0][i].weightsMatrix[j];
					oFile.write((char*)&o,sizeof(float));
				}
			}
			for(int i=1;i<layers;++i) {
				for(int j=0;j<NNlayersQ[0][i].allN;++j) {
					float o=NNlayersQ[0][i].biases[j];
					oFile.write((char*)&o,sizeof(float));
				}
			}
			oFile.close();
		}
		cout << "Done\n";
	}


	struct idLink {
    int whichThread;
    int interval;
    vector<float> *data;
    vector<float> *labels;
    vector<NN_layer> *NNlayersQ;
    vector<int> *hiddenMatrix;
    float learningRate;
    int batchSize;
};

/*void *fourthThread(void *thread_parm) {
	idLink data=*((idLink*) thread_parm);
	int myID=data.whichThread;
	int howMany=data.interval;
	vector<int> hiddenMatrix=*data.hiddenMatrix;
	int layers=hiddenMatrix.size();
	int outputsIndex=layers-1;
	int batchSize=data.batchSize;
	int numOutputs=hiddenMatrix[outputsIndex];
	int mOut, ii, mPlus;
	UINT prevSize, nextSize, thisSize;
	vector<float> *which;
	bool gotTime=false;
	high_resolution_clock::time_point startTime, endTime;
	int timeCountDown=10;

	//float toDivideRMS=data.learningRate/(float)batchSize;
	while(!threadExit) {

		//for(int i=0;i<outputsIndex;++i) {
		//	ii=i+1;
		//	fill((*data.NNlayersQ)[ii].outerDeltaB.begin(),(*data.NNlayersQ)[ii].outerDeltaB.end(),0.0);
		//	fill((*data.NNlayersQ)[i].outerDeltaW.begin(),(*data.NNlayersQ)[i].outerDeltaW.end(),0.0);
		//}

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
				cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nextSize, batchSize, thisSize, 1.0, &(*data.NNlayersQ)[i].weightsMatrix[0], nextSize, &(*which)[0], thisSize, 0.0, &(*data.NNlayersQ)[ii].atNeuronInputs[0], nextSize);
				transform((*data.NNlayersQ)[ii].counterN.begin(),(*data.NNlayersQ)[ii].counterN.end(),(*data.NNlayersQ)[ii].atNeuronOutputs.begin(),forwardFeed_helper(&(*data.NNlayersQ)[ii].atNeuronInputs[0],&(*data.NNlayersQ)[ii].biases[0]));
				//transform(std::make_counting_iterator(0),counting_iterator((*data.NNlayersQ)[ii].allN),(*data.NNlayersQ)[ii].atNeuronOutputs.begin(),forwardFeed_helper(&(*data.NNlayersQ)[ii].atNeuronInputs[0],&(*data.NNlayersQ)[ii].biases[0]));
				which=&(*data.NNlayersQ)[ii].atNeuronOutputs;
			}

			//Backward propagation
			mOut=outputsIndex-1;
			mPlus=outputsIndex;
			prevSize=hiddenMatrix[mOut];
			transform((*data.NNlayersQ)[outputsIndex].counterN.begin(),(*data.NNlayersQ)[outputsIndex].counterN.end(),(*data.NNlayersQ)[outputsIndex].innerDeltaB.begin(),output_helper(&(*data.NNlayersQ)[outputsIndex].atNeuronOutputs[0],&(*data.NNlayersQ)[outputsIndex].atNeuronInputs[0],&(*data.NNlayersQ)[outputsIndex].innerDeltaB[0],&(*data.labels)[0]));
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, numOutputs, prevSize, batchSize, 1.0, &(*data.NNlayersQ)[outputsIndex].innerDeltaB[0], numOutputs, &(*data.NNlayersQ)[mOut].atNeuronOutputs[0], prevSize, 0.0, &(*data.NNlayersQ)[mOut].innerDeltaW[0], numOutputs);

			--mOut;
			for(int i=outputsIndex-1;i;--i) {
				thisSize=hiddenMatrix[i];
				nextSize=hiddenMatrix[i+1];
				prevSize=hiddenMatrix[i-1];
				cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, thisSize, batchSize, nextSize, 1.0, &(*data.NNlayersQ)[i].weightsMatrix[0], nextSize, &(*data.NNlayersQ)[i+1].innerDeltaB[0], nextSize, 0.0, &(*data.NNlayersQ)[i].innerDeltaB[0], thisSize);
				if(i!=1) {
					transform((*data.NNlayersQ)[i].counterN.begin(),(*data.NNlayersQ)[i].counterN.end(),(*data.NNlayersQ)[i].innerDeltaB.begin(),backProp_helper2(&(*data.NNlayersQ)[i].atNeuronOutputs[0],&(*data.NNlayersQ)[i].innerDeltaB[0]));
					cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &(*data.NNlayersQ)[i].innerDeltaB[0], thisSize, &(*data.NNlayersQ)[i-1].atNeuronOutputs[0], prevSize, 0.0, &(*data.NNlayersQ)[mOut].innerDeltaW[0], thisSize);
				} else {
					transform((*data.NNlayersQ)[i].counterN.begin(),(*data.NNlayersQ)[i].counterN.end(),(*data.NNlayersQ)[i].innerDeltaB.begin(),backProp_helper(&(*data.NNlayersQ)[i].innerDeltaB[0],&(*data.NNlayersQ)[i].atNeuronInputs[0]));
					cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, thisSize, prevSize, batchSize, 1.0, &(*data.NNlayersQ)[i].innerDeltaB[0], thisSize, &(*data.data)[0], prevSize, 0.0, &(*data.NNlayersQ)[mOut].innerDeltaW[0], thisSize);
				}
				--mOut;
				--mPlus;
			}
			//for(int i=0;i<outputsIndex;++i) {
			//	ii=i+1;
			//	transform((*data.NNlayersQ)[ii].innerDeltaB.begin(),(*data.NNlayersQ)[ii].innerDeltaB.end(),(*data.NNlayersQ)[ii].outerDeltaB.begin(),(*data.NNlayersQ)[ii].outerDeltaB.begin(),plus<float>());
			//	transform((*data.NNlayersQ)[i].innerDeltaW.begin(),(*data.NNlayersQ)[i].innerDeltaW.end(),(*data.NNlayersQ)[i].outerDeltaW.begin(),(*data.NNlayersQ)[i].outerDeltaW.begin(),plus<float>());
			//}
			//for(int i=0;i<outputsIndex;++i) {
			//	ii=i+1;
			//	for_each((*data.NNlayersQ)[i].counterW.begin(),(*data.NNlayersQ)[i].counterW.end(),update_w(&(*data.NNlayersQ)[i].weightsMatrix[0],&(*data.NNlayersQ)[i].innerDeltaW[0],toDivideRMS));
			//	for_each((*data.NNlayersQ)[ii].counterN.begin(),(*data.NNlayersQ)[ii].counterN.end(),update_b(&(*data.NNlayersQ)[ii].biases[0],&(*data.NNlayersQ)[ii].innerDeltaB[0],toDivideRMS));
			//}
			//for(int i=0;i<outputsIndex;++i) {
			//	for_each(make_counting_iterator(0),make_counting_iterator(&(*data.NNlayersQ)[i].allW),update_w(&(*data.NNlayersQ)[i].weightsMatrix[0],&(*data.outerDeltaW)[i][0],toDivideRMS));
			//	for_each(make_counting_iterator(0),make_counting_iterator(&(*data.NNlayersQ)[i+1].allN),update_b(&(*data.NNlayersQ)[i+1].biases[0],&(*data.outerDeltaB)[i][0],toDivideRMS));
			//}
			if(!myID && !my_rank) {
				if(!gotTime) {
					if(timeCountDown) {
						--timeCountDown;
					} else {
						endTime=high_resolution_clock::now();
						float seconds=duration_cast<microseconds>(endTime-startTime).count()/1000000.0;
						printf("Update time interval approximately %.5f seconds apart(%.5f seconds per)\n",(seconds*(float)howMany)+1.0,seconds);
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
}*/