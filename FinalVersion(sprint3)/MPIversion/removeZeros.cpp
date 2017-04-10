#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <bitset>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits.h>
#include <float.h>
#include <sys/stat.h>
#include <dirent.h>
using namespace std;
using namespace chrono;
using nanoSec = std::chrono::nanoseconds;
#define GetCurrentDir getcwd

vector<char> inVector;
vector<char> outVector;
#define ULLI unsigned long long int
#define UNCHAR unsigned char
#define ZERO_THRESHOLD 10
#define OUTLIER_MIN_COUNT 20
//#define OUTLIER_MIN_COUNT 0
unordered_map<int,UNCHAR> toRemove;
vector<string> types;
ULLI fsize;
ULLI newSize;
char CurrentPath[1000];
string slash = "/";

void printTime(high_resolution_clock::time_point start, high_resolution_clock::time_point end) {
	float seconds=duration_cast<microseconds>(end-start).count()/1000000.0;
	cout << "Processing time (milliseconds): " << duration_cast<milliseconds>(end - start).count() << endl;
	cout << "Processing time (microseconds): " << duration_cast<microseconds>(end - start).count() << endl;
	cout << "Processing time (nanoseconds): " << duration_cast<nanoseconds>(end - start).count() << endl;
	printf("Processing time (seconds): %.04f\n",seconds);
	printf("Processing time (minutes): %.04f\n",seconds/60.0f);
	printf("Processing time (hours): %.04f\n",seconds/3600.0f);
}

void getFile(string filename) {
	filename="KagglePre/"+filename;
	ifstream inFile(filename,ios::in | ios::binary);
   	if (inFile.is_open()) {
       	inFile.read((char*)&inVector[0],fsize);
       	inFile.close();
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

void putFile(string filename) {
	//filename="stage1Preprocessed2/"+filename;
	//filename+="2";//"_ones_and_zeros";
	cout << filename << endl;
	fstream inFile(filename,ios::out | ios::binary);
   	if (inFile.is_open()) {
       	//inFile.write((char*)&outVector[0],newSize);
       	inFile.write((char*)&toRemove[0],newSize*sizeof(int));
       	inFile.close();
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
			chdir(CurrentPath);
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
				chdir(CurrentPath);
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

int main(int argc, char *argv[]) {

	high_resolution_clock::time_point startTime, endTime;
	startTime=high_resolution_clock::now();

	if (!GetCurrentDir(CurrentPath, sizeof(CurrentPath))) {
		cout << "Couldn't retrieve current directory\n";
		return 0;
	}


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
		if(outliers[i]>OUTLIER_MIN_COUNT) {
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
   	endTime=high_resolution_clock::now();
   	printTime(startTime,endTime);
   	return 0;



	//ChangeDir("..");
	/*ChangeDir("KagglePre");
	vector<string> patients;
	GetFilesInDirectory(patients,types);
	int numPatients=patients.size();
	sort(patients.begin(), patients.end());
	ChangeDir("..");

	ifstream inFile("KagglePre/"+patients[0],ios::in | ios::binary | ios::ate);
	if (inFile.is_open()) {
		fsize = inFile.tellg();
	  	inFile.close();
	} else {
		cout << "what?\n";
		return 0;
	}

	inVector=vector<char>(fsize);
	vector<char> zeros(fsize,0);
	vector<char> ones(fsize,1);
	fill(ones.begin(),ones.end(),1);
	int count=0, ocount=0;

	for(auto p:patients) {
		cout << p;
		getFile(p);
		for(int i=0;i<fsize;++i) {
			if(inVector[i]==49) {
				zeros[i]=1;
			} else {
				if(inVector[i]==48) {
					ones[i]=0;
				}
			}
		}
		count=0;
		ocount=0;
		for(int i=0;i<fsize;++i) {
			if(!zeros[i]) {
				++count;
			}
			if(ones[i]) {
				++ocount;
			}
		}//*/
		/*cout << " zero count now: " << count << " one count now: " << ocount << " fsize can be reduced to: " << (fsize-ocount-count) << endl;
		//cout << " zero count now: " << count << " fsize can be reduced to: " << (fsize-count) << endl;
	}
	index=0;
	count=0;
	//vector<int> toRemove;
	for(auto z:zeros) {
		if(!z) {
			++count;
			//cout << "index: " << index << " can be removed\n";
			toRemove.push_back(index);
		}
		++index;
	}
	cout << "toRemove size: " << toRemove.size() << endl;
	newSize=toRemove.size();
	putFile("indexesThatCanBeRemoved.dat");
	/*for(auto p:patients) {
		//cout << p << endl;
		inVector=vector<char>(fsize);
		getFile(p);
		for(int i=toRemove.size()-1;i>-1;--i) {
			inVector.erase(inVector.begin()+toRemove[i]);
			//cout << "erasing " << toRemove[i] << endl;
		}
		newSize=inVector.size();
		putFile(p);
	}
	//cout << "total removed: " << count << " new size: " << (fsize-count) << endl;*/

	return 0;
}