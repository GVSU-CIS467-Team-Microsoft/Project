#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

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

int main(int argc, char *argv[]) {

	char o=49;
	char z=48;
	cout << o << " -- " << z << endl;
	return 0;
	
	char temp[500];
	vector<float> pred[5];
	vector<string> pats[5];
	int ii=0;
	for(int i=3;i<5;++i) {
		string filename="submissionFile";
		sprintf(temp,"%03d",i+1);
		filename+=string(temp);
		if(i==4) {
			filename+="(3)";
		}
		filename+=".csv";
		bool once=true;
		float f;
		ifstream iFile(filename,ios::in|ios::binary);
		if(iFile.is_open()) {
			//cout << filename << endl;
			while(iFile.getline(temp,500)) {
				if(!once) {
					vector<string> splitTemp=split(string(temp),",");
					pats[ii].push_back(splitTemp[0]);
					sscanf(splitTemp[1].c_str(),"%f",&f);
					pred[ii].push_back(f);
				} else {
					once=false;
				}
			}
			iFile.close();
		}
		++ii;
	}
	for(int i=1;i<2;++i) {
		for(int j=0;j<pred[0].size();++j) {
			pred[0][j]-=pred[i][j];
			pred[0][j]=-pred[0][j];
		}
	}
	for(int i=0;i<pats[0].size();++i) {
		cout << pats[0][i] << ": change: " << pred[0][i] << endl;
	}

	return 0;
}