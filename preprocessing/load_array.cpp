/*******************************************************************************
 * Author(s): Reese De Wind
 * Version: 0.0
 * Created: Tue Feb 28 12:01:47 2017
 *******************************************************************************/

//compiled with g++ load_array.cpp -Wall -std=gnu++11
//tested on linux mint 18 "sarah" (cinnamon)
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <fstream>

std::vector<short> read(std::string filename){
  std::vector<short> flattened = std::vector<short>();
  short outer_index = -1;
  short inner_index = -1;
  std::ifstream input(filename);


  for( std::string line; getline( input, line ); ){
    std::regex ws_re(",");
    std::sregex_token_iterator it(line.begin(), line.end(), ws_re, -1);
    std::sregex_token_iterator reg_end;

    for(; it !=reg_end; ++it){
      if(it->str().compare("/1") == 0){
	outer_index++;;
	inner_index = -1;
      }else if(it->str().compare("/2") == 0){
	inner_index++;
      }else{
	if(it->str().compare("") != 0){
	  flattened.push_back(std::stoi(it->str()));
	}
      }
    }
  }
  //printf("flattened array size is: %lu bytes or %f megabytes", flattened.size() * sizeof(short), flattened.size() * sizeof(short) / 1024.0 / 1024.0);
  return flattened;
}

std::vector<short> read_numbered_file(std::string parent_directory, int file_num){
  std::string name = (new std::string(parent_directory))->append("/patient_" + std::to_string(file_num) + ".dat");
  //printf("name: %s\n", name.c_str());
  return read(name);
}

//for testing
int main(int argc, char **argv){
  //read("patient_0.dat"); //file must be in same directory
  read_numbered_file("patient_data", 0);
}
