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

std::vector<std::vector<std::vector<int>>> read(std::string filename){
  
  std::vector<std::vector<std::vector<int>>> overall_array = std::vector<std::vector<std::vector<int>>>();
  int outer_index = -1;
  int inner_index = -1;
  std::ifstream input(filename);


  for( std::string line; getline( input, line ); ){
    std::regex ws_re(",");
    std::sregex_token_iterator it(line.begin(), line.end(), ws_re, -1);
    std::sregex_token_iterator reg_end;

    for(; it !=reg_end; ++it){
      if(it->str().compare("/1") == 0){
	outer_index++;;
	inner_index = -1;
	overall_array.push_back(std::vector<std::vector<int>>());
      }else if(it->str().compare("/2") == 0){
	inner_index++;
	overall_array.at(outer_index).push_back(std::vector<int>());
      }else{
	if(it->str().compare("") != 0){
	  overall_array.at(outer_index).at(inner_index).push_back(std::stoi(it->str()));
	}
      }
    }
  }
  //return overall_array;
  printf("overall_array size: %lu\n", overall_array.size());
  printf("overall_array first item size: %lu\n", overall_array.at(0).size());
  printf("overall_array first items first item size: %lu\n", overall_array.at(0).at(0).size());
}


//for testing
int main(int argc, char **argv){
  read("patient_0.dat");
}
