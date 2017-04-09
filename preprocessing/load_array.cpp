/*******************************************************************************
 * Author(s): Reese De Wind
 * Version: 0.0
 * Created: Tue Feb 28 12:01:47 2017
 *******************************************************************************/

//compiled with g++ load_array.cpp -Wall -std=gnu++11
//tested on linux mint 18 "sarah" (cinnamon)
#include <sstream>
#include <string>
#include <iterator>
#include <vector>
#include <regex>
#include <fstream>
#include <fstream>
#include <iostream>

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

  return flattened;
}

std::vector<char> read_binary(std::string filename){
  std::ifstream input (filename, std::ifstream::binary);
  if(input){
    //get file length
    input.seekg (0, input.end);
    int length = input.tellg();
    input.seekg (0, input.beg);

    //char* arr = new char[length];
    std::vector<char> arr(length);
    input.read(&arr[0], length);
    if(input){
      std::cout << "File " << filename  << " was read successfully" << std::endl;
    }else{
      std::cout << "ERROR! File " << filename << " couldn't be read, only: " << input.gcount() << " bytes could be read!" << std::endl;
    }

    // for (std::vector<char>::const_iterator i = arr.begin(); i != arr.end(); ++i){
    //   std::cout << *i << ' ';
    // }
    return arr;
  }

  return std::vector<char>();
}

std::vector<char> read_numbered_file(std::string parent_directory, int file_num){
  std::string name = (new std::string(parent_directory))->append("/patient_" + std::to_string(file_num));// + ".dat");
  //printf("name: %s\n", name.c_str());
  std::vector<short> orig = read(name + ".dat");
  std::vector<char> bin = read_binary(name + ".bin");
  std::vector<uint8_t> bin_better(bin.size());
  memcpy(&bin_better[0], &bin[0], bin.size());
  std::cout << "size of original: " << orig.size() << ", size of bin_better: " << bin_better.size() << std::endl;
  // int failed = 0;
  // for(unsigned int i = 0; i < bin_better.size(); i++){
  //   if((unsigned int)((unsigned int)bin_better[i] >> 7) != (unsigned int)(orig[i])){
  //     failed++;
  //   }
  // }
  // if(failed){
  //   std::cout << "something didn't match!: " << failed << std::endl;
  // }
  
  // std::cout << "first byte is: " << (unsigned int)bin_better[failed] << std::endl;
  // std::cout << "first few are: " << orig[failed] << "," << orig[failed+1] << "," << orig[failed+2] << "," << orig[failed+3] << "," << orig[failed+4] << "," << orig[failed+5] << "," << orig[failed+6] << "," << orig[failed+7] << std::endl;
  return bin;
    
}

//for testing
int main(int argc, char **argv){
  read("patient_data_0"); //file must be in same directory
  read_numbered_file("patient_data_0", 0);
}
