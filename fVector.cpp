/*******************************************************************************
 * Author(s): Reese De Wind and Mark Jannenga
 * Version: 0.1
 * Created: Wed Feb  19 20:36:05 2017
 *******************************************************************************/

/*
  testing was compiled with the following command:
  g++ fVector.cpp -std=c++11 -o fVector -L/home/jannengm/workspace/CIS467/artifacts -limebra -Wl,-rpath,/home/jannengm/workspace/CIS467/artifacts

  artifacts must be created when compiling imebra.
 */

/*Something is goofy with my imebra install, so I need to specify the full path*/
#include </home/jannengm/workspace/CIS467/Project/Imebra_files/library/include/imebra/imebra.h>
//#include <imebra/imebra.h>
#include <iostream>
#include <vector>

using namespace imebra;
using namespace std;

int main(int argc, char **argv){
    /*Generate an Imebra DataSet object from the .dcm file supplied as standard input*/
    unique_ptr<imebra::DataSet> loadedDataSet(imebra::CodecFactory::load(argv[1]));

    /*Load the image from the DataSet*/
    std::unique_ptr<imebra::Image> image(loadedDataSet->getImageApplyModalityTransform(0));

    /*Get the color space (should be MONOCHROME2 for Kaggle DICOM images)*/
    std::string colorSpace = image->getColorSpace();

    /*Get the size in pixels*/
    std::uint32_t width = image->getWidth();
    std::uint32_t height = image->getHeight();

    /*Output some basic info about the image*/
    std::cout << "Colorspace: " << colorSpace << std::endl;
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    /*Create a data handler for the image*/
    std::unique_ptr<imebra::ReadingDataHandlerNumeric> dataHandler(image->getReadingDataHandler());

    /*Create a 2D vector for easy access later*/
    std::int32_t sum = 0;
//    std::vector<vector<int32_t>> pixelData;
//    pixelData.resize(height);
//    for (int i = 0; i < height; ++i)
//    {
//        pixelData[i].resize(width);
//    }

    /*Calculate the average "color" of the image from -1024 for black to 1024 for white*/
    for(std::uint32_t scanY(0); scanY != height; ++scanY)
    {
        for(std::uint32_t scanX(0); scanX != width; ++scanX)
        {
//            pixelData[scanY][scanX] = dataHandler->getSignedLong(scanY * width + scanX);
            sum += dataHandler->getSignedLong(scanY * width + scanX);
        }
    }

    cout << "Average color: " << sum / (double)(height * width) << endl;

    /*Print ascii art representation the pixel image (zoom waaaay out)*/
//    for(int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            if(pixelData[i][j] >= 0){
//                cout << "# ";
//            }
//            else {
//                cout << "  ";
//            }
////            cout << pixelData[i][j] << " ";
//        }
//        cout << endl;
//    }

    return 0;
}