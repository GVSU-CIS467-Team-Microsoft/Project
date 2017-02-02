/*******************************************************************************
 * Author(s): Reese De Wind
 * Version: 0.0
 * Created: Wed Feb  1 20:36:05 2017
 *******************************************************************************/

/*
  testing was compiled with the following command:
  g++ imebraTest.cpp -std=c++11 -L/home/reese/Projects/Project/artifacts -limebra -Wl,-rpath,/home/reese/Projects/Project/artifacts

  artifacts must be created when compiling imebra.
 */
#include <imebra/imebra.h>
#include <iostream>
int main(int argc, char **argv){
  // loadedDataSet = unique_ptr<imebra::DataSet>(imebra::CodecFactory::load(argv[1]));
  std::unique_ptr<imebra::DataSet> loadedDataSet(imebra::CodecFactory::load(argv[1]));
  
  std::wstring patientNameCharacter = loadedDataSet->getUnicodeString(imebra::TagId(imebra::tagId_t::PatientName_0010_0010), 0);
  //std::wstring patientNameIdeographic = loadedDataSet->getUnicodeString(imebra::TagId(imebra::tagId_t::PatientName_0010_0010), 1);
  
  // // A patient's name can contain up to 5 values, representing different interpretations of the same name
  // // (e.g. alphabetic representation, ideographic representation and phonetic representation)
  // // Here we retrieve the first 2 interpretations (index 0 and 1)
  std::wstring patientNameCharacter2 = loadedDataSet->getUnicodeString(imebra::TagId(0x10, 0x10), 0);
  // std::wstring patientNameIdeographic2 = loadedDataSet->getUnicodeString(imebra::TagId(0x10, 0x10), 1);

  //  std::wcout << patientNameCharacter << "\n";
  // // Retrieve the first image (index = 0)
  std::unique_ptr<imebra::Image> image(loadedDataSet->getImageApplyModalityTransform(0));

  // // Get the color space
  std::string colorSpace = image->getColorSpace();

  // // Get the size in pixels
  std::uint32_t width = image->getWidth();
  std::uint32_t height = image->getHeight();

  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;

  /*-------------------------READ PIXEL DATA - SLOW VERSION----------------------------------*/

  // let's assume that we already have the image's size in the variables width and height
  // (see previous code snippet)

  // Retrieve the data handler
  std::unique_ptr<imebra::ReadingDataHandlerNumeric> dataHandler(image->getReadingDataHandler());

  for(std::uint32_t scanY(0); scanY != height; ++scanY)
    {
      for(std::uint32_t scanX(0); scanX != width; ++scanX)
  	{
  	  // For monochrome images
  	  std::int32_t luminance = dataHandler->getSignedLong(scanY * width + scanX);

  	  // // For RGB images
  	  // std::int32_t r = dataHandler->getSignedLong((scanY * width + scanX) * 3);
  	  // std::int32_t g = dataHandler->getSignedLong((scanY * width + scanX) * 3 + 1);
  	  // std::int32_t b = dataHandler->getSignedLong((scanY * width + scanX) * 3 + 2);
  	}
    }

  /*-------------------------READ PIXEL DATA - FAST VERSION----------------------------------*/
  // Retrieve the data handler
  std::unique_ptr<imebra::ReadingDataHandlerNumeric> dataHandler2(image->getReadingDataHandler());

  // Get the memory pointer and the size (in bytes)
  size_t dataLength;
  const char* data = dataHandler->data(&dataLength);

  // Get the number of bytes per each value (1, 2, or 4 for images)
  size_t bytesPerValue = dataHandler->getUnitSize();

  // Are the values signed?
  bool bIsSigned = dataHandler->isSigned();

  // Do something with the pixels...A template function would come handy 
  /*----------------------------------------------------------------------------------------*/

 
  return 0;
}

