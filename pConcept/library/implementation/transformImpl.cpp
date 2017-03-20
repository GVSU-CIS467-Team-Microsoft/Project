/*

Imebra 4.0.8.1 changeset b15762068bd2

Imebra: a C++ Dicom library

Copyright (c) 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016
by Paolo Brandoli/Binarno s.p.

All rights reserved.

This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License version 2 as published by
 the Free Software Foundation.

This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

-------------------

If you want to use Imebra commercially then you have to buy the commercial
 license available at http://imebra.com

After you buy the commercial license then you can use Imebra according
 to the terms described in the Imebra Commercial License.
A copy of the Imebra Commercial License is available at http://imebra.com.

Imebra is available at http://imebra.com

The author can be contacted by email at info@binarno.com or by mail at
 the following address:
 Binarno s.p., Paolo Brandoli
 Rakuseva 14
 1000 Ljubljana
 Slovenia



*/

/*! \file transform.cpp
    \brief Implementation of the base class used by the transforms.

*/

#include "exceptionImpl.h"
#include "transformImpl.h"
#include "imageImpl.h"
#include "transformHighBitImpl.h"
#include "../include/imebra/exceptions.h"

namespace imebra
{

namespace implementation
{

namespace transforms
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Declare an input parameter
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
bool transform::isEmpty() const
{
	return false;
}


void transform::runTransform(
            const std::shared_ptr<const image>& inputImage,
            std::uint32_t inputTopLeftX, std::uint32_t inputTopLeftY, std::uint32_t inputWidth, std::uint32_t inputHeight,
            const std::shared_ptr<image>& outputImage,
            std::uint32_t outputTopLeftX, std::uint32_t outputTopLeftY) const
{
    IMEBRA_FUNCTION_START();

    std::uint32_t inputImageWidth, inputImageHeight;
    inputImage->getSize(&inputImageWidth, &inputImageHeight);
    std::uint32_t outputImageWidth, outputImageHeight;
    outputImage->getSize(&outputImageWidth, &outputImageHeight);

    if(inputTopLeftX + inputWidth > inputImageWidth ||
        inputTopLeftY + inputHeight > inputImageHeight ||
        outputTopLeftX + inputWidth > outputImageWidth ||
        outputTopLeftY + inputHeight > outputImageHeight)
    {
        IMEBRA_THROW(TransformInvalidAreaError, "The input and/or output areas are invalid");
    }

    std::shared_ptr<handlers::readingDataHandlerNumericBase> inputHandler(inputImage->getReadingDataHandler());
	std::shared_ptr<palette> inputPalette(inputImage->getPalette());
    std::string inputColorSpace(inputImage->getColorSpace());
	std::uint32_t inputHighBit(inputImage->getHighBit());
    bitDepth_t inputDepth(inputImage->getDepth());

    std::shared_ptr<handlers::writingDataHandlerNumericBase> outputHandler(outputImage->getWritingDataHandler());
	std::shared_ptr<palette> outputPalette(outputImage->getPalette());
    std::string outputColorSpace(outputImage->getColorSpace());
	std::uint32_t outputHighBit(outputImage->getHighBit());
    bitDepth_t outputDepth(outputImage->getDepth());

	if(isEmpty())
	{
        std::shared_ptr<transformHighBit> emptyTransform(std::make_shared<transformHighBit>());
        emptyTransform->runTransformHandlers(inputHandler, inputDepth, inputImageWidth, inputColorSpace, inputPalette, inputHighBit,
											 inputTopLeftX, inputTopLeftY, inputWidth, inputHeight,
                                             outputHandler, outputDepth, outputImageWidth, outputColorSpace, outputPalette, outputHighBit,
											 outputTopLeftX, outputTopLeftY);
		return;
	}

    runTransformHandlers(inputHandler, inputDepth, inputImageWidth, inputColorSpace, inputPalette, inputHighBit,
		inputTopLeftX, inputTopLeftY, inputWidth, inputHeight,
        outputHandler, outputDepth, outputImageWidth, outputColorSpace, outputPalette, outputHighBit,
		outputTopLeftX, outputTopLeftY);

    IMEBRA_FUNCTION_END();
}




} // namespace transforms

} // namespace implementation

} // namespace imebra
