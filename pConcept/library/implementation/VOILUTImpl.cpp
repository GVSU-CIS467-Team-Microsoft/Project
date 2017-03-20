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

/*! \file VOILUT.cpp
    \brief Implementation of the class VOILUT.

*/

#include "exceptionImpl.h"
#include "VOILUTImpl.h"
#include "dataSetImpl.h"
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
// Set the lut
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void VOILUT::setLUT(const std::shared_ptr<lut>& pLut)
{
	m_pLUT = pLut;
	m_windowCenter = 0;
	m_windowWidth = 0;
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the center/width
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void VOILUT::setCenterWidth(double center, double width)
{
	m_windowCenter = center;
	m_windowWidth = width;
    m_pLUT.reset();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Get the center/width
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void VOILUT::getCenterWidth(double* pCenter, double* pWidth)
{
	*pCenter = m_windowCenter;
	*pWidth = m_windowWidth;
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Returns true if the transform is empty
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
bool VOILUT::isEmpty() const
{
	return m_windowWidth <= 1 && (m_pLUT == 0 || m_pLUT->getSize() == 0);
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Allocate the output image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<image> VOILUT::allocateOutputImage(
        bitDepth_t inputDepth,
        const std::string& inputColorSpace,
        std::uint32_t inputHighBit,
        std::shared_ptr<palette> /* inputPalette */,
        std::uint32_t outputWidth, std::uint32_t outputHeight) const
{
    IMEBRA_FUNCTION_START();

    if(isEmpty())
	{
        return std::make_shared<image>(outputWidth, outputHeight, inputDepth, inputColorSpace, inputHighBit);
	}

	if(m_pLUT != 0 && m_pLUT->getSize() != 0)
	{
		std::uint8_t bits = m_pLUT->getBits();

        if(bits > 8)
        {
            inputDepth = bitDepth_t::depthU16;
        }
        else
        {
            inputDepth = bitDepth_t::depthU8;
        }
        return std::make_shared<image>(outputWidth, outputHeight, inputDepth, inputColorSpace, bits - 1);
	}

	//
	// LUT not found.
	// Use the window's center/width
	//
	///////////////////////////////////////////////////////////
	if(m_windowWidth <= 1)
	{
        return std::make_shared<image>(outputWidth, outputHeight, inputDepth, inputColorSpace, inputHighBit);
	}

    if(inputDepth == bitDepth_t::depthS8)
        inputDepth = bitDepth_t::depthU8;
    if(inputDepth == bitDepth_t::depthS16 || inputDepth == bitDepth_t::depthU32 || inputDepth == bitDepth_t::depthS32)
        inputDepth = bitDepth_t::depthU16;

    return std::make_shared<image>(outputWidth, outputHeight, inputDepth, inputColorSpace, inputHighBit);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Finds and applies the optimal VOI.
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void VOILUT::applyOptimalVOI(const std::shared_ptr<imebra::implementation::image>& inputImage, std::uint32_t inputTopLeftX, std::uint32_t inputTopLeftY, std::uint32_t inputWidth, std::uint32_t inputHeight)
{
    IMEBRA_FUNCTION_START();

    std::uint32_t width, height;
    inputImage->getSize(&width, &height);

    if(inputTopLeftX + inputWidth > width || inputTopLeftY + inputHeight > height)
    {
        IMEBRA_THROW(TransformInvalidAreaError, "The input and/or output areas are invalid");
    }

    std::shared_ptr<handlers::readingDataHandlerNumericBase> handler(inputImage->getReadingDataHandler());
    HANDLER_CALL_TEMPLATE_FUNCTION_WITH_PARAMS(templateFindOptimalVOI, handler, width, inputTopLeftX, inputTopLeftY, inputWidth, inputHeight);

    IMEBRA_FUNCTION_END();
}




} // namespace transforms

} // namespace implementation

} // namespace imebra
