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

/*! \file modalityVOILUT.cpp
    \brief Implementation of the class modalityVOILUT.

*/

#include "exceptionImpl.h"
#include "modalityVOILUTImpl.h"
#include "dataSetImpl.h"
#include "colorTransformsFactoryImpl.h"
#include <math.h>
#include <limits>

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
// Modality VOILUT transform
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
modalityVOILUT::modalityVOILUT(std::shared_ptr<const dataSet> pDataSet):
        m_pDataSet(pDataSet), m_voiLut(0), m_rescaleIntercept(pDataSet->getDouble(0x0028, 0, 0x1052, 0, 0, 0)), m_rescaleSlope(1.0), m_bEmpty(true)

{
    IMEBRA_FUNCTION_START();

	// Only monochrome images can have the modality voi-lut
	///////////////////////////////////////////////////////
    const std::string colorSpace(pDataSet->getString(0x0028, 0x0, 0x0004, 0, 0));

	if(!colorTransforms::colorTransformsFactory::isMonochrome(colorSpace))
	{
		return;
	}

    try
    {
        std::shared_ptr<handlers::readingDataHandler> rescaleHandler(m_pDataSet->getReadingDataHandler(0x0028, 0, 0x1053, 0x0));
        m_rescaleSlope = rescaleHandler->getDouble(0);
        m_bEmpty = false;
    }
    catch(const MissingDataElementError&)
    {
        try
        {
            m_voiLut = pDataSet->getLut(0x0028, 0x3000, 0);
            m_bEmpty = m_voiLut->getSize() == 0;
        }
        catch(const MissingDataElementError&)
        {
            // Nothing to do. Transform is empty
        }

    }

    IMEBRA_FUNCTION_END();
}

bool modalityVOILUT::isEmpty() const
{
	return m_bEmpty;
}


std::shared_ptr<image> modalityVOILUT::allocateOutputImage(
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

	// LUT
	///////////////////////////////////////////////////////////
    if(m_voiLut != 0 && m_voiLut->getSize() != 0)
	{
		std::uint8_t bits(m_voiLut->getBits());

        bitDepth_t depth;
        if(bits > 8)
        {
            depth = bitDepth_t::depthU16;
        }
        else
        {
            depth = bitDepth_t::depthU8;
        }

        return std::make_shared<image>(outputWidth, outputHeight, depth, inputColorSpace, bits - 1);
	}

	// Rescale
	///////////////////////////////////////////////////////////
    if(fabs(m_rescaleSlope) <= std::numeric_limits<double>::denorm_min())
	{
        return std::make_shared<image>(outputWidth, outputHeight, inputDepth, inputColorSpace, inputHighBit);
	}

	std::int32_t value0 = 0;
    std::int32_t value1 = ((std::int32_t)1 << (inputHighBit + 1)) - 1;
    if(inputDepth == bitDepth_t::depthS16 || inputDepth == bitDepth_t::depthS8)
	{
        value0 = ((std::int32_t)(-1) << inputHighBit);
        value1 = ((std::int32_t)1 << inputHighBit);
	}
    std::int32_t finalValue0((std::int32_t) ((double)value0 * m_rescaleSlope + m_rescaleIntercept) );
    std::int32_t finalValue1((std::int32_t) ((double)value1 * m_rescaleSlope + m_rescaleIntercept) );

	std::int32_t minValue, maxValue;
	if(finalValue0 < finalValue1)
	{
		minValue = finalValue0;
		maxValue = finalValue1;
	}
	else
	{
		minValue = finalValue1;
		maxValue = finalValue0;
	}

	if(minValue >= 0 && maxValue <= 255)
	{
        return std::make_shared<image>(outputWidth, outputHeight, bitDepth_t::depthU8, inputColorSpace, 7);
	}
	if(minValue >= -128 && maxValue <= 127)
	{
        return std::make_shared<image>(outputWidth, outputHeight, bitDepth_t::depthS8, inputColorSpace, 7);
	}
	if(minValue >= 0 && maxValue <= 65535)
	{
        return std::make_shared<image>(outputWidth, outputHeight, bitDepth_t::depthU16, inputColorSpace, 15);
	}
	if(minValue >= -32768 && maxValue <= 32767)
	{
        return std::make_shared<image>(outputWidth, outputHeight, bitDepth_t::depthS16, inputColorSpace, 15);
	}

    return std::make_shared<image>(outputWidth, outputHeight, bitDepth_t::depthS32, inputColorSpace, 31);

    IMEBRA_FUNCTION_END();
}



} // namespace transforms

} // namespace implementation

} // namespace imebra
