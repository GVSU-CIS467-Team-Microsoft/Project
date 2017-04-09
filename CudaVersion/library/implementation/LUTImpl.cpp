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

/*! \file LUT.cpp
    \brief Implementation of the class lut.

*/

#include "exceptionImpl.h"
#include "LUTImpl.h"
#include "dataHandlerNumericImpl.h"
#include "bufferImpl.h"
#include "../include/imebra/exceptions.h"

#include <string.h>

namespace imebra
{

namespace implementation
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Create a LUT from a data handler
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
lut::lut(std::shared_ptr<handlers::readingDataHandlerNumericBase> pDescriptor, std::shared_ptr<handlers::readingDataHandlerNumericBase> pData, const std::wstring& description, bool signedData):
    m_size(0),
      m_firstMapped(0),
      m_bits(0)

{
    IMEBRA_FUNCTION_START();

    if(pDescriptor->getSize() != 3)
    {
        IMEBRA_THROW(LutCorruptedError, "The LUT is corrupted");
    }

    m_size = descriptorSignedToUnsigned(pDescriptor->getSignedLong(0));

    m_firstMapped = (std::int32_t)pDescriptor->getUnsignedLong(1);
    if(signedData)
    {
        if(m_firstMapped >= 32768)
        {
            m_firstMapped -= 65536;
        }
    }

    m_bits = (std::uint8_t)pDescriptor->getUnsignedLong(2);

    // If the LUT descriptor says 8 bit but it is actually 16 bits
    // then correct the descriptor's information
    //////////////////////////////////////////////////////////////
    if(m_bits <= 8 && pData->getUnitSize() * pData->getSize() == m_size * 2)
    {
        m_bits = 16;
    }

    // More than 16 bits per element are not allowed
    ////////////////////////////////////////////////
    if(m_bits > 16)
    {
        IMEBRA_THROW(LutCorruptedError, "The LUT items cannot be more than 16 bit wide");
    }

    // If 8 bits are stored in 16 bit elements, then extract them
    /////////////////////////////////////////////////////////////
    if(m_bits <= 8 && pData->getUnitSize() == 2)
    {
        std::shared_ptr<buffer> temporaryBuffer(std::make_shared<buffer>());
        std::shared_ptr<handlers::writingDataHandlerNumericBase> writingHandler(temporaryBuffer->getWritingDataHandlerNumeric(tagVR_t::OB, m_size));
        for(size_t scanData(0); scanData != pData->getSize(); ++scanData)
        {
            std::uint32_t data = pData->getUnsignedLong(scanData);
            writingHandler->setUnsignedLong(scanData * 2, data & 0xff);
            writingHandler->setUnsignedLong(scanData * 2 + 1, data >> 8);
        }
        writingHandler.reset();
        pData = temporaryBuffer->getReadingDataHandlerNumeric(tagVR_t::OB);
    }

    if(m_size != pData->getSize())
    {
        IMEBRA_THROW(LutCorruptedError, "The LUT is corrupted");
    }

    m_pDataHandler = pData;

    m_description = description;

    IMEBRA_FUNCTION_END();
}


std::shared_ptr<handlers::readingDataHandlerNumericBase> lut::getReadingDataHandler() const
{
    return m_pDataHandler;
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Destructor
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
lut::~lut()
{
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Convert a signed value in the lut descriptor into an
//  unsigned value
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t lut::descriptorSignedToUnsigned(std::int32_t signedValue)
{
    IMEBRA_FUNCTION_START();

    if(signedValue == 0)
    {
        return 0x010000;
    }
    else if(signedValue < 0)
    {
        return (signedValue & 0x0FFFF);
    }
    else
    {
        return (std::uint32_t)signedValue;
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the number of values stored into the LUT
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t lut::getSize() const
{
	return m_size;
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the first mapped index
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::int32_t lut::getFirstMapped() const
{
	return m_firstMapped;
}


std::uint32_t lut::getMappedValue(std::int32_t index) const
{
    IMEBRA_FUNCTION_START();

    if(index < m_firstMapped)
    {
        index = m_firstMapped;
    }

    std::uint32_t correctedIndex = (std::uint32_t)(index - m_firstMapped);
    if(correctedIndex >= m_size)
    {
        correctedIndex = m_size - 1;
    }
    return m_pDataHandler->getUnsignedLong(correctedIndex);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the lut's description.
// The returned value must be removed by the client
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::wstring lut::getDescription() const
{
	return m_description;
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Return the lut's bits
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint8_t lut::getBits() const
{
	return m_bits;
}


palette::palette(std::shared_ptr<lut> red, std::shared_ptr<lut> green, std::shared_ptr<lut> blue):
m_redLut(red), m_greenLut(green), m_blueLut(blue)
{}

void palette::setLuts(std::shared_ptr<lut> red, std::shared_ptr<lut> green, std::shared_ptr<lut> blue)
{
    IMEBRA_FUNCTION_START();

    m_redLut = red;
	m_greenLut = green;
	m_blueLut = blue;

    IMEBRA_FUNCTION_END();
}

std::shared_ptr<lut> palette::getRed() const
{
	return m_redLut;
}

std::shared_ptr<lut> palette::getGreen() const
{
	return m_greenLut;
}

std::shared_ptr<lut> palette::getBlue() const
{
	return m_blueLut;
}

} // namespace implementation

} // namespace imebra