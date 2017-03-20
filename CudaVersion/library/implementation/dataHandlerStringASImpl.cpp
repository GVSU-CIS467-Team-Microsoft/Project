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

/*! \file dataHandlerStringAS.cpp
    \brief Implementation of the class dataHandlerStringAS.

*/

#include <sstream>
#include <string>
#include <iomanip>
#include "../include/imebra/exceptions.h"
#include "exceptionImpl.h"
#include "dataHandlerStringASImpl.h"
#include "memoryImpl.h"
#include <memory.h>

namespace imebra
{

namespace implementation
{

namespace handlers
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// dataHandlerStringAS
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Constructor
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
readingDataHandlerStringAS::readingDataHandlerStringAS(const memory& parseMemory): readingDataHandlerString(parseMemory, tagVR_t::AS, '\\', 0x20)
{
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the age
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t readingDataHandlerStringAS::getAge(const size_t index, ageUnit_t* pUnit) const
{
    IMEBRA_FUNCTION_START();

    std::string ageString = getString(index);
    if(ageString.size() != 4)
    {
        IMEBRA_THROW(DataHandlerCorruptedBufferError, "The AGE string should be 4 bytes long but it is "<< ageString.size() << " bytes long");
    }
    std::istringstream ageStream(ageString);
	std::uint32_t age;
    if(!(ageStream >> age))
    {
        IMEBRA_THROW(DataHandlerCorruptedBufferError, "The AGE is not a number");
    }
    char unit = ageString[3];
    if(
            unit != (char)ageUnit_t::days &&
            unit != (char)ageUnit_t::weeks &&
            unit == (char)ageUnit_t::months &&
            unit == (char)ageUnit_t::years)
    {
        IMEBRA_THROW(DataHandlerCorruptedBufferError, "The AGE unit should be D, W, M or Y but is ascii code "<< (int)unit);
    }
    *pUnit = (ageUnit_t)unit;
    return age;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the age in years as a signed long
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::int32_t readingDataHandlerStringAS::getSignedLong(const size_t index) const
{
    IMEBRA_FUNCTION_START();

	return (std::int32_t)getDouble(index);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the age in years as an unsigned long
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t readingDataHandlerStringAS::getUnsignedLong(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    return (std::uint32_t)getDouble(index);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the age in years as a double
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
double readingDataHandlerStringAS::getDouble(const size_t /* index */) const
{
    IMEBRA_FUNCTION_START();

    IMEBRA_THROW(DataHandlerConversionError, "Cannot convert an Age to a number")

	IMEBRA_FUNCTION_END();
}


writingDataHandlerStringAS::writingDataHandlerStringAS(const std::shared_ptr<buffer> &pBuffer):
    writingDataHandlerString(pBuffer, tagVR_t::AS, '\\', 4, 4, 0x20)
{
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the age
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerStringAS::setAge(const size_t index, const std::uint32_t age, const ageUnit_t unit)
{
    IMEBRA_FUNCTION_START();

    if(index >= getSize())
    {
        setSize(index + 1);
    }

    std::ostringstream ageStream;
    ageStream << std::setfill('0');
    ageStream << std::setw(3) << age;
    ageStream << std::setw(1) << (char)unit;

    setString(index, ageStream.str());

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the age in years as a signed long
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerStringAS::setSignedLong(const size_t index, const std::int32_t value)
{
    IMEBRA_FUNCTION_START();

	setDouble(index, (double)value);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the age in years as an unsigned long
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerStringAS::setUnsignedLong(const size_t index, const std::uint32_t value)
{
    IMEBRA_FUNCTION_START();

	setDouble(index, (double)value);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the age in years as a double
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerStringAS::setDouble(const size_t /* index */, const double /* value */)
{
    IMEBRA_FUNCTION_START();

    IMEBRA_THROW(DataHandlerConversionError, "Cannot convert to VR AS from double");

	IMEBRA_FUNCTION_END();
}

void writingDataHandlerStringAS::validate() const
{
    IMEBRA_FUNCTION_START();

    memory parseMemory(m_strings[0].size());
    ::memcpy(parseMemory.data(), m_strings[0].data(), parseMemory.size());
    try
    {
        readingDataHandlerStringAS readingHandler(parseMemory);
    }
    catch(const DataHandlerCorruptedBufferError& e)
    {
        IMEBRA_THROW(DataHandlerConversionError, e.what());
    }
    writingDataHandlerString::validate();

    IMEBRA_FUNCTION_END();
}


} // namespace handlers

} // namespace implementation

} // namespace imebra
