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

/*! \file dataHandlerStringDS.cpp
    \brief Implementation of the class dataHandlerStringDS.

*/

#include <sstream>
#include <iomanip>

#include "exceptionImpl.h"
#include "dataHandlerStringDSImpl.h"

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
// dataHandlerStringDS
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

readingDataHandlerStringDS::readingDataHandlerStringDS(const memory& parseMemory): readingDataHandlerString(parseMemory, tagVR_t::DS, '\\', 0x20)
{
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Get the value as a signed long.
// Overwritten to use getDouble()
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::int32_t readingDataHandlerStringDS::getSignedLong(const size_t index) const
{
    IMEBRA_FUNCTION_START();

	return (std::int32_t)getDouble(index);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Get the value as an unsigned long.
// Overwritten to use getDouble()
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t readingDataHandlerStringDS::getUnsignedLong(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    return (std::uint32_t)getDouble(index);

	IMEBRA_FUNCTION_END();
}


writingDataHandlerStringDS::writingDataHandlerStringDS(const std::shared_ptr<buffer> pBuffer):
    writingDataHandlerString(pBuffer, tagVR_t::DS, '\\', 0, 16, 0x20)
{

}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the value as a signed long.
// Overwritten to use setDouble()
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerStringDS::setSignedLong(const size_t index, const std::int32_t value)
{
    IMEBRA_FUNCTION_START();

	setDouble(index, (double)value);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the value as an unsigned long.
// Overwritten to use setDouble()
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerStringDS::setUnsignedLong(const size_t index, const std::uint32_t value)
{
    IMEBRA_FUNCTION_START();

	setDouble(index, (double)value);

	IMEBRA_FUNCTION_END();
}

} // namespace handlers

} // namespace implementation

} // namespace imebra
