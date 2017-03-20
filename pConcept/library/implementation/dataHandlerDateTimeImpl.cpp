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

/*! \file dataHandlerDateTime.cpp
    \brief Implementation of the dataHandlerDateTime class.

*/

#include <sstream>
#include <iomanip>
#include <stdlib.h>

#include "exceptionImpl.h"
#include "dataHandlerDateTimeImpl.h"

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
// dataHandlerDateTime
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

readingDataHandlerDateTime::readingDataHandlerDateTime(const memory& parseMemory): readingDataHandlerDateTimeBase(parseMemory, tagVR_t::DT)
{

}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// Retrieve the date
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void readingDataHandlerDateTime::getDate(const size_t index,
        std::uint32_t* pYear,
        std::uint32_t* pMonth,
        std::uint32_t* pDay,
        std::uint32_t* pHour,
        std::uint32_t* pMinutes,
        std::uint32_t* pSeconds,
        std::uint32_t* pNanoseconds,
		std::int32_t* pOffsetHours,
		std::int32_t* pOffsetMinutes) const
{
    IMEBRA_FUNCTION_START();

    std::string dateTimeString = getString(index);

    parseDate(dateTimeString.substr(0, 8), pYear, pMonth, pDay);

    if(dateTimeString.size() <= 8)
    {
        parseTime("", pHour, pMinutes, pSeconds, pNanoseconds, pOffsetHours, pOffsetMinutes);
    }
    else
    {
        parseTime(dateTimeString.substr(8), pHour, pMinutes, pSeconds, pNanoseconds, pOffsetHours, pOffsetMinutes);
    }

	IMEBRA_FUNCTION_END();
}


writingDataHandlerDateTime::writingDataHandlerDateTime(const std::shared_ptr<buffer> &pBuffer):
    writingDataHandlerDateTimeBase(pBuffer, tagVR_t::DT, 0, 54)
{

}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the date
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void writingDataHandlerDateTime::setDate(const size_t index,
        std::uint32_t year,
        std::uint32_t month,
        std::uint32_t day,
        std::uint32_t hour,
        std::uint32_t minutes,
        std::uint32_t seconds,
        std::uint32_t nanoseconds,
		std::int32_t offsetHours,
		std::int32_t offsetMinutes)
{
    IMEBRA_FUNCTION_START();

    std::string dateTimeString;
	dateTimeString = buildDate(year, month, day);
	dateTimeString += buildTime(hour, minutes, seconds, nanoseconds, offsetHours, offsetMinutes);
    setString(index, dateTimeString);

	IMEBRA_FUNCTION_END();
}


} // namespace handlers

} // namespace implementation

} // namespace imebra
