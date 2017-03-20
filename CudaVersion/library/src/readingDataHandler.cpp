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

/*! \file dataHandler.cpp
    \brief Implementation of the classes ReadingDataHandler & WritingDataHandler.
*/

#include "../include/imebra/readingDataHandler.h"
#include "../implementation/dataHandlerImpl.h"
#include "../implementation/dataHandlerNumericImpl.h"
#include <cstring>

namespace imebra
{

ReadingDataHandler::~ReadingDataHandler()
{
}

ReadingDataHandler::ReadingDataHandler(std::shared_ptr<imebra::implementation::handlers::readingDataHandler> pDataHandler): m_pDataHandler(pDataHandler)
{}

size_t ReadingDataHandler::getSize() const
{
    return m_pDataHandler->getSize();
}

tagVR_t ReadingDataHandler::getDataType() const
{
    return m_pDataHandler->getDataType();
}

std::int32_t ReadingDataHandler::getSignedLong(size_t index) const
{
    return m_pDataHandler->getSignedLong(index);
}

std::uint32_t ReadingDataHandler::getUnsignedLong(size_t index) const
{
    return m_pDataHandler->getUnsignedLong(index);
}

double ReadingDataHandler::getDouble(size_t index) const
{
    return m_pDataHandler->getDouble(index);
}

std::string ReadingDataHandler::getString(size_t index) const
{
    return m_pDataHandler->getString(index);
}

std::wstring ReadingDataHandler::getUnicodeString(size_t index) const
{
    return m_pDataHandler->getUnicodeString(index);
}

Date ReadingDataHandler::getDate(size_t index) const
{
    std::uint32_t year, month, day, hour, minutes, seconds, nanoseconds;
    std::int32_t offsetHours, offsetMinutes;
    m_pDataHandler->getDate(index, &year, &month, &day, &hour, &minutes, &seconds, &nanoseconds, &offsetHours, &offsetMinutes);

    return Date(
                (unsigned int)year,
                (unsigned int)month,
                (unsigned int)day,
                (unsigned int)hour,
                (unsigned int)minutes,
                (unsigned int)seconds,
                (unsigned int)nanoseconds,
                (int)offsetHours,
                (int)offsetMinutes);
}

Age ReadingDataHandler::getAge(size_t index) const
{
    imebra::ageUnit_t ageUnits;
    std::uint32_t age = m_pDataHandler->getAge(index, &ageUnits);
    return Age(age, ageUnits);
}

}
