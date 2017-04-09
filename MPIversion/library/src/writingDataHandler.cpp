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

#include "../include/imebra/writingDataHandler.h"
#include "../implementation/dataHandlerImpl.h"
#include "../implementation/dataHandlerNumericImpl.h"
#include <cstring>

namespace imebra
{

WritingDataHandler::~WritingDataHandler()
{
}

WritingDataHandler::WritingDataHandler(std::shared_ptr<imebra::implementation::handlers::writingDataHandler> pDataHandler): m_pDataHandler(pDataHandler)
{}

void WritingDataHandler::setSize(size_t elementsNumber)
{
    m_pDataHandler->setSize(elementsNumber);
}

size_t WritingDataHandler::getSize() const
{
    return m_pDataHandler->getSize();
}

tagVR_t WritingDataHandler::getDataType() const
{
    return m_pDataHandler->getDataType();
}


void WritingDataHandler::setDate(size_t index, const Date& date)
{
    m_pDataHandler->setDate(
        (std::uint32_t)index,
        (std::uint32_t)date.year,
        (std::uint32_t)date.month,
        (std::uint32_t)date.day,
        (std::uint32_t)date.hour,
        (std::uint32_t)date.minutes,
        (std::uint32_t)date.seconds,
        (std::uint32_t)date.nanoseconds,
        (std::int32_t)date.offsetHours,
        (std::int32_t)date.offsetMinutes);
}

void WritingDataHandler::setAge(size_t index, const Age& age)
{
    m_pDataHandler->setAge(index, age.age, age.units);
}

void WritingDataHandler::setSignedLong(size_t index, std::int32_t value)
{
    m_pDataHandler->setSignedLong(index, value);
}

void WritingDataHandler::setUnsignedLong(size_t index, std::uint32_t value)
{
    m_pDataHandler->setUnsignedLong(index, value);
}

void WritingDataHandler::setDouble(size_t index, double value)
{
    m_pDataHandler->setDouble(index, value);
}

void WritingDataHandler::setString(size_t index, const std::string& value)
{
    m_pDataHandler->setString(index, value);
}

void WritingDataHandler::setUnicodeString(size_t index, const std::wstring& value)
{
    m_pDataHandler->setUnicodeString(index, value);
}

}