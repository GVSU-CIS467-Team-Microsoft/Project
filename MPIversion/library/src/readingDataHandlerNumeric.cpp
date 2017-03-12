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

#include "../include/imebra/readingDataHandlerNumeric.h"
#include "../include/imebra/writingDataHandlerNumeric.h"
#include "../implementation/dataHandlerImpl.h"
#include "../implementation/dataHandlerNumericImpl.h"
#include <cstring>

namespace imebra
{

ReadingDataHandlerNumeric::ReadingDataHandlerNumeric(std::shared_ptr<implementation::handlers::readingDataHandlerNumericBase> pDataHandler):
    ReadingDataHandler(pDataHandler)
{
}

ReadingDataHandlerNumeric::~ReadingDataHandlerNumeric()
{
}

ReadMemory* ReadingDataHandlerNumeric::getMemory() const
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    return new ReadMemory(numericDataHandler->getMemory());
}

size_t ReadingDataHandlerNumeric::data(char* destination, size_t destinationSize) const
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    size_t memorySize = numericDataHandler->getMemorySize();
    if(destination != 0 && destinationSize >= memorySize && memorySize != 0)
    {
        ::memcpy(destination, numericDataHandler->getMemoryBuffer(), memorySize);
    }
    return memorySize;
}

const char* ReadingDataHandlerNumeric::data(size_t* pDataSize) const
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    *pDataSize = numericDataHandler->getMemorySize();
    return (const char*)numericDataHandler->getMemoryBuffer();
}

size_t ReadingDataHandlerNumeric::getUnitSize() const
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->getUnitSize();
}

bool ReadingDataHandlerNumeric::isSigned() const
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->isSigned();
}

bool ReadingDataHandlerNumeric::isFloat() const
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->isFloat();
}

void ReadingDataHandlerNumeric::copyTo(const WritingDataHandlerNumeric& destination)
{
    std::shared_ptr<imebra::implementation::handlers::readingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->copyTo(std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(destination.m_pDataHandler));
}

}
