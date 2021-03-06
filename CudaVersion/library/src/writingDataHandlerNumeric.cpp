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

#include "../include/imebra/writingDataHandlerNumeric.h"
#include "../include/imebra/readingDataHandlerNumeric.h"
#include "../implementation/dataHandlerImpl.h"
#include "../implementation/dataHandlerNumericImpl.h"
#include <cstring>

namespace imebra
{

WritingDataHandlerNumeric::WritingDataHandlerNumeric(std::shared_ptr<implementation::handlers::writingDataHandlerNumericBase> pDataHandler):
    WritingDataHandler(pDataHandler)
{
}

WritingDataHandlerNumeric::~WritingDataHandlerNumeric()
{
}

ReadWriteMemory* WritingDataHandlerNumeric::getMemory() const
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    return new ReadWriteMemory(numericDataHandler->getMemory());
}

void WritingDataHandlerNumeric::assign(const char* source, size_t sourceSize)
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    numericDataHandler->getMemory()->assign((std::uint8_t*) source, sourceSize);
}

char* WritingDataHandlerNumeric::data(size_t* pDataSize) const
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    *pDataSize = numericDataHandler->getMemorySize();
    return (char*)numericDataHandler->getMemoryBuffer();
}

size_t WritingDataHandlerNumeric::data(char* destination, size_t destinationSize) const
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    size_t memorySize = numericDataHandler->getMemorySize();
    if(destination != 0 && destinationSize >= memorySize && memorySize != 0)
    {
        ::memcpy(destination, numericDataHandler->getMemoryBuffer(), memorySize);
    }
    return memorySize;
}

size_t WritingDataHandlerNumeric::getUnitSize() const
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->getUnitSize();
}

bool WritingDataHandlerNumeric::isSigned() const
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->isSigned();
}

bool WritingDataHandlerNumeric::isFloat() const
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->isFloat();
}

void WritingDataHandlerNumeric::copyFrom(const ReadingDataHandlerNumeric& source)
{
    std::shared_ptr<imebra::implementation::handlers::writingDataHandlerNumericBase> numericDataHandler = std::dynamic_pointer_cast<imebra::implementation::handlers::writingDataHandlerNumericBase>(m_pDataHandler);
    return numericDataHandler->copyFrom(std::dynamic_pointer_cast<imebra::implementation::handlers::readingDataHandlerNumericBase>(source.m_pDataHandler));
}

}
