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

/*! \file dataHandlerNumeric.cpp
    \brief Implementation of the handler for the numeric tags.

*/

#include "dataHandlerNumericImpl.h"
#include "memoryImpl.h"
#include "bufferImpl.h"

namespace imebra
{

namespace implementation
{

namespace handlers
{

readingDataHandlerNumericBase::readingDataHandlerNumericBase(const std::shared_ptr<const memory>& parseMemory, tagVR_t dataType):
    readingDataHandler(dataType), m_pMemory(parseMemory)
{
}

const std::uint8_t* readingDataHandlerNumericBase::getMemoryBuffer() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemory->data();

    IMEBRA_FUNCTION_END();
}

size_t readingDataHandlerNumericBase::getMemorySize() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemory->size();

    IMEBRA_FUNCTION_END();
}

std::shared_ptr<const memory> readingDataHandlerNumericBase::getMemory() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemory;

    IMEBRA_FUNCTION_END();
}

void readingDataHandlerNumericBase::copyTo(std::shared_ptr<writingDataHandlerNumericBase> pDestination)
{
    IMEBRA_FUNCTION_START();

    imebra::implementation::handlers::writingDataHandlerNumericBase* pHandler(pDestination.get());
    if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<std::uint8_t>) ||
        dynamic_cast<imebra::implementation::handlers::writingDataHandlerNumeric<std::uint8_t>* >(pHandler) != 0)
    {
        copyTo((std::uint8_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<std::int8_t>))
    {
        copyTo((std::int8_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<std::uint16_t>))
    {
        copyTo((std::uint16_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<std::int16_t>))
    {
        copyTo((std::int16_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<std::uint32_t>))
    {
        copyTo((std::uint32_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<std::int32_t>))
    {
        copyTo((std::int32_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<float>))
    {
        copyTo((float*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::writingDataHandlerNumeric<double>))
    {
        copyTo((double*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else
    {
        IMEBRA_THROW(std::runtime_error, "Data type not valid");
    }

    IMEBRA_FUNCTION_END();
}


writingDataHandlerNumericBase::writingDataHandlerNumericBase(const std::shared_ptr<buffer> &pBuffer, const size_t initialSize, tagVR_t dataType, size_t unitSize):
    writingDataHandler(pBuffer, dataType, 0), m_pMemory(std::make_shared<memory>(initialSize * unitSize))
{
}

size_t writingDataHandlerNumericBase::getSize() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemory->size() / getUnitSize();

    IMEBRA_FUNCTION_END();
}

std::shared_ptr<memory> writingDataHandlerNumericBase::getMemory() const
{
    return m_pMemory;
}

// Set the buffer's size, in data elements
///////////////////////////////////////////////////////////
void writingDataHandlerNumericBase::setSize(const size_t elementsNumber)
{
    IMEBRA_FUNCTION_START();

    m_pMemory->resize(elementsNumber * getUnitSize());

    IMEBRA_FUNCTION_END();
}


writingDataHandlerNumericBase::~writingDataHandlerNumericBase()
{
    if(m_buffer != 0)
    {
        // The buffer's size must be an even number
        ///////////////////////////////////////////////////////////
        size_t memorySize = m_pMemory->size();
        if((memorySize & 0x1) != 0)
        {
            m_pMemory->resize(++memorySize);
            *(m_pMemory->data() + (memorySize - 1)) = m_paddingByte;
        }

        m_buffer->commit(m_pMemory);
    }
}

std::uint8_t* writingDataHandlerNumericBase::getMemoryBuffer() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemory->data();

    IMEBRA_FUNCTION_END();
}

size_t writingDataHandlerNumericBase::getMemorySize() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemory->size();

    IMEBRA_FUNCTION_END();
}

// Copy the data from another handler
///////////////////////////////////////////////////////////
void writingDataHandlerNumericBase::copyFrom(std::shared_ptr<readingDataHandlerNumericBase> pSource)
{
    IMEBRA_FUNCTION_START();

    imebra::implementation::handlers::readingDataHandlerNumericBase* pHandler(pSource.get());
    if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<std::uint8_t>) ||
        dynamic_cast<imebra::implementation::handlers::readingDataHandlerNumeric<std::uint8_t>* >(pHandler) != 0)
    {
        copyFrom((std::uint8_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<std::int8_t>))
    {
        copyFrom((std::int8_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<std::uint16_t>))
    {
        copyFrom((std::uint16_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<std::int16_t>))
    {
        copyFrom((std::int16_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<std::uint32_t>))
    {
        copyFrom((std::uint32_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<std::int32_t>))
    {
        copyFrom((std::int32_t*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<float>))
    {
        copyFrom((float*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else if(typeid(*pHandler) == typeid(imebra::implementation::handlers::readingDataHandlerNumeric<double>))
    {
        copyFrom((double*)pHandler->getMemoryBuffer(), pHandler->getSize());
    }
    else
    {
        IMEBRA_THROW(std::runtime_error, "Data type not valid");
    }

    IMEBRA_FUNCTION_END();

}


}

}

}
