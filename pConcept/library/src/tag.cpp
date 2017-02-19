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

#include "../include/imebra/tag.h"
#include "../include/imebra/dataSet.h"
#include "../implementation/dataImpl.h"
#include "../implementation/dataHandlerNumericImpl.h"

namespace imebra
{

Tag::~Tag()
{
}

Tag::Tag(std::shared_ptr<imebra::implementation::data> pData): m_pData(pData)
{
}

size_t Tag::getBuffersCount() const
{
    return m_pData->getBuffersCount();
}

bool Tag::bufferExists(size_t bufferId) const
{
    return m_pData->bufferExists(bufferId);
}

size_t Tag::getBufferSize(size_t bufferId) const
{
    return m_pData->getBufferSize(bufferId);
}

ReadingDataHandler* Tag::getReadingDataHandler(size_t bufferId) const
{
    return new ReadingDataHandler(m_pData->getReadingDataHandler(bufferId));
}

WritingDataHandler* Tag::getWritingDataHandler(size_t bufferId)
{
    return new WritingDataHandler(m_pData->getWritingDataHandler(bufferId));
}

ReadingDataHandlerNumeric* Tag::getReadingDataHandlerNumeric(size_t bufferId) const
{
    std::shared_ptr<implementation::handlers::readingDataHandlerNumericBase> numericHandler =
            std::dynamic_pointer_cast<implementation::handlers::readingDataHandlerNumericBase>(m_pData->getReadingDataHandler(bufferId));
    if(numericHandler.get() == 0)
    {
        throw std::bad_cast();
    }
    return new ReadingDataHandlerNumeric(numericHandler);
}

ReadingDataHandlerNumeric* Tag::getReadingDataHandlerRaw(size_t bufferId) const
{
    std::shared_ptr<implementation::handlers::readingDataHandlerNumericBase> numericHandler = m_pData->getReadingDataHandlerRaw(bufferId);
    return new ReadingDataHandlerNumeric(numericHandler);
}

WritingDataHandlerNumeric* Tag::getWritingDataHandlerNumeric(size_t bufferId)
{
    std::shared_ptr<implementation::handlers::writingDataHandlerNumericBase> numericHandler =
            std::dynamic_pointer_cast<implementation::handlers::writingDataHandlerNumericBase>(m_pData->getWritingDataHandler(bufferId));
    if(numericHandler.get() == 0)
    {
        throw std::bad_cast();
    }
    return new WritingDataHandlerNumeric(numericHandler);
}

WritingDataHandlerNumeric* Tag::getWritingDataHandlerRaw(size_t bufferId)
{
    std::shared_ptr<implementation::handlers::writingDataHandlerNumericBase> numericHandler = m_pData->getWritingDataHandlerRaw(bufferId);
    return new WritingDataHandlerNumeric(numericHandler);
}

StreamReader* Tag::getStreamReader(size_t bufferId)
{
    return new StreamReader(m_pData->getStreamReader(bufferId));
}

StreamWriter* Tag::getStreamWriter(size_t bufferId)
{
    return new StreamWriter(m_pData->getStreamWriter(bufferId));
}

DataSet* Tag::getSequenceItem(size_t dataSetId) const
{
    return new DataSet(m_pData->getSequenceItem(dataSetId));
}

bool Tag::sequenceItemExists(size_t dataSetId) const
{
    return m_pData->dataSetExists(dataSetId);
}

void Tag::setSequenceItem(size_t dataSetId, const DataSet& dataSet)
{
    m_pData->setSequenceItem(dataSetId, dataSet.m_pDataSet);
}

void Tag::appendSequenceItem(const DataSet& dataSet)
{
    m_pData->appendDataSet(dataSet.m_pDataSet);
}

tagVR_t Tag::getDataType() const
{
    return m_pData->getDataType();
}

}
