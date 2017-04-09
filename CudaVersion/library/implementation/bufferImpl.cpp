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

/*! \file buffer.cpp
    \brief Implementation of the buffer class.

*/

#include "exceptionImpl.h"
#include "streamReaderImpl.h"
#include "streamWriterImpl.h"
#include "bufferImpl.h"
#include "bufferStreamImpl.h"
#include "dataHandlerImpl.h"
#include "dataHandlerNumericImpl.h"
#include "dataHandlerStringAEImpl.h"
#include "dataHandlerStringASImpl.h"
#include "dataHandlerStringCSImpl.h"
#include "dataHandlerStringDSImpl.h"
#include "dataHandlerStringISImpl.h"
#include "dataHandlerStringLOImpl.h"
#include "dataHandlerStringLTImpl.h"
#include "dataHandlerStringPNImpl.h"
#include "dataHandlerStringSHImpl.h"
#include "dataHandlerStringSTImpl.h"
#include "dataHandlerStringUCImpl.h"
#include "dataHandlerStringUIImpl.h"
#include "dataHandlerStringURImpl.h"
#include "dataHandlerStringUTImpl.h"
#include "dataHandlerDateImpl.h"
#include "dataHandlerDateTimeImpl.h"
#include "dataHandlerTimeImpl.h"
#include "dicomDictImpl.h"
#include "../include/imebra/exceptions.h"

#include <vector>


namespace imebra
{

namespace implementation
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// imebraBuffer
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
// Buffer's constructor
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
buffer::buffer():
    m_originalBufferPosition(0),
    m_originalBufferLength(0),
    m_originalWordLength(1),
    m_originalEndianType(streamController::lowByteEndian)
{
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Buffer's constructor (on demand content)
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
buffer::buffer(
        const std::shared_ptr<baseStreamInput>& originalStream,
        size_t bufferPosition,
        size_t bufferLength,
        size_t wordLength,
		streamController::tByteOrdering endianType):
		m_originalStream(originalStream),
		m_originalBufferPosition(bufferPosition),
		m_originalBufferLength(bufferLength),
		m_originalWordLength(wordLength),
        m_originalEndianType(endianType)
{
}


std::shared_ptr<const memory> buffer::getLocalMemory() const
{
    IMEBRA_FUNCTION_START();

    // If the object must be loaded from the original stream,
    //  then load it...
    ///////////////////////////////////////////////////////////
    if(m_originalStream != 0)
    {
        std::shared_ptr<memory> localMemory(std::make_shared<memory>(m_originalBufferLength));
        if(m_originalBufferLength != 0)
        {
            std::shared_ptr<streamReader> reader(std::make_shared<streamReader>(m_originalStream, m_originalBufferPosition, m_originalBufferLength));
            std::vector<std::uint8_t> localBuffer;
            localBuffer.resize(m_originalBufferLength);
            reader->read(&localBuffer[0], m_originalBufferLength);
            if(m_originalWordLength != 0)
            {
                reader->adjustEndian(&localBuffer[0], m_originalWordLength, m_originalEndianType, m_originalBufferLength/m_originalWordLength);
            }
            localMemory->assign(&localBuffer[0], m_originalBufferLength);
        }
        return localMemory;
    }

    if(m_memory == 0)
    {
        return std::make_shared<memory>();
    }

    return m_memory;

    IMEBRA_FUNCTION_END();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Create a data handler and connect it to the buffer
// (raw or normal)
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<handlers::readingDataHandler> buffer::getReadingDataHandler(tagVR_t tagVR) const
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    std::shared_ptr<const memory> localMemory(getLocalMemory());

    switch(tagVR)
    {
    case tagVR_t::AE:
        return std::make_shared<handlers::readingDataHandlerStringAE>(*localMemory);

    case tagVR_t::AS:
        return std::make_shared<handlers::readingDataHandlerStringAS>(*localMemory);

    case tagVR_t::CS:
        return std::make_shared<handlers::readingDataHandlerStringCS>(*localMemory);

    case tagVR_t::DS:
        return std::make_shared<handlers::readingDataHandlerStringDS>(*localMemory);

    case tagVR_t::IS:
        return std::make_shared<handlers::readingDataHandlerStringIS>(*localMemory);

    case tagVR_t::LO:
        return std::make_shared<handlers::readingDataHandlerStringLO>(*localMemory, m_charsetsList);

    case tagVR_t::LT:
        return std::make_shared<handlers::readingDataHandlerStringLT>(*localMemory, m_charsetsList);

    case tagVR_t::PN:
        return std::make_shared<handlers::readingDataHandlerStringPN>(*localMemory, m_charsetsList);

    case tagVR_t::SH:
        return std::make_shared<handlers::readingDataHandlerStringSH>(*localMemory, m_charsetsList);

    case tagVR_t::ST:
        return std::make_shared<handlers::readingDataHandlerStringST>(*localMemory, m_charsetsList);

    case tagVR_t::UC:
        return std::make_shared<handlers::readingDataHandlerStringUC>(*localMemory, m_charsetsList);

    case tagVR_t::UI:
        return std::make_shared<handlers::readingDataHandlerStringUI>(*localMemory);

    case tagVR_t::UR:
        return std::make_shared<handlers::readingDataHandlerStringUR>(*localMemory);

    case tagVR_t::UT:
        return std::make_shared< handlers::readingDataHandlerStringUT>(*localMemory, m_charsetsList);

    case tagVR_t::OB:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::uint8_t> >(localMemory, tagVR);

    case tagVR_t::OL:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::int32_t> >(localMemory, tagVR);

    case tagVR_t::SB:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::int8_t> >(localMemory, tagVR);

    case tagVR_t::UN:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::uint8_t> >(localMemory, tagVR);

    case tagVR_t::OW:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::uint16_t> >(localMemory, tagVR);

    case tagVR_t::AT:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::uint16_t> >(localMemory, tagVR);

    case tagVR_t::FL:
        return std::make_shared<handlers::readingDataHandlerNumeric<float> >(localMemory, tagVR);

    case tagVR_t::OF:
        return std::make_shared<handlers::readingDataHandlerNumeric<float> >(localMemory, tagVR);

    case tagVR_t::FD:
        return std::make_shared<handlers::readingDataHandlerNumeric<double> >(localMemory, tagVR);

    case tagVR_t::OD:
        return std::make_shared<handlers::readingDataHandlerNumeric<double> >(localMemory, tagVR);

    case tagVR_t::SL:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::int32_t> >(localMemory, tagVR);

    case tagVR_t::SS:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::int16_t> >(localMemory, tagVR);

    case tagVR_t::UL:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::uint32_t> >(localMemory, tagVR);

    case tagVR_t::US:
        return std::make_shared<handlers::readingDataHandlerNumeric<std::uint16_t> >(localMemory, tagVR);

    case tagVR_t::DA:
        return std::make_shared<handlers::readingDataHandlerDate>(*localMemory);

    case tagVR_t::DT:
        return std::make_shared<handlers::readingDataHandlerDateTime>(*localMemory);

    case tagVR_t::TM:
        return std::make_shared<handlers::readingDataHandlerTime>(*localMemory);

    case tagVR_t::SQ:
        IMEBRA_THROW(std::logic_error, "Cannot retrieve a SQ data handler");
    }

	IMEBRA_FUNCTION_END();
}

std::shared_ptr<handlers::writingDataHandler> buffer::getWritingDataHandler(tagVR_t tagVR, std::uint32_t size)
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    // Reset the pointer to the data handler
    ///////////////////////////////////////////////////////////
    std::shared_ptr<handlers::writingDataHandler> handler;

    switch(tagVR)
    {
    case tagVR_t::AE:
        return std::make_shared<handlers::writingDataHandlerStringAE>(shared_from_this());

    case tagVR_t::AS:
        return std::make_shared<handlers::writingDataHandlerStringAS>(shared_from_this());

    case tagVR_t::CS:
        return std::make_shared<handlers::writingDataHandlerStringCS>(shared_from_this());

    case tagVR_t::DS:
        return std::make_shared<handlers::writingDataHandlerStringDS>(shared_from_this());

    case tagVR_t::IS:
        return std::make_shared<handlers::writingDataHandlerStringIS>(shared_from_this());

    case tagVR_t::UR:
        return std::make_shared<handlers::writingDataHandlerStringUR>(shared_from_this());

    case tagVR_t::LO:
        return std::make_shared<handlers::writingDataHandlerStringLO>(shared_from_this(), m_charsetsList);

    case tagVR_t::LT:
        return std::make_shared<handlers::writingDataHandlerStringLT>(shared_from_this(), m_charsetsList);

    case tagVR_t::PN:
        return std::make_shared<handlers::writingDataHandlerStringPN>(shared_from_this(), m_charsetsList);

    case tagVR_t::SH:
        return std::make_shared<handlers::writingDataHandlerStringSH>(shared_from_this(), m_charsetsList);

    case tagVR_t::ST:
        return std::make_shared<handlers::writingDataHandlerStringST>(shared_from_this(), m_charsetsList);

    case tagVR_t::UC:
        return std::make_shared<handlers::writingDataHandlerStringUC>(shared_from_this(), m_charsetsList);

    case tagVR_t::UI:
        return std::make_shared<handlers::writingDataHandlerStringUI>(shared_from_this());

    case tagVR_t::UT:
        return std::make_shared< handlers::writingDataHandlerStringUT>(shared_from_this(), m_charsetsList);

    case tagVR_t::OB:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::uint8_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::OL:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::int32_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::SB:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::int8_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::UN:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::uint8_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::OW:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::uint16_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::AT:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::uint16_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::FL:
        return std::make_shared<handlers::writingDataHandlerNumeric<float> >(shared_from_this(), size, tagVR);

    case tagVR_t::OF:
        return std::make_shared<handlers::writingDataHandlerNumeric<float> >(shared_from_this(), size, tagVR);

    case tagVR_t::FD:
        return std::make_shared<handlers::writingDataHandlerNumeric<double> >(shared_from_this(), size, tagVR);

    case tagVR_t::OD:
        return std::make_shared<handlers::writingDataHandlerNumeric<double> >(shared_from_this(), size, tagVR);

    case tagVR_t::SL:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::int32_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::SS:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::int16_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::UL:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::uint32_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::US:
        return std::make_shared<handlers::writingDataHandlerNumeric<std::uint16_t> >(shared_from_this(), size, tagVR);

    case tagVR_t::DA:
        return std::make_shared<handlers::writingDataHandlerDate>(shared_from_this());

    case tagVR_t::DT:
        return std::make_shared<handlers::writingDataHandlerDateTime>(shared_from_this());

    case tagVR_t::TM:
        return std::make_shared<handlers::writingDataHandlerTime>(shared_from_this());

    case tagVR_t::SQ:
        IMEBRA_THROW(std::logic_error, "Cannot retrieve a SQ data handler");
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Get a reading stream for the buffer
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<streamReader> buffer::getStreamReader()
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

	// If the object must be loaded from the original stream,
	//  then return the original stream
	///////////////////////////////////////////////////////////
    if(m_originalStream != 0 && (m_originalWordLength <= 1 && m_originalEndianType != streamReader::getPlatformEndian()))
	{
        std::shared_ptr<streamReader> reader(std::make_shared<streamReader>(m_originalStream, m_originalBufferPosition, m_originalBufferLength));
		return reader;
	}

	// Build a stream from the buffer's memory
	///////////////////////////////////////////////////////////
    std::shared_ptr<streamReader> reader;
    std::shared_ptr<memoryStreamInput> memoryStream = std::make_shared<memoryStreamInput>(getLocalMemory());
    reader = std::shared_ptr<streamReader>(std::make_shared<streamReader>(memoryStream));

	return reader;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Get a writing stream for the buffer
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<streamWriter> buffer::getStreamWriter(tagVR_t tagVR)
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<handlers::writingDataHandlerRaw> tempHandlerRaw = getWritingDataHandlerRaw(tagVR);
    return std::make_shared<streamWriter>(std::make_shared<bufferStreamOutput>(tempHandlerRaw));

	IMEBRA_FUNCTION_END();
}



///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Create a raw data handler and connect it to the buffer
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<handlers::readingDataHandlerRaw> buffer::getReadingDataHandlerRaw(tagVR_t tagVR) const
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    return std::make_shared<handlers::readingDataHandlerRaw>(getLocalMemory(), tagVR);

	IMEBRA_FUNCTION_END();
}


std::shared_ptr<handlers::writingDataHandlerRaw> buffer::getWritingDataHandlerRaw(tagVR_t tagVR, std::uint32_t size)
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    return std::make_shared<handlers::writingDataHandlerRaw>(shared_from_this(), size, tagVR);

    IMEBRA_FUNCTION_END();
}


std::shared_ptr<handlers::readingDataHandlerNumericBase> buffer::getReadingDataHandlerNumeric(tagVR_t tagVR) const
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<handlers::readingDataHandler> handler = getReadingDataHandler(tagVR);
    std::shared_ptr<handlers::readingDataHandlerNumericBase> numericHandler = std::dynamic_pointer_cast<handlers::readingDataHandlerNumericBase>(handler);
    if(numericHandler == 0)
    {
        IMEBRA_THROW(DataHandlerConversionError, "The data handler does not handle numeric data");
    }

    return numericHandler;

    IMEBRA_FUNCTION_END();
}


std::shared_ptr<handlers::writingDataHandlerNumericBase> buffer::getWritingDataHandlerNumeric(tagVR_t tagVR, std::uint32_t size)
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<handlers::writingDataHandler> handler = getWritingDataHandler(tagVR, size);
    std::shared_ptr<handlers::writingDataHandlerNumericBase> numericHandler = std::dynamic_pointer_cast<handlers::writingDataHandlerNumericBase>(handler);
    if(numericHandler == 0)
    {
        IMEBRA_THROW(DataHandlerConversionError, "The data handler does not handle numeric data");
    }

    return numericHandler;

    IMEBRA_FUNCTION_END();
}



///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
// Return the buffer's size in bytes
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
size_t buffer::getBufferSizeBytes() const
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    // The buffer has not been loaded yet
	///////////////////////////////////////////////////////////
    if(m_originalStream != 0)
	{
		return m_originalBufferLength;
	}

	// The buffer has no memory
	///////////////////////////////////////////////////////////
	if(m_memory == 0)
	{
		return 0;
	}

	// Return the memory's size
	///////////////////////////////////////////////////////////
	return m_memory->size();

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
// Commit the changes made by copyBack
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void buffer::commit(std::shared_ptr<memory> newMemory, const charsetsList::tCharsetsList& newCharsetsList)
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    m_memory = newMemory;
    m_originalStream.reset();
    m_charsetsList = newCharsetsList;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
// Commit the changes made by copyBack
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void buffer::commit(std::shared_ptr<memory> newMemory)
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);

    m_memory = newMemory;
    m_originalStream.reset();

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set the charsets used by the buffer
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void buffer::setCharsetsList(const charsetsList::tCharsetsList& charsets)
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);
    m_charsetsList = charsets;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the charsets used by the buffer
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void buffer::getCharsetsList(charsetsList::tCharsetsList* pCharsetsList) const
{
    IMEBRA_FUNCTION_START();

    std::lock_guard<std::mutex> lock(m_mutex);
    pCharsetsList->insert(pCharsetsList->end(), m_charsetsList.begin(), m_charsetsList.end());

	IMEBRA_FUNCTION_END();
}


} // namespace implementation

} // namespace imebra
