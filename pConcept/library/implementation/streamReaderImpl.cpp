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

/*! \file streamReader.cpp
    \brief Implementation of the streamReader class.

*/

#include "streamReaderImpl.h"
#include <string.h>

namespace imebra
{

namespace implementation
{

///////////////////////////////////////////////////////////
//
// Constructor
//
///////////////////////////////////////////////////////////
streamReader::streamReader(std::shared_ptr<baseStreamInput> pControlledStream):
    streamController(0, 0),
    m_pControlledStream(pControlledStream),
    m_inBitsBuffer(0),
    m_inBitsNum(0)
{
}

streamReader::streamReader(std::shared_ptr<baseStreamInput> pControlledStream, size_t virtualStart, size_t virtualLength):
    streamController(virtualStart, virtualLength),
    m_pControlledStream(pControlledStream),
	m_inBitsBuffer(0),
	m_inBitsNum(0)
{
    IMEBRA_FUNCTION_START();

    if(virtualLength == 0)
    {
        IMEBRA_THROW(StreamEOFError, "Virtual stream with zero length");
    }

    IMEBRA_FUNCTION_END();
}


std::shared_ptr<baseStreamInput> streamReader::getControlledStream()
{
    return m_pControlledStream;
}

std::shared_ptr<streamReader> streamReader::getReader(size_t virtualLength)
{
    IMEBRA_FUNCTION_START();

    if(virtualLength == 0)
    {
        IMEBRA_THROW(StreamEOFError, "Virtual stream with zero length");
    }
    size_t currentPosition = position();
    if(currentPosition + virtualLength > m_virtualLength && m_virtualLength != 0)
    {
        virtualLength = m_virtualLength - currentPosition;
    }
    seekForward((std::uint32_t)virtualLength);
    return std::make_shared<streamReader>(m_pControlledStream, currentPosition + m_virtualStart, virtualLength);

    IMEBRA_FUNCTION_END();
}

///////////////////////////////////////////////////////////
//
// Returns true if the last byte has been read
//
///////////////////////////////////////////////////////////
bool streamReader::endReached()
{
    IMEBRA_FUNCTION_START();

    return (m_dataBufferCurrent == m_dataBufferEnd && fillDataBuffer() == 0);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Refill the data buffer
//
///////////////////////////////////////////////////////////
size_t streamReader::fillDataBuffer()
{
    IMEBRA_FUNCTION_START();

    size_t readBytes = fillDataBuffer(&(m_dataBuffer[0]), m_dataBuffer.size());
	if(readBytes == 0)
	{
        m_dataBufferCurrent = m_dataBufferEnd = 0;
		return 0;
	}
    m_dataBufferEnd = readBytes;
    m_dataBufferCurrent = 0;
	return readBytes;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Read data from the stream into the specified buffer
//
///////////////////////////////////////////////////////////
size_t streamReader::fillDataBuffer(std::uint8_t* pDestinationBuffer, size_t readLength)
{
    IMEBRA_FUNCTION_START();

    m_dataBufferStreamPosition = position();
	if(m_virtualLength != 0)
	{
        if(m_dataBufferStreamPosition >= m_virtualLength)
		{
			m_dataBufferStreamPosition = m_virtualLength;
			return 0;
		}
        if(m_dataBufferStreamPosition + readLength > m_virtualLength)
		{
            readLength = m_virtualLength - m_dataBufferStreamPosition;
		}
	}
    return m_pControlledStream->read(m_dataBufferStreamPosition + m_virtualStart, pDestinationBuffer, readLength);

    IMEBRA_FUNCTION_END();
}



///////////////////////////////////////////////////////////
//
// Return the specified number of bytes from the stream
//
///////////////////////////////////////////////////////////
void streamReader::read(std::uint8_t* pBuffer, size_t bufferLength)
{
    IMEBRA_FUNCTION_START();

    while(bufferLength != 0)
	{
		// Update the data buffer if it is empty
		///////////////////////////////////////////////////////////
        if(m_dataBufferCurrent == m_dataBufferEnd)
		{
            if(bufferLength >= m_dataBuffer.size())
			{
				// read the data directly into the destination buffer
				///////////////////////////////////////////////////////////
                size_t readBytes(fillDataBuffer(pBuffer, bufferLength));

                m_dataBufferCurrent = m_dataBufferEnd = 0;
				m_dataBufferStreamPosition += readBytes;
				pBuffer += readBytes;
				bufferLength -= readBytes;
				if(readBytes == 0)
				{
                    IMEBRA_THROW(StreamEOFError, "Attempt to read past the end of the file");
				}
				continue;
			}

			if(fillDataBuffer() == 0)
			{
                IMEBRA_THROW(StreamEOFError, "Attempt to read past the end of the file");
			}
		}

		// Copy the available data into the return buffer
		///////////////////////////////////////////////////////////
        size_t copySize = bufferLength;
        size_t maxSize = (size_t)(m_dataBufferEnd - m_dataBufferCurrent);
		if(copySize > maxSize)
		{
			copySize = maxSize;
		}
        ::memcpy(pBuffer, &(m_dataBuffer[m_dataBufferCurrent]), copySize);
		bufferLength -= copySize;
		pBuffer += copySize;
        m_dataBufferCurrent += copySize;
	}

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Seek the read position
//
///////////////////////////////////////////////////////////
void streamReader::seek(size_t newPosition)
{
    IMEBRA_FUNCTION_START();

    // The requested position is already in the data buffer?
	///////////////////////////////////////////////////////////
    size_t bufferEndPosition = m_dataBufferStreamPosition + m_dataBufferEnd;
    if(newPosition >= m_dataBufferStreamPosition && newPosition < bufferEndPosition)
	{
        m_dataBufferCurrent = newPosition - m_dataBufferStreamPosition;
		return;
	}

	// The requested position is not in the data buffer
	///////////////////////////////////////////////////////////
    m_dataBufferCurrent = m_dataBufferEnd = 0;
    m_dataBufferStreamPosition = newPosition;

    IMEBRA_FUNCTION_END();
}

void streamReader::seekForward(std::uint32_t newPosition)
{
    IMEBRA_FUNCTION_START();

    size_t finalPosition = position() + newPosition;

    seek(finalPosition);

    IMEBRA_FUNCTION_END();
}

} // namespace implementation

} // namespace imebra
