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

/*! \file streamWriter.cpp
    \brief Implementation of the streamWriter class.

*/

#include "streamWriterImpl.h"
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
streamWriter::streamWriter(std::shared_ptr<baseStreamOutput> pControlledStream):
    streamController(0, 0),
    m_pControlledStream(pControlledStream),
    m_outBitsBuffer(0),
    m_outBitsNum(0)
{
}


///////////////////////////////////////////////////////////
//
// Constructor
//
///////////////////////////////////////////////////////////
streamWriter::streamWriter(std::shared_ptr<baseStreamOutput> pControlledStream, size_t virtualStart, size_t virtualLength):
    streamController(virtualStart, virtualLength),
    m_pControlledStream(pControlledStream),
	m_outBitsBuffer(0),
	m_outBitsNum(0)
{
}


///////////////////////////////////////////////////////////
//
// Destructor
//
///////////////////////////////////////////////////////////
streamWriter::~streamWriter()
{
	flushDataBuffer();
}


///////////////////////////////////////////////////////////
//
// Flush the data buffer
//
///////////////////////////////////////////////////////////
void streamWriter::flushDataBuffer()
{
    IMEBRA_FUNCTION_START();

    if(m_dataBufferCurrent == 0)
	{
		return;
	}
    m_pControlledStream->write(m_dataBufferStreamPosition + m_virtualStart, m_dataBuffer.data(), m_dataBufferCurrent);
    m_dataBufferStreamPosition += m_dataBufferCurrent;
    m_dataBufferCurrent = 0;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Write into the stream
//
///////////////////////////////////////////////////////////
void streamWriter::write(const std::uint8_t* pBuffer, size_t bufferLength)
{
    IMEBRA_FUNCTION_START();

    while(bufferLength != 0)
	{
        if(m_dataBufferCurrent == m_dataBuffer.size())
		{
			flushDataBuffer();
            if(bufferLength > (size_t)(m_dataBuffer.size() - m_dataBufferCurrent) )
            {
                m_pControlledStream->write(m_dataBufferStreamPosition + m_virtualStart, pBuffer, bufferLength);
                m_dataBufferStreamPosition += bufferLength;
                return;
            }
		}
        size_t copySize = (size_t)(m_dataBuffer.size() - m_dataBufferCurrent);
		if(copySize > bufferLength)
		{
			copySize = bufferLength;
		}
        ::memcpy(&(m_dataBuffer[m_dataBufferCurrent]), pBuffer, copySize);
		pBuffer += copySize;
		bufferLength -= copySize;
        m_dataBufferCurrent += copySize;
        m_dataBufferEnd = m_dataBufferCurrent;
	}

    IMEBRA_FUNCTION_END();
}

} // namespace implementation

} // namespace imebra
