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

/*! \file codec.cpp
    \brief Implementation of the base class for the codecs.

*/

#include "exceptionImpl.h"
#include "streamReaderImpl.h"
#include "streamWriterImpl.h"
#include "codecImpl.h"
#include "dataSetImpl.h"
#include "codecFactoryImpl.h"
#include "../include/imebra/exceptions.h"
#include <string.h>


namespace imebra
{

namespace implementation
{

namespace codecs
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read a stream and write it into a dataset.
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<dataSet> codec::read(std::shared_ptr<streamReader> pSourceStream, std::uint32_t maxSizeBufferLoad /* = 0xffffffff */)
{
    IMEBRA_FUNCTION_START();

	// Reset the codec's bits buffer
	///////////////////////////////////////////////////////////
	pSourceStream->resetInBitsBuffer();

	// Store the stream's position
	///////////////////////////////////////////////////////////
    size_t position = pSourceStream->position();

	// Create a new dataset
	///////////////////////////////////////////////////////////
    std::shared_ptr<dataSet> pDestDataSet(std::make_shared<dataSet>());

	// Read the stream
	///////////////////////////////////////////////////////////
	try
	{
		readStream(pSourceStream, pDestDataSet, maxSizeBufferLoad);
	}
    catch(CodecWrongFormatError&)
	{
        pSourceStream->seek(position);
        IMEBRA_RETHROW("Detected a wrong format. Rewinding file");
	}

	// Update the charsets in the tags
	///////////////////////////////////////////////////////////
	pDestDataSet->updateTagsCharset();

	return pDestDataSet;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write a dataset into a stream.
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void codec::write(std::shared_ptr<streamWriter> pDestStream, std::shared_ptr<dataSet> pSourceDataSet)
{
    IMEBRA_FUNCTION_START();

	// Update charsets tag
	///////////////////////////////////////////////////////////
	pSourceDataSet->updateCharsetTag();

	pDestStream->resetOutBitsBuffer();
	writeStream(pDestStream, pSourceDataSet);
	pDestStream->flushDataBuffer();

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Allocate a channel's memory
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void channel::allocate(std::uint32_t width, std::uint32_t height)
{
    IMEBRA_FUNCTION_START();

    m_width = width;
    m_height = height;
    m_bufferSize = width * height;
    m_memory = std::make_shared<memory>(m_bufferSize * sizeof(std::int32_t) );
    m_pBuffer = (std::int32_t*)(m_memory->data());

    ::memset(m_pBuffer, 0, m_bufferSize * sizeof(std::int32_t));

    IMEBRA_FUNCTION_END();
}



} // namespace codecs

} // namespace implementation

} // namespace imebra

