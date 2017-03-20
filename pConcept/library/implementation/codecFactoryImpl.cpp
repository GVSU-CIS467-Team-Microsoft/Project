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

/*! \file codecFactory.cpp
    \brief Implementation of the codecFactory class.

*/

#include "codecFactoryImpl.h"
#include "configurationImpl.h"
#include "exceptionImpl.h"
#include "streamReaderImpl.h"
#include "codecImpl.h"
#include "jpegCodecImpl.h"
#include "dicomCodecImpl.h"
#include "../include/imebra/exceptions.h"

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
// Force the creation of the codec factory before main()
//  starts.
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
static codecFactory::forceCodecFactoryCreation forceCreation;


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Constructor
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
codecFactory::codecFactory(): m_maximumImageWidth(MAXIMUM_IMAGE_WIDTH), m_maximumImageHeight(MAXIMUM_IMAGE_HEIGHT)
{
    IMEBRA_FUNCTION_START();

    registerCodec(std::make_shared<dicomCodec>());
    registerCodec(std::make_shared<jpegCodec>());


    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Register a codec
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void codecFactory::registerCodec(std::shared_ptr<codec> pCodec)
{
    IMEBRA_FUNCTION_START();

	if(pCodec == 0)
	{
		return;
	}

	m_codecsList.push_back(pCodec);

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve a codec that can handle the specified
//  transfer syntax
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<codec> codecFactory::getCodec(const std::string& transferSyntax)
{
    IMEBRA_FUNCTION_START();

	std::shared_ptr<codecFactory> pFactory(getCodecFactory());

	for(std::list<std::shared_ptr<codec> >::iterator scanCodecs=pFactory->m_codecsList.begin(); scanCodecs!=pFactory->m_codecsList.end(); ++scanCodecs)
	{
		if((*scanCodecs)->canHandleTransferSyntax(transferSyntax))
		{
			return (*scanCodecs)->createCodec();
		}
	}

    IMEBRA_THROW(DataSetUnknownTransferSyntaxError, "None of the codecs support the specified transfer syntax");

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Retrieve the only instance of the codec factory
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<codecFactory> codecFactory::getCodecFactory()
{
    IMEBRA_FUNCTION_START();

    // Violation to requirement REQ_MAKE_SHARED due to protected constructor
    static std::shared_ptr<codecFactory> m_codecFactory(new codecFactory());

	return m_codecFactory;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Load the data from the specified stream and build a
//  dicomSet structure
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<dataSet> codecFactory::load(std::shared_ptr<streamReader> pStream, std::uint32_t maxSizeBufferLoad /* = 0xffffffff */)
{
    IMEBRA_FUNCTION_START();

	// Copy the list of codecs in a local list so we don't have
	//  to lock the object for a long time
	///////////////////////////////////////////////////////////
	std::list<std::shared_ptr<codec> > localCodecsList;
	std::shared_ptr<codecFactory> pFactory(getCodecFactory());
	{
		for(std::list<std::shared_ptr<codec> >::iterator scanCodecs=pFactory->m_codecsList.begin(); scanCodecs!=pFactory->m_codecsList.end(); ++scanCodecs)
		{
			std::shared_ptr<codec> copyCodec((*scanCodecs)->createCodec());
			localCodecsList.push_back(copyCodec);
		}
	}

	std::shared_ptr<dataSet> pDataSet;
	for(std::list<std::shared_ptr<codec> >::iterator scanCodecs=localCodecsList.begin(); scanCodecs != localCodecsList.end() && pDataSet == 0; ++scanCodecs)
	{
		try
		{
			return (*scanCodecs)->read(pStream, maxSizeBufferLoad);
		}
        catch(CodecWrongFormatError& /* e */)
		{
            exceptionsManagerGetter::getExceptionsManagerGetter().getExceptionsManager().getMessage(); // Reset the messages stack
			continue;
		}
	}

	if(pDataSet == 0)
	{
        IMEBRA_THROW(CodecWrongFormatError, "none of the codecs recognized the file format");
	}

	return pDataSet;

	IMEBRA_FUNCTION_END();
}


void codecFactory::setMaximumImageSize(const uint32_t maximumWidth, const uint32_t maximumHeight)
{
    IMEBRA_FUNCTION_START();

    m_maximumImageWidth = maximumWidth;
    m_maximumImageHeight = maximumHeight;

    IMEBRA_FUNCTION_END();
}


std::uint32_t codecFactory::getMaximumImageWidth()
{
    return m_maximumImageWidth;
}

std::uint32_t codecFactory::getMaximumImageHeight()
{
    return m_maximumImageHeight;
}

} // namespace codecs

} // namespace implementation

} // namespace imebra

