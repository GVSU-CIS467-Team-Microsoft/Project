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
	\brief Implementation of the class used to retrieve the codec able to
        handle the requested transfer syntax.

*/

#include "../include/imebra/fileStreamInput.h"
#include "../include/imebra/fileStreamOutput.h"
#include "../include/imebra/codecFactory.h"
#include "../include/imebra/definitions.h"

#include "../implementation/dicomCodecImpl.h"
#include "../implementation/jpegCodecImpl.h"
#include "../implementation/codecFactoryImpl.h"
#include "../implementation/codecImpl.h"
#include "../implementation/exceptionImpl.h"

namespace imebra
{

DataSet* CodecFactory::load(StreamReader& reader, size_t maxSizeBufferLoad /*  = std::numeric_limits<size_t>::max()) */)
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<imebra::implementation::codecs::codecFactory> factory(imebra::implementation::codecs::codecFactory::getCodecFactory());
    return new DataSet(factory->load(reader.m_pReader, (std::uint32_t)maxSizeBufferLoad));

    IMEBRA_FUNCTION_END();
}

DataSet* CodecFactory::load(const std::wstring& fileName, size_t maxSizeBufferLoad)
{
    IMEBRA_FUNCTION_START();

    FileStreamInput file(fileName);

    StreamReader reader(file);
    return load(reader, maxSizeBufferLoad);

    IMEBRA_FUNCTION_END();
}

DataSet* CodecFactory::load(const std::string& fileName, size_t maxSizeBufferLoad)
{
    IMEBRA_FUNCTION_START();

    FileStreamInput file(fileName);

    StreamReader reader(file);
    return load(reader, maxSizeBufferLoad);

    IMEBRA_FUNCTION_END();
}

void CodecFactory::saveImage(
        StreamWriter& destStream,
        const Image& sourceImage,
        const std::string& transferSyntax,
        imageQuality_t imageQuality,
        tagVR_t dataType,
        std::uint32_t allocatedBits,
        bool bSubSampledX,
        bool bSubSampledY,
        bool bInterleaved,
        bool b2Complement)
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<imebra::implementation::codecs::codecFactory> factory(imebra::implementation::codecs::codecFactory::getCodecFactory());
    std::shared_ptr<implementation::codecs::codec> pCodec = factory->getCodec(transferSyntax);
    pCodec->setImage(destStream.m_pWriter, sourceImage.m_pImage, transferSyntax, imageQuality, dataType, allocatedBits, bSubSampledX, bSubSampledY, bInterleaved, b2Complement);

    IMEBRA_FUNCTION_END();
}

void CodecFactory::setMaximumImageSize(const std::uint32_t maximumWidth, const std::uint32_t maximumHeight)
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<imebra::implementation::codecs::codecFactory> factory(imebra::implementation::codecs::codecFactory::getCodecFactory());
    factory->setMaximumImageSize(maximumWidth, maximumHeight);

    IMEBRA_FUNCTION_END();

}


void CodecFactory::save(const DataSet& dataSet, StreamWriter& writer, codecType_t codecType)
{
    IMEBRA_FUNCTION_START();

    std::shared_ptr<imebra::implementation::codecs::codec> codec;

    switch(codecType)
    {
    case codecType_t::jpeg:
        codec = std::make_shared<imebra::implementation::codecs::jpegCodec>();
        break;
    default:
        codec = std::make_shared<imebra::implementation::codecs::dicomCodec>();
        break;
    }

    codec->write(writer.m_pWriter, dataSet.m_pDataSet);

    IMEBRA_FUNCTION_END();
}

void CodecFactory::save(const DataSet &dataSet, const std::wstring& fileName, codecType_t codecType)
{
    IMEBRA_FUNCTION_START();

    FileStreamOutput file(fileName);

    StreamWriter writer(file);
    CodecFactory::save(dataSet, writer, codecType);

    IMEBRA_FUNCTION_END();
}

void CodecFactory::save(const DataSet &dataSet, const std::string& fileName, codecType_t codecType)
{
    IMEBRA_FUNCTION_START();

    FileStreamOutput file(fileName);

    StreamWriter writer(file);
    CodecFactory::save(dataSet, writer, codecType);

    IMEBRA_FUNCTION_END();
}


}
