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

/*! \file dicomCodec.cpp
    \brief Implementation of the class dicomCodec.

*/

#include <list>
#include <vector>
#include <string.h>
#include "exceptionImpl.h"
#include "streamReaderImpl.h"
#include "streamWriterImpl.h"
#include "memoryImpl.h"
#include "dicomCodecImpl.h"
#include "dataSetImpl.h"
#include "dicomDictImpl.h"
#include "imageImpl.h"
#include "colorTransformsFactoryImpl.h"
#include "codecFactoryImpl.h"
#include "bufferImpl.h"
#include "../include/imebra/exceptions.h"

namespace imebra
{

namespace implementation
{

namespace codecs
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// dicomCodec
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
// Create another DICOM codec
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::shared_ptr<codec> dicomCodec::createCodec()
{
    IMEBRA_FUNCTION_START();

    return std::shared_ptr<codec>(std::make_shared<dicomCodec>());

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write a Dicom stream
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writeStream(std::shared_ptr<streamWriter> pStream, std::shared_ptr<dataSet> pDataSet)
{
    IMEBRA_FUNCTION_START();

    // Retrieve the transfer syntax
    ///////////////////////////////////////////////////////////
    std::string transferSyntax = pDataSet->getString(0x0002, 0, 0x0010, 0, 0, "1.2.840.10008.1.2");

    // Adjust the flags
    ///////////////////////////////////////////////////////////
    bool bExplicitDataType = (transferSyntax != "1.2.840.10008.1.2");        // Implicit VR little endian

    // Explicit VR big endian
    ///////////////////////////////////////////////////////////
    streamController::tByteOrdering endianType = (transferSyntax == "1.2.840.10008.1.2.2") ? streamController::highByteEndian : streamController::lowByteEndian;

    // Write the dicom header
    ///////////////////////////////////////////////////////////
    std::uint8_t zeroBuffer[128];
    ::memset(zeroBuffer, 0L, 128L);
    pStream->write(zeroBuffer, 128);

    // Write the DICM signature
    ///////////////////////////////////////////////////////////
    pStream->write((std::uint8_t*)"DICM", 4);

    // Build the stream
    ///////////////////////////////////////////////////////////
    buildStream(pStream, pDataSet, bExplicitDataType, endianType, streamType_t::mediaStorage);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Build a dicom stream, without header and DICM signature
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::buildStream(std::shared_ptr<streamWriter> pStream, std::shared_ptr<dataSet> pDataSet, bool bExplicitDataType, streamController::tByteOrdering endianType, streamType_t streamType)
{
    IMEBRA_FUNCTION_START();

    dataSet::tGroupsIds groups = pDataSet->getGroups();

    for(dataSet::tGroupsIds::const_iterator scanGroups(groups.begin()), endGroups(groups.end()); scanGroups != endGroups; ++scanGroups)
    {
        size_t numGroups = pDataSet->getGroupsNumber(*scanGroups);
        for(size_t scanGroupsNumber(0); scanGroupsNumber != numGroups; ++scanGroupsNumber)
        {
            const dataSet::tTags& tags(pDataSet->getGroupTags(*scanGroups, scanGroupsNumber));

            // When writing a media storage file, the tag 0002,0001 must be 0,1 (OB)
            ////////////////////////////////////////////////////////////////////////
            if(streamType == streamType_t::mediaStorage && *scanGroups == 0x0002 && tags.find(0x0001) == tags.end())
            {
                dataSet::tTags temporaryTags(tags);
                charsetsList::tCharsetsList charsets;
                pDataSet->getCharsetsList(&charsets);
                std::shared_ptr<data> metaInformationTag(std::make_shared<data>(tagVR_t::OB, charsets));
                {
                    std::shared_ptr<handlers::writingDataHandler> handler(metaInformationTag->getWritingDataHandler(0));
                    handler->setUnsignedLong(0, 0);
                    handler->setUnsignedLong(1, 1);
                }
                temporaryTags[1] = metaInformationTag;
                writeGroup(pStream, temporaryTags, *scanGroups, bExplicitDataType, endianType);
                continue;
            }

            writeGroup(pStream, tags, *scanGroups, bExplicitDataType, endianType);
        }
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write a single data group
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writeGroup(std::shared_ptr<streamWriter> pDestStream, const dataSet::tTags& tags, std::uint16_t groupId, bool bExplicitDataType, streamController::tByteOrdering endianType)
{
    IMEBRA_FUNCTION_START();

    if(groupId == 2)
    {
        bExplicitDataType = true;
        endianType = streamController::lowByteEndian;
    }

    // Calculate the group's length
    ///////////////////////////////////////////////////////////
    std::uint32_t groupLength = getGroupLength(tags, bExplicitDataType);

    // Write the group's length
    ///////////////////////////////////////////////////////////
    const char lengthDataType[] = "UL";

    std::uint16_t adjustedGroupId = pDestStream->adjustEndian(groupId, endianType);;

    std::uint16_t tagId = 0;
    pDestStream->write((std::uint8_t*)&adjustedGroupId, 2);
    pDestStream->write((std::uint8_t*)&tagId, 2);

    if(bExplicitDataType)
    {
        pDestStream->write((std::uint8_t*)&lengthDataType, 2);
        std::uint16_t tagLengthWord = pDestStream->adjustEndian((std::uint16_t)4, endianType);;
        pDestStream->write((std::uint8_t*)&tagLengthWord, 2);
    }
    else
    {
        std::uint32_t tagLengthDword = pDestStream->adjustEndian((std::uint32_t)4, endianType);
        pDestStream->write((std::uint8_t*)&tagLengthDword, 4);
    }

    pDestStream->adjustEndian((std::uint8_t*)&groupLength, 4, endianType);
    pDestStream->write((std::uint8_t*)&groupLength, 4);

    // Write all the tags
    ///////////////////////////////////////////////////////////
    for(dataSet::tTags::const_iterator scanTags(tags.begin()), endTags(tags.end()); scanTags != endTags; ++scanTags)
    {
        std::uint16_t tagId = scanTags->first;
        if(tagId == 0)
        {
            continue;
        }
        pDestStream->write((std::uint8_t*)&adjustedGroupId, 2);
        writeTag(pDestStream, scanTags->second, tagId, bExplicitDataType, endianType);
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write a single tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writeTag(std::shared_ptr<streamWriter> pDestStream, std::shared_ptr<data> pData, std::uint16_t tagId, bool bExplicitDataType, streamController::tByteOrdering endianType)
{
    IMEBRA_FUNCTION_START();

    // Calculate the tag's length
    ///////////////////////////////////////////////////////////
    bool bSequence;
    std::uint32_t tagHeader;
    std::uint32_t tagLength = getTagLength(pData, bExplicitDataType, &tagHeader, &bSequence);

    // Prepare the identifiers for the sequence (adjust the
    //  endian)
    ///////////////////////////////////////////////////////////
    std::uint16_t sequenceItemGroup = 0xfffe;
    std::uint16_t sequenceItemDelimiter = 0xe000;
    std::uint16_t sequenceItemEnd = 0xe0dd;
    pDestStream->adjustEndian((std::uint8_t*)&sequenceItemGroup, 2, endianType);
    pDestStream->adjustEndian((std::uint8_t*)&sequenceItemDelimiter, 2, endianType);
    pDestStream->adjustEndian((std::uint8_t*)&sequenceItemEnd, 2, endianType);

    // Check the data type
    ///////////////////////////////////////////////////////////
    tagVR_t dataType = pData->getDataType();

    // Adjust the tag id endian and write it
    ///////////////////////////////////////////////////////////
    std::uint16_t adjustedTagId = tagId;
    pDestStream->adjustEndian((std::uint8_t*)&adjustedTagId, 2, endianType);
    pDestStream->write((std::uint8_t*)&adjustedTagId, 2);

    // Write the data type if it is explicit
    ///////////////////////////////////////////////////////////
    if(bExplicitDataType)
    {
        std::string dataTypeString(dicomDictionary::getDicomDictionary()->enumDataTypeToString(dataType));
        pDestStream->write((std::uint8_t*)(dataTypeString.c_str()), 2);

        std::uint16_t tagLengthWord = (std::uint16_t)tagLength;

        if(dicomDictionary::getDicomDictionary()->getLongLength(dataType))
        {
            std::uint32_t tagLengthDWord = bSequence ? 0xffffffff : tagLength;
            tagLengthWord = 0;
            pDestStream->adjustEndian((std::uint8_t*)&tagLengthDWord, 4, endianType);
            pDestStream->write((std::uint8_t*)&tagLengthWord, 2);
            pDestStream->write((std::uint8_t*)&tagLengthDWord, 4);
        }
        else
        {
            if(bSequence)
            {
                IMEBRA_THROW(InvalidSequenceItemError, "Sequences cannot be used with dataType " << dataTypeString);
            }
            pDestStream->adjustEndian((std::uint8_t*)&tagLengthWord, 2, endianType);
            pDestStream->write((std::uint8_t*)&tagLengthWord, 2);
        }
    }
    else
    {
        std::uint32_t tagLengthDword = bSequence ? 0xffffffff : tagLength;
        pDestStream->adjustEndian((std::uint8_t*)&tagLengthDword, 4, endianType);
        pDestStream->write((std::uint8_t*)&tagLengthDword, 4);
    }

    // Write all the buffers or datasets
    ///////////////////////////////////////////////////////////
    for(std::uint32_t scanBuffers = 0; ; ++scanBuffers)
    {
        if(pData->bufferExists(scanBuffers))
        {
            std::shared_ptr<handlers::readingDataHandlerRaw> pDataHandlerRaw = pData->getReadingDataHandlerRaw(scanBuffers);

            std::uint32_t wordSize = dicomDictionary::getDicomDictionary()->getWordSize(dataType);
            size_t bufferSize = pDataHandlerRaw->getSize();

            // write the sequence item header
            ///////////////////////////////////////////////////////////
            if(bSequence)
            {
                pDestStream->write((std::uint8_t*)&sequenceItemGroup, 2);
                pDestStream->write((std::uint8_t*)&sequenceItemDelimiter, 2);
                std::uint32_t sequenceItemLength = (std::uint32_t)bufferSize;
                pDestStream->adjustEndian((std::uint8_t*)&sequenceItemLength, 4, endianType);
                pDestStream->write((std::uint8_t*)&sequenceItemLength, 4);
            }

            if(bufferSize == 0)
            {
                continue;
            }

            // Adjust the buffer's endian
            ///////////////////////////////////////////////////////////
            if(wordSize > 1)
            {
                std::vector<std::uint8_t> tempBuffer(bufferSize);
                ::memcpy(tempBuffer.data(), pDataHandlerRaw->getMemoryBuffer(), pDataHandlerRaw->getSize());
                streamController::adjustEndian(tempBuffer.data(), wordSize, endianType, bufferSize / wordSize);
                pDestStream->write(tempBuffer.data(), bufferSize);
                continue;
            }

            pDestStream->write((std::uint8_t*)pDataHandlerRaw->getMemoryBuffer(), bufferSize);
            continue;
        }

        // Write a nested dataset
        ///////////////////////////////////////////////////////////
        if(!pData->dataSetExists(scanBuffers))
        {
            break;
        }
        std::shared_ptr<dataSet> pDataSet = pData->getSequenceItem(scanBuffers);

        // Don't write empty datasets
        ///////////////////////////////////////////////////////////
        if(pDataSet->getGroups().empty())
        {
            continue;
        }

        // Remember the position at which the item has been written
        ///////////////////////////////////////////////////////////
        pDataSet->setItemOffset((std::uint32_t)pDestStream->getControlledStreamPosition());

        // write the sequence item header
        ///////////////////////////////////////////////////////////
        pDestStream->write((std::uint8_t*)&sequenceItemGroup, 2);
        pDestStream->write((std::uint8_t*)&sequenceItemDelimiter, 2);
        std::uint32_t sequenceItemLength = getDataSetLength(pDataSet, bExplicitDataType);
        pDestStream->adjustEndian((std::uint8_t*)&sequenceItemLength, 4, endianType);
        pDestStream->write((std::uint8_t*)&sequenceItemLength, 4);

        // write the dataset
        ///////////////////////////////////////////////////////////
        buildStream(pDestStream, pDataSet, bExplicitDataType, endianType, streamType_t::normal);
    }

    // write the sequence item end marker
    ///////////////////////////////////////////////////////////
    if(bSequence)
    {
        pDestStream->write((std::uint8_t*)&sequenceItemGroup, 2);
        pDestStream->write((std::uint8_t*)&sequenceItemEnd, 2);
        std::uint32_t sequenceItemLength = 0;
        pDestStream->write((std::uint8_t*)&sequenceItemLength, 4);
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Calculate the tag's length
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t dicomCodec::getTagLength(const std::shared_ptr<data>& pData, bool bExplicitDataType, std::uint32_t* pHeaderLength, bool *pbSequence) const
{
    IMEBRA_FUNCTION_START();

    tagVR_t dataType = pData->getDataType();
    *pbSequence = (dataType == tagVR_t::SQ);
    std::uint32_t numberOfElements = 0;
    std::uint32_t totalLength = 0;
    for(std::uint32_t scanBuffers = 0; ; ++scanBuffers, ++numberOfElements)
    {
        if(pData->dataSetExists(scanBuffers))
        {
            std::shared_ptr<dataSet> pDataSet = pData->getSequenceItem(scanBuffers);
            totalLength += getDataSetLength(pDataSet, bExplicitDataType);
            totalLength += 8; // item tag and item length
            *pbSequence = true;
            continue;
        }
        if(!pData->bufferExists(scanBuffers))
        {
            break;
        }
        totalLength += (std::uint32_t)pData->getBufferSize(scanBuffers);
    }

    (*pbSequence) |= (numberOfElements > 1);

    // Find the tag type
    bool bLongLength = dicomDictionary::getDicomDictionary()->getLongLength(dataType);

    *pHeaderLength = 8;
    if((bLongLength || (*pbSequence)) && bExplicitDataType)
    {
        (*pHeaderLength) +=4;
    }

    if(*pbSequence)
    {
        totalLength += (numberOfElements+1) * 8;
    }

    return totalLength;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Calculate the group's length
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t dicomCodec::getGroupLength(const dataSet::tTags tags, bool bExplicitDataType) const
{
    IMEBRA_FUNCTION_START();

    std::uint32_t totalLength(0);

    for(dataSet::tTags::const_iterator scanTags(tags.begin()), endTags(tags.end()); scanTags != endTags; ++scanTags)
    {
        if(scanTags->first == 0)
        {
            continue;
        }

        std::uint32_t tagHeaderLength;
        bool bSequence;
        totalLength += getTagLength(scanTags->second, bExplicitDataType, &tagHeaderLength, &bSequence);
        totalLength += tagHeaderLength;
    }

    return totalLength;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Calculate the dataset's length
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t dicomCodec::getDataSetLength(std::shared_ptr<dataSet> pDataSet, bool bExplicitDataType) const
{
    IMEBRA_FUNCTION_START();

    dataSet::tGroupsIds groups(pDataSet->getGroups());

    std::uint32_t totalLength(0);

    for(dataSet::tGroupsIds::const_iterator scanGroups(groups.begin()), endGroups(groups.end()); scanGroups != endGroups; ++scanGroups)
    {
        size_t numGroups(pDataSet->getGroupsNumber(*scanGroups));
        for(size_t scanGroupsNumber(0); scanGroupsNumber != numGroups; ++scanGroupsNumber)
        {
            const dataSet::tTags& tags(pDataSet->getGroupTags(*scanGroups, scanGroupsNumber));
            totalLength += getGroupLength(tags, bExplicitDataType);
            totalLength += 4; // Add space for the tag 0
            if(bExplicitDataType) // Add space for the data type
            {
                totalLength += 2;
            }
            totalLength += 2; // Add space for the tag's length
            totalLength += 4; // Add space for the group's length
        }
    }

    return totalLength;

    IMEBRA_FUNCTION_END();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read a DICOM stream and fill the dataset with the
//  DICOM's content
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::readStream(std::shared_ptr<streamReader> pStream, std::shared_ptr<dataSet> pDataSet, std::uint32_t maxSizeBufferLoad /* = 0xffffffff */)
{
    IMEBRA_FUNCTION_START();

    // Save the starting position
    ///////////////////////////////////////////////////////////
    size_t position = pStream->position();

    // This flag signals a failure
    ///////////////////////////////////////////////////////////
    bool bFailed=false;

    // Read the old dicom signature (NEMA)
    ///////////////////////////////////////////////////////////
    std::uint8_t oldDicomSignature[8];

    try
    {
        pStream->read(oldDicomSignature, 8);
    }
    catch(StreamEOFError&)
    {
        IMEBRA_THROW(CodecWrongFormatError, "detected a wrong format");
    }

    // Skip the first 128 bytes (8 already skipped)
    ///////////////////////////////////////////////////////////
    pStream->seekForward(120);

    // Read the DICOM signature (DICM)
    ///////////////////////////////////////////////////////////
    std::uint8_t dicomSignature[4];
    pStream->read(dicomSignature, 4);

    // Check the DICM signature
    ///////////////////////////////////////////////////////////
    const char* checkSignature="DICM";
    if(::memcmp(dicomSignature, checkSignature, 4) != 0)
    {
        bFailed=true;
    }

    bool bExplicitDataType = true;
    streamController::tByteOrdering endianType=streamController::lowByteEndian;
    if(bFailed)
    {
        // Tags 0x8 and 0x2 are accepted in the begin of the file
        ///////////////////////////////////////////////////////////
        if(
                (oldDicomSignature[0]!=0x8 && oldDicomSignature[0]!=0x2) ||
                oldDicomSignature[1]!=0x0 ||
                oldDicomSignature[3]!=0x0)
        {
            IMEBRA_THROW(CodecWrongFormatError, "detected a wrong format (checked old NEMA signature)");
        }

        // Go back to the beginning of the file
        ///////////////////////////////////////////////////////////
        pStream->seek(position);

        // Set "explicit data type" to true if a valid data type
        //  is found
        ///////////////////////////////////////////////////////////
        std::string firstDataType;
        firstDataType.push_back((char)(oldDicomSignature[4]));
        firstDataType.push_back((char)(oldDicomSignature[5]));
        bExplicitDataType = dicomDictionary::getDicomDictionary()->isDataTypeValid(firstDataType);
    }

    // Signature OK. Now scan all the tags.
    ///////////////////////////////////////////////////////////
    parseStream(pStream, pDataSet, bExplicitDataType, endianType, maxSizeBufferLoad);

    IMEBRA_FUNCTION_END();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Parse a Dicom stream
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::parseStream(std::shared_ptr<streamReader> pStream,
                             std::shared_ptr<dataSet> pDataSet,
                             bool bExplicitDataType,
                             streamController::tByteOrdering endianType,
                             std::uint32_t maxSizeBufferLoad /* = 0xffffffff */,
                             std::uint32_t subItemLength /* = 0xffffffff */,
                             std::uint32_t* pReadSubItemLength /* = 0 */,
                             std::uint32_t depth)
{
    IMEBRA_FUNCTION_START();

    if(depth > IMEBRA_DATASET_MAX_DEPTH)
    {
        IMEBRA_THROW(DicomCodecDepthLimitReachedError, "Depth for embedded dataset reached");
    }

    std::uint16_t tagId;
    std::uint16_t tagSubId;
    std::uint16_t tagLengthWord;
    std::uint32_t tagLengthDWord;

    // Used to calculate the group order
    ///////////////////////////////////////////////////////////
    std::uint16_t order = 0;
    std::uint16_t lastGroupId = 0;
    std::uint16_t lastTagId = 0;

    std::uint32_t tempReadSubItemLength = 0; // used when the last parameter is not defined
    bool       bStopped = false;
    bool       bFirstTag = (pReadSubItemLength == 0);
    bool       bCheckTransferSyntax = bFirstTag;
    size_t     wordSize;

    if(pReadSubItemLength == 0)
    {
        pReadSubItemLength = &tempReadSubItemLength;
    }
    *pReadSubItemLength = 0;

    ///////////////////////////////////////////////////////////
    //
    // Read all the tags
    //
    ///////////////////////////////////////////////////////////
    while(!bStopped && !pStream->endReached() && (*pReadSubItemLength < subItemLength))
    {
        // Get the tag's ID
        ///////////////////////////////////////////////////////////
        pStream->read((std::uint8_t*)&tagId, sizeof(tagId));
        pStream->adjustEndian((std::uint8_t*)&tagId, sizeof(tagId), endianType);
        (*pReadSubItemLength) += (std::uint32_t)sizeof(tagId);

        // Check for EOF
        ///////////////////////////////////////////////////////////
        if(pStream->endReached())
        {
            break;
        }

        // Check the byte order
        ///////////////////////////////////////////////////////////
        if(bFirstTag && tagId==0x0200)
        {
            // Reverse the last adjust
            pStream->adjustEndian((std::uint8_t*)&tagId, sizeof(tagId), endianType);

            // Fix the byte adjustment
            endianType=streamController::highByteEndian;

            // Redo the byte adjustment
            pStream->adjustEndian((std::uint8_t*)&tagId, sizeof(tagId), endianType);
        }

        // If this tag's id is not 0x0002, then load the
        //  transfer syntax and set the byte endian.
        ///////////////////////////////////////////////////////////
        if(tagId!=0x0002 && bCheckTransferSyntax)
        {
            // Reverse the last adjust
            pStream->adjustEndian((std::uint8_t*)&tagId, sizeof(tagId), endianType);

            std::string transferSyntax = pDataSet->getString(
                        0x0002,
                        0x0,
                        0x0010,
                        0,
                        0,
                        endianType == streamController::lowByteEndian ? "1.2.840.10008.1.2.1" : "1.2.840.10008.1.2.2");

            if(transferSyntax == "1.2.840.10008.1.2.2")
                endianType = streamController::highByteEndian;
            if(transferSyntax == "1.2.840.10008.1.2")
                bExplicitDataType=false;

            bCheckTransferSyntax=false;

            // Redo the byte adjustment
            pStream->adjustEndian((std::uint8_t*)&tagId, sizeof(tagId), endianType);
        }

        // The first tag's ID has been read
        ///////////////////////////////////////////////////////////
        bFirstTag = false;

        // Set the word's length to the default value
        ///////////////////////////////////////////////////////////
        wordSize = 1;

        // Get the tag's sub ID
        ///////////////////////////////////////////////////////////
        pStream->read((std::uint8_t*)&tagSubId, sizeof(tagSubId));
        pStream->adjustEndian((std::uint8_t*)&tagSubId, sizeof(tagSubId), endianType);
        (*pReadSubItemLength) += (std::uint32_t)sizeof(tagSubId);

        // Check for the end of the dataset
        ///////////////////////////////////////////////////////////
        if(tagId==0xfffe && tagSubId==0xe00d)
        {
            // skip the tag's length and exit
            std::uint32_t dummyDWord;
            pStream->read((std::uint8_t*)&dummyDWord, 4);
            (*pReadSubItemLength) += 4;
            break;
        }

        //
        // Explicit data type
        //
        ///////////////////////////////////////////////////////////
        tagVR_t tagType;

        if(bExplicitDataType && tagId!=0xfffe)
        {
            // Get the tag's type
            ///////////////////////////////////////////////////////////
            std::string tagTypeString((size_t)2, ' ');

            pStream->read((std::uint8_t*)&(tagTypeString[0]), 2);
            (*pReadSubItemLength) += 2;

            // Get the tag's length
            ///////////////////////////////////////////////////////////
            pStream->read((std::uint8_t*)&tagLengthWord, sizeof(tagLengthWord));
            pStream->adjustEndian((std::uint8_t*)&tagLengthWord, sizeof(tagLengthWord), endianType);
            (*pReadSubItemLength) += (std::uint32_t)sizeof(tagLengthWord);

            // The data type is valid
            ///////////////////////////////////////////////////////////
            try
            {
                tagType = dicomDictionary::getDicomDictionary()->stringDataTypeToEnum(tagTypeString);
                tagLengthDWord=(std::uint32_t)tagLengthWord;
                wordSize = dicomDictionary::getDicomDictionary()->getWordSize(tagType);
                if(dicomDictionary::getDicomDictionary()->getLongLength(tagType))
                {
                    pStream->read((std::uint8_t*)&tagLengthDWord, sizeof(tagLengthDWord));
                    pStream->adjustEndian((std::uint8_t*)&tagLengthDWord, sizeof(tagLengthDWord), endianType);
                    (*pReadSubItemLength) += (std::uint32_t)sizeof(tagLengthDWord);
                }
            }
            catch(const DictionaryUnknownDataTypeError&)
            {
                // The data type is not valid. Switch to implicit data type
                ///////////////////////////////////////////////////////////
                bExplicitDataType = false;
                if(endianType == streamController::lowByteEndian)
                    tagLengthDWord=(((std::uint32_t)tagLengthWord)<<16) | ((std::uint32_t)tagTypeString[0]) | (((std::uint32_t)tagTypeString[1])<<8);
                else
                    tagLengthDWord=(std::uint32_t)tagLengthWord | (((std::uint32_t)tagTypeString[0])<<24) | (((std::uint32_t)tagTypeString[1])<<16);
            }


        } // End of the explicit data type read block


        ///////////////////////////////////////////////////////////
        //
        // Implicit data type
        //
        ///////////////////////////////////////////////////////////
        else
        {
            // Get the tag's length
            ///////////////////////////////////////////////////////////
            pStream->read((std::uint8_t*)&tagLengthDWord, sizeof(tagLengthDWord));
            pStream->adjustEndian((std::uint8_t*)&tagLengthDWord, sizeof(tagLengthDWord), endianType);
            (*pReadSubItemLength) += (std::uint32_t)sizeof(tagLengthDWord);
        }

        ///////////////////////////////////////////////////////////
        //
        // Find the default data type and the tag's word's size
        //
        ///////////////////////////////////////////////////////////
        if((!bExplicitDataType || tagId==0xfffe))
        {
            // Group length. Data type is always UL
            ///////////////////////////////////////////////////////////
            if(tagSubId == 0)
            {
                tagType = tagVR_t::UL;
            }
            else
            {
                tagType = dicomDictionary::getDicomDictionary()->getTagType(tagId, tagSubId);
                wordSize = dicomDictionary::getDicomDictionary()->getWordSize(tagType);
            }
        }

        // Check for the end of a sequence
        ///////////////////////////////////////////////////////////
        if(tagId==0xfffe && tagSubId==0xe0dd)
        {
            break;
        }

        ///////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////
        //
        //
        // Read the tag's buffer
        //
        //
        ///////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        //
        // Adjust the order when multiple groups with the same
        //  id are present.
        //
        ///////////////////////////////////////////////////////////
        if(tagId<=lastGroupId && tagSubId<=lastTagId)
        {
            ++order;
        }
        else
        {
            if(tagId>lastGroupId)
            {
                order = 0;
            }
        }
        lastGroupId=tagId;
        lastTagId=tagSubId;

        if(tagLengthDWord != 0xffffffff && tagType != tagVR_t::SQ)
        {
            (*pReadSubItemLength) += readTag(pStream, pDataSet, tagLengthDWord, tagId, order, tagSubId, tagType, endianType, wordSize, 0, maxSizeBufferLoad);
            continue;
        }

        ///////////////////////////////////////////////////////////
        //
        // We are within an undefined-length tag or a sequence
        //
        ///////////////////////////////////////////////////////////

        // Parse all the sequence's items
        ///////////////////////////////////////////////////////////
        std::uint16_t subItemGroupId;
        std::uint16_t subItemTagId;
        std::uint32_t sequenceItemLength;
        std::uint32_t bufferId = 0;
        while(tagLengthDWord && !pStream->endReached())
        {
            // Add the tag to the dataset
            ///////////////////////////////////////////////////////////
            std::shared_ptr<data> sequenceTag = pDataSet->getTagCreate(tagId, 0x0, tagSubId, tagType);

            // Remember the item's position (used by DICOMDIR
            //  structures)
            ///////////////////////////////////////////////////////////
            std::uint32_t itemOffset((std::uint32_t)pStream->getControlledStreamPosition());

            // Read the sequence item's group
            ///////////////////////////////////////////////////////////
            pStream->read((std::uint8_t*)&subItemGroupId, sizeof(subItemGroupId));
            pStream->adjustEndian((std::uint8_t*)&subItemGroupId, sizeof(subItemGroupId), endianType);
            (*pReadSubItemLength) += (std::uint32_t)sizeof(subItemGroupId);

            // Read the sequence item's id
            ///////////////////////////////////////////////////////////
            pStream->read((std::uint8_t*)&subItemTagId, sizeof(subItemTagId));
            pStream->adjustEndian((std::uint8_t*)&subItemTagId, sizeof(subItemTagId), endianType);
            (*pReadSubItemLength) += (std::uint32_t)sizeof(subItemTagId);

            // Read the sequence item's length
            ///////////////////////////////////////////////////////////
            pStream->read((std::uint8_t*)&sequenceItemLength, sizeof(sequenceItemLength));
            pStream->adjustEndian((std::uint8_t*)&sequenceItemLength, sizeof(sequenceItemLength), endianType);
            (*pReadSubItemLength) += (std::uint32_t)sizeof(sequenceItemLength);

            if(tagLengthDWord!=0xffffffff)
            {
                tagLengthDWord-=8;
            }

            // check for the end of the undefined length sequence
            ///////////////////////////////////////////////////////////
            if(subItemGroupId==0xfffe && subItemTagId==0xe0dd)
            {
                break;
            }

            ///////////////////////////////////////////////////////////
            // Parse a sub element
            ///////////////////////////////////////////////////////////
            if((sequenceItemLength == 0xffffffff) || tagType == tagVR_t::SQ)
            {
                std::shared_ptr<dataSet> sequenceDataSet(std::make_shared<dataSet>());
                sequenceDataSet->setItemOffset(itemOffset);
                std::uint32_t effectiveLength(0);
                parseStream(pStream, sequenceDataSet, bExplicitDataType, endianType, maxSizeBufferLoad, sequenceItemLength, &effectiveLength, depth + 1);
                (*pReadSubItemLength) += effectiveLength;
                if(tagLengthDWord!=0xffffffff)
                    tagLengthDWord-=effectiveLength;
                sequenceTag->setSequenceItem(bufferId, sequenceDataSet);
                ++bufferId;

                continue;
            }

            ///////////////////////////////////////////////////////////
            // Read a buffer's element
            ///////////////////////////////////////////////////////////
            sequenceItemLength=readTag(pStream, pDataSet, sequenceItemLength, tagId, order, tagSubId, tagType, endianType, wordSize, bufferId++, maxSizeBufferLoad);
            (*pReadSubItemLength) += sequenceItemLength;
            if(tagLengthDWord!=0xffffffff)
            {
                tagLengthDWord -= sequenceItemLength;
            }
        }

    } // End of the tags-read block

    IMEBRA_FUNCTION_END();

}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Get a DICOM raw or RLE image from a dicom structure
//
//
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
std::shared_ptr<image> dicomCodec::getImage(const dataSet& dataset, std::shared_ptr<streamReader> pStream, tagVR_t dataType)
{
    IMEBRA_FUNCTION_START();

    streamReader* pSourceStream = pStream.get();

    // Check for RLE compression
    ///////////////////////////////////////////////////////////
    std::string transferSyntax = dataset.getString(0x0002, 0x0, 0x0010, 0, 0, "1.2.840.10008.1.2");
    bool bRleCompressed = (transferSyntax == "1.2.840.10008.1.2.5");

    // Check for color space and subsampled channels
    ///////////////////////////////////////////////////////////
    std::string colorSpace = dataset.getString(0x0028, 0x0, 0x0004, 0, 0);

    // Retrieve the number of planes
    ///////////////////////////////////////////////////////////
    std::uint8_t channelsNumber=(std::uint8_t)dataset.getUnsignedLong(0x0028, 0x0, 0x0002, 0, 0);

    // Adjust the colorspace and the channels number for old
    //  NEMA files that don't specify those data
    ///////////////////////////////////////////////////////////
    if(colorSpace.empty() && (channelsNumber == 0 || channelsNumber == 1))
    {
        colorSpace = "MONOCHROME2";
        channelsNumber = 1;
    }

    if(colorSpace.empty() && channelsNumber == 3)
    {
        colorSpace = "RGB";
    }

    // Retrieve the image's size
    ///////////////////////////////////////////////////////////
    std::uint32_t imageWidth = dataset.getUnsignedLong(0x0028, 0x0, 0x0011, 0, 0);
    std::uint32_t imageHeight = dataset.getUnsignedLong(0x0028, 0x0, 0x0010, 0, 0);

    if(
            imageWidth > codecFactory::getCodecFactory()->getMaximumImageWidth() ||
            imageHeight > codecFactory::getCodecFactory()->getMaximumImageHeight())
    {
        IMEBRA_THROW(CodecImageTooBigError, "The factory settings prevented the loading of this image. Consider using codecFactory::setMaximumImageSize() to modify the settings");
    }

    if((imageWidth == 0) || (imageHeight == 0))
    {
        IMEBRA_THROW(CodecCorruptedFileError, "The size tags are not available");
    }

    // Check for interleaved planes.
    ///////////////////////////////////////////////////////////
    bool bInterleaved(dataset.getUnsignedLong(0x0028, 0x0, 0x0006, 0, 0, 0) == 0x0);

    // Check for 2's complement
    ///////////////////////////////////////////////////////////
    bool b2Complement = dataset.getUnsignedLong(0x0028, 0x0, 0x0103, 0, 0, 0) != 0x0;

    // Retrieve the allocated/stored/high bits
    ///////////////////////////////////////////////////////////
    std::uint8_t allocatedBits=(std::uint8_t)dataset.getUnsignedLong(0x0028, 0x0, 0x0100, 0, 0);
    std::uint8_t storedBits=(std::uint8_t)dataset.getUnsignedLong(0x0028, 0x0, 0x0101, 0, 0);
    std::uint8_t highBit=(std::uint8_t)dataset.getUnsignedLong(0x0028, 0x0, 0x0102, 0, 0);
    if(highBit < storedBits - 1)
        throw;


    // If the chrominance channels are subsampled, then find
    //  the right image's size
    ///////////////////////////////////////////////////////////
    bool bSubSampledY = channelsNumber > 0x1 && transforms::colorTransforms::colorTransformsFactory::isSubsampledY(colorSpace);
    bool bSubSampledX = channelsNumber > 0x1 && transforms::colorTransforms::colorTransformsFactory::isSubsampledX(colorSpace);

    // Create an image
    ///////////////////////////////////////////////////////////
    bitDepth_t depth;
    if(b2Complement)
    {
        if(highBit >= 16)
        {
            depth = bitDepth_t::depthS32;
        }
        else if(highBit >= 8)
        {
            depth = bitDepth_t::depthS16;
        }
        else
        {
            depth = bitDepth_t::depthS8;
        }
    }
    else
    {
        if(highBit >= 16)
        {
            depth = bitDepth_t::depthU32;
        }
        else if(highBit >= 8)
        {
            depth = bitDepth_t::depthU16;
        }
        else
        {
            depth = bitDepth_t::depthU8;
        }
    }

    std::shared_ptr<image> pImage(std::make_shared<image>(imageWidth, imageHeight, depth, colorSpace, highBit));
    std::shared_ptr<handlers::writingDataHandlerNumericBase> handler = pImage->getWritingDataHandler();
    std::uint32_t tempChannelsNumber = pImage->getChannelsNumber();

    if(handler == 0 || tempChannelsNumber != channelsNumber)
    {
        IMEBRA_THROW(CodecCorruptedFileError, "Cannot allocate the image's buffer");
    }

    // Allocate the dicom channels
    ///////////////////////////////////////////////////////////
    allocChannels(channelsNumber, imageWidth, imageHeight, bSubSampledX, bSubSampledY);

    std::uint32_t mask = (std::uint32_t)( ((std::uint64_t)1 << (highBit + 1)) - 1);
    mask -= (std::uint32_t)(((std::uint64_t)1 << (highBit + 1 - storedBits)) - 1);

    //
    // The image is not compressed
    //
    ///////////////////////////////////////////////////////////
    if(!bRleCompressed)
    {
        std::uint32_t wordSizeBytes = (dataType == tagVR_t::OW) ? 2 : 1;

        // The planes are interleaved
        ///////////////////////////////////////////////////////////
        if(bInterleaved && channelsNumber != 1)
        {
            readUncompressedInterleaved(
                        channelsNumber,
                        bSubSampledX,
                        bSubSampledY,
                        pSourceStream,
                        wordSizeBytes,
                        allocatedBits,
                        mask);
        }
        else
        {
            readUncompressedNotInterleaved(
                        channelsNumber,
                        pSourceStream,
                        wordSizeBytes,
                        allocatedBits,
                        mask);
        }
    }

    //
    // The image is RLE compressed
    //
    ///////////////////////////////////////////////////////////
    else
    {
        if(bSubSampledX || bSubSampledY)
        {
            IMEBRA_THROW(CodecCorruptedFileError, "Cannot read subsampled RLE images");
        }

        readRLECompressed(imageWidth, imageHeight, channelsNumber, pSourceStream, allocatedBits, mask, bInterleaved);

    } // ...End of RLE decoding

    // Adjust b2complement buffers
    ///////////////////////////////////////////////////////////
    if(b2Complement)
    {
        std::int32_t checkSign = (std::int32_t)1 << highBit;
        std::int32_t orMask = ((std::int32_t)-1) << highBit;

        for(size_t adjChannels = 0; adjChannels < m_channels.size(); ++adjChannels)
        {
            std::int32_t* pAdjBuffer = m_channels[adjChannels]->m_pBuffer;
            std::uint32_t adjSize = m_channels[adjChannels]->m_bufferSize;
            while(adjSize != 0)
            {
                if(*pAdjBuffer & checkSign)
                {
                    *pAdjBuffer |= orMask;
                }
                ++pAdjBuffer;
                --adjSize;
            }
        }
    }


    // Copy the dicom channels into the image
    ///////////////////////////////////////////////////////////
    std::uint32_t maxSamplingFactorX = bSubSampledX ? 2 : 1;
    std::uint32_t maxSamplingFactorY = bSubSampledY ? 2 : 1;
    for(std::uint32_t copyChannels = 0; copyChannels < channelsNumber; ++copyChannels)
    {
        ptrChannel dicomChannel = m_channels[copyChannels];
        handler->copyFromInt32Interleaved(
                    dicomChannel->m_pBuffer,
                    maxSamplingFactorX /dicomChannel->m_samplingFactorX,
                    maxSamplingFactorY /dicomChannel->m_samplingFactorY,
                    0, 0,
                    dicomChannel->m_width * maxSamplingFactorX / dicomChannel->m_samplingFactorX,
                    dicomChannel->m_height * maxSamplingFactorY / dicomChannel->m_samplingFactorY,
                    copyChannels,
                    imageWidth,
                    imageHeight,
                    channelsNumber);
    }

    // Return OK
    ///////////////////////////////////////////////////////////
    return pImage;

    IMEBRA_FUNCTION_END();

}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Allocate the channels used to read/write an image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::allocChannels(std::uint32_t channelsNumber, std::uint32_t width, std::uint32_t height, bool bSubSampledX, bool bSubSampledY)
{
    IMEBRA_FUNCTION_START();

    if(bSubSampledX && (width & 0x1) != 0)
    {
        ++width;
    }

    if(bSubSampledY && (height & 0x1) != 0)
    {
        ++height;
    }

    m_channels.resize(channelsNumber);
    for(std::uint32_t channelNum = 0; channelNum < channelsNumber; ++channelNum)
    {
        std::uint32_t channelWidth = width;
        std::uint32_t channelHeight = height;
        std::uint32_t samplingFactorX = 1;
        std::uint32_t samplingFactorY = 1;
        if(channelNum != 0)
        {
            if(bSubSampledX)
            {
                channelWidth >>= 1;
            }
            if(bSubSampledY)
            {
                channelHeight >>= 1;
            }
        }
        else
        {
            if(bSubSampledX)
            {
                ++samplingFactorX;
            }
            if(bSubSampledY)
            {
                ++samplingFactorY;
            }
        }

        ptrChannel newChannel(std::make_shared<channel>());
        newChannel->allocate(channelWidth, channelHeight);
        newChannel->m_samplingFactorX = samplingFactorX;
        newChannel->m_samplingFactorY = samplingFactorY;

        m_channels[channelNum] = newChannel;
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read an uncompressed interleaved image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::readUncompressedInterleaved(
        std::uint32_t channelsNumber,
        bool bSubSampledX,
        bool bSubSampledY,
        streamReader* pSourceStream,
        std::uint32_t wordSizeBytes,
        std::uint32_t allocatedBits,
        std::uint32_t mask
        )
{
    IMEBRA_FUNCTION_START();

    std::uint8_t  bitPointer=0x0;

    std::vector<std::int32_t*> channelsMemory(m_channels.size());
    for(size_t copyChannelsPntr = 0; copyChannelsPntr < m_channels.size(); ++copyChannelsPntr)
    {
        channelsMemory[copyChannelsPntr] = m_channels[copyChannelsPntr]->m_pBuffer;
    }

    // No subsampling here
    ///////////////////////////////////////////////////////////
    if(!bSubSampledX && !bSubSampledY)
    {
        std::uint8_t readBuffer[4];
        for(std::uint32_t totalSize = m_channels[0]->m_bufferSize; totalSize != 0; --totalSize)
        {
            for(std::uint32_t scanChannels = 0; scanChannels != channelsNumber; ++scanChannels)
            {
                readPixel(pSourceStream, channelsMemory[scanChannels]++, 1, &bitPointer, readBuffer, wordSizeBytes, allocatedBits, mask);
            }
        }
        return;
    }

    std::uint32_t numValuesPerBlock(channelsNumber);
    if(bSubSampledX)
    {
        ++numValuesPerBlock;
    }
    if(bSubSampledY)
    {
        numValuesPerBlock += 2;
    }
    std::vector<std::int32_t> readBlockValues((size_t)numValuesPerBlock);

    // Read the subsampled channels.
    // Find the number of blocks to read
    ///////////////////////////////////////////////////////////
    std::uint32_t adjWidth = m_channels[0]->m_width;
    std::uint32_t adjHeight = m_channels[0]->m_height;

    std::uint32_t maxSamplingFactorX = bSubSampledX ? 2 : 1;
    std::uint32_t maxSamplingFactorY = bSubSampledY ? 2 : 1;

    std::shared_ptr<memory> readBuffer(std::make_shared<memory>(numValuesPerBlock * ((7 + allocatedBits) >> 3)));

    // Read all the blocks
    ///////////////////////////////////////////////////////////
    std::uint32_t numBlocksY = adjHeight / maxSamplingFactorY;
    std::uint32_t numBlocksX = adjWidth / maxSamplingFactorX;
    for(std::uint32_t blockY(0); blockY != numBlocksY; ++blockY)
    {
        for(std::uint32_t blockX(0); blockX != numBlocksX; ++blockX)
        {
            std::int32_t* readBlockValuesPtr(&(readBlockValues[0]));
            readPixel(pSourceStream, readBlockValuesPtr, numValuesPerBlock, &bitPointer, readBuffer->data(), wordSizeBytes, allocatedBits, mask);

            // Read channel 0 (not subsampled)
            ///////////////////////////////////////////////////////////
            *(channelsMemory[0]++) = *readBlockValuesPtr++;
            if(bSubSampledX)
            {
                *(channelsMemory[0]++) = *readBlockValuesPtr++;
            }
            if(bSubSampledY)
            {
                *(channelsMemory[0]+adjWidth-2) = *readBlockValuesPtr++;
                *(channelsMemory[0]+adjWidth-1) = *readBlockValuesPtr++;
            }
            // Read channels 1... (subsampled)
            ///////////////////////////////////////////////////////////
            for(std::uint32_t scanSubSampled = 1; scanSubSampled < channelsNumber; ++scanSubSampled)
            {
                *(channelsMemory[scanSubSampled]++) = *readBlockValuesPtr++;
            }
        }
        if(bSubSampledY)
        {
            channelsMemory[0] += adjWidth;
        }
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write an uncompressed interleaved image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writeUncompressedInterleaved(
        std::uint32_t channelsNumber,
        bool bSubSampledX,
        bool bSubSampledY,
        streamWriter* pDestStream,
        std::uint32_t wordSizeBytes,
        std::uint32_t allocatedBits,
        std::uint32_t mask
        )
{
    IMEBRA_FUNCTION_START();

    std::uint8_t  bitPointer=0x0;

    std::vector<std::int32_t*> channelsMemory(m_channels.size());
    for(size_t copyChannelsPntr = 0; copyChannelsPntr < m_channels.size(); ++copyChannelsPntr)
    {
        channelsMemory[copyChannelsPntr] = m_channels[copyChannelsPntr]->m_pBuffer;
    }

    // No subsampling here
    ///////////////////////////////////////////////////////////
    if(!bSubSampledX && !bSubSampledY)
    {
        for(std::uint32_t totalSize = m_channels[0]->m_bufferSize; totalSize != 0; --totalSize)
        {
            for(std::uint32_t scanChannels = 0; scanChannels < channelsNumber; ++scanChannels)
            {
                writePixel(pDestStream, *(channelsMemory[scanChannels]++), &bitPointer, wordSizeBytes, allocatedBits, mask);
            }
        }
        flushUnwrittenPixels(pDestStream, &bitPointer, wordSizeBytes);
        return;
    }

    // Write the subsampled channels.
    // Find the number of blocks to write
    ///////////////////////////////////////////////////////////
    std::uint32_t adjWidth = m_channels[0]->m_width;
    std::uint32_t adjHeight = m_channels[0]->m_height;

    std::uint32_t maxSamplingFactorX = bSubSampledX ? 2 : 1;
    std::uint32_t maxSamplingFactorY = bSubSampledY ? 2 : 1;

    // Write all the blocks
    ///////////////////////////////////////////////////////////
    std::uint32_t numBlocksY = adjHeight / maxSamplingFactorY;
    std::uint32_t numBlocksX = adjWidth / maxSamplingFactorX;
    for(std::uint32_t blockY(0); blockY != numBlocksY; ++blockY)
    {
        for(std::uint32_t blockX(0); blockX != numBlocksX; ++blockX)
        {
            // Write channel 0 (not subsampled)
            ///////////////////////////////////////////////////////////
            writePixel(pDestStream, *(channelsMemory[0]++), &bitPointer, wordSizeBytes, allocatedBits, mask);
            if(bSubSampledX)
            {
                writePixel(pDestStream, *(channelsMemory[0]++), &bitPointer, wordSizeBytes, allocatedBits, mask);
            }
            if(bSubSampledY)
            {
                writePixel(pDestStream, *(channelsMemory[0]+adjWidth-2), &bitPointer, wordSizeBytes, allocatedBits, mask);
                writePixel(pDestStream, *(channelsMemory[0]+adjWidth-1), &bitPointer, wordSizeBytes, allocatedBits, mask);
            }
            // Write channels 1... (subsampled)
            ///////////////////////////////////////////////////////////
            for(std::uint32_t scanSubSampled = 1; scanSubSampled < channelsNumber; ++scanSubSampled)
            {
                writePixel(pDestStream, *(channelsMemory[scanSubSampled]++), &bitPointer, wordSizeBytes, allocatedBits, mask);
            }
        }
        if(bSubSampledY)
        {
            channelsMemory[0] += adjWidth;
        }
    }

    flushUnwrittenPixels(pDestStream, &bitPointer, wordSizeBytes);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read an uncompressed not interleaved image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::readUncompressedNotInterleaved(
        std::uint32_t channelsNumber,
        streamReader* pSourceStream,
        std::uint32_t wordSizeBytes,
        std::uint32_t allocatedBits,
        std::uint32_t mask
        )
{
    IMEBRA_FUNCTION_START();

    std::uint8_t  bitPointer=0x0;

    std::shared_ptr<memory> readBuffer;
    std::uint32_t lastBufferSize(0);

    // Read all the pixels
    ///////////////////////////////////////////////////////////
    for(std::uint32_t channel = 0; channel < channelsNumber; ++channel)
    {
        if(m_channels[channel]->m_bufferSize != lastBufferSize)
        {
            lastBufferSize = m_channels[channel]->m_bufferSize;
            readBuffer = std::make_shared<memory>(lastBufferSize * ((7+allocatedBits) >> 3));
        }
        std::int32_t* pMemoryDest = m_channels[channel]->m_pBuffer;
        readPixel(pSourceStream, pMemoryDest, m_channels[channel]->m_bufferSize, &bitPointer, readBuffer->data(), wordSizeBytes, allocatedBits, mask);
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write an uncompressed not interleaved image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writeUncompressedNotInterleaved(
        std::uint32_t channelsNumber,
        streamWriter* pDestStream,
        std::uint32_t wordSizeBytes,
        std::uint32_t allocatedBits,
        std::uint32_t mask
        )
{
    IMEBRA_FUNCTION_START();

    std::uint8_t  bitPointer=0x0;

    // Write all the pixels
    ///////////////////////////////////////////////////////////
    for(std::uint32_t channel = 0; channel < channelsNumber; ++channel)
    {
        std::int32_t* pMemoryDest = m_channels[channel]->m_pBuffer;
        for(std::uint32_t scanPixels = m_channels[channel]->m_bufferSize; scanPixels != 0; --scanPixels)
        {
            writePixel(pDestStream, *pMemoryDest++, &bitPointer, wordSizeBytes, allocatedBits, mask);
        }
    }
    flushUnwrittenPixels(pDestStream, &bitPointer, wordSizeBytes);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write a RLE compressed image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writeRLECompressed(
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        std::uint32_t channelsNumber,
        streamWriter* pDestStream,
        std::uint32_t allocatedBits,
        std::uint32_t mask
        )
{
    IMEBRA_FUNCTION_START();

    std::uint32_t segmentsOffset[16];
    ::memset(segmentsOffset, 0, sizeof(segmentsOffset));

    // The first phase fills the segmentsOffset pointers, the
    //  second phase writes to the stream.
    ///////////////////////////////////////////////////////////
    for(int phase = 0; phase < 2; ++phase)
    {
        if(phase == 1)
        {
            pDestStream->adjustEndian((std::uint8_t*)segmentsOffset, 4, streamController::lowByteEndian, sizeof(segmentsOffset) / sizeof(segmentsOffset[0]));
            pDestStream->write((std::uint8_t*)segmentsOffset, sizeof(segmentsOffset));
        }

        std::uint32_t segmentNumber = 0;
        std::uint32_t offset = 64;

        for(std::uint32_t scanChannels = 0; scanChannels < channelsNumber; ++scanChannels)
        {
            std::vector<std::uint8_t> rowBytes(imageWidth);

            std::vector<std::uint8_t> differentBytes;
            differentBytes.reserve(imageWidth);

            for(std::int32_t rightShift = (std::int32_t)(((allocatedBits + 7) & 0xfffffff8) - 8); rightShift >= 0; rightShift -= 8)
            {
                std::int32_t* pPixel = m_channels[scanChannels]->m_pBuffer;

                if(phase == 0)
                {
                    segmentsOffset[++segmentNumber] = offset;
                    segmentsOffset[0] = segmentNumber;
                }
                else
                {
                    offset = segmentsOffset[++segmentNumber];
                }

                for(std::uint32_t scanY = imageHeight; scanY != 0; --scanY)
                {
                    std::uint8_t* rowBytesPointer = &(rowBytes[0]);

                    for(std::uint32_t scanX = imageWidth; scanX != 0; --scanX)
                    {
                        *(rowBytesPointer++) = (std::uint8_t)((*pPixel & mask) >> rightShift);
                        ++pPixel;
                    }

                    for(size_t scanBytes = 0; scanBytes < imageWidth; /* left empty */)
                    {
                        std::uint8_t currentByte = rowBytes[scanBytes];

                        // Calculate the run-length
                        ///////////////////////////
                        size_t runLength(1);
                        for(; ((scanBytes + runLength) != imageWidth) && rowBytes[scanBytes + runLength] == currentByte; ++runLength)
                        {
                        }

                        // Write the runlength
                        //////////////////////
                        if(runLength > 3)
                        {
                            if(!differentBytes.empty())
                            {
                                offset += (std::uint32_t)writeRLEDifferentBytes(&differentBytes, pDestStream, phase == 1);
                            }
                            if(runLength > 128)
                            {
                                runLength = 128;
                            }
                            offset += 2;
                            scanBytes += runLength;
                            if(phase == 1)
                            {
                                std::uint8_t lengthByte = (std::uint8_t)(1 - runLength);
                                pDestStream->write(&lengthByte, 1);
                                pDestStream->write(&currentByte, 1);
                            }
                            continue;
                        }

                        // Remmember sequence of different bytes
                        ////////////////////////////////////////
                        differentBytes.push_back(currentByte);
                        ++scanBytes;
                    } // for(std::uint32_t scanBytes = 0; scanBytes < imageWidth; )

                    offset += (std::uint32_t)writeRLEDifferentBytes(&differentBytes, pDestStream, phase == 1);

                } // for(std::uint32_t scanY = imageHeight; scanY != 0; --scanY)

                if((offset & 1) != 0)
                {
                    ++offset;
                    if(phase == 1)
                    {
                        const std::uint8_t command = 0x80;
                        pDestStream->write(&command, 1);
                    }
                }

            } // for(std::int32_t rightShift = ((allocatedBits + 7) & 0xfffffff8) -8; rightShift >= 0; rightShift -= 8)

        } // for(int scanChannels = 0; scanChannels < channelsNumber; ++scanChannels)

    } // for(int phase = 0; phase < 2; ++phase)

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Write RLE sequence of different bytes
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
size_t dicomCodec::writeRLEDifferentBytes(std::vector<std::uint8_t>* pDifferentBytes, streamWriter* pDestStream, bool bWrite)
{
    IMEBRA_FUNCTION_START();

    size_t writtenLength = 0;
    for(size_t offset(0); offset != pDifferentBytes->size();)
    {
        size_t writeSize = pDifferentBytes->size() - offset;
        if(writeSize > 128)
        {
            writeSize = 128;
        }
        writtenLength += writeSize + 1;
        if(bWrite)
        {
            const std::uint8_t writeLength((std::uint8_t)(writeSize - 1));
            pDestStream->write(&writeLength, 1);
            pDestStream->write(&(pDifferentBytes->at(offset)), writeSize);
        }
        offset += writeSize;
    }
    pDifferentBytes->clear();

    // return number of written bytes
    /////////////////////////////////
    return writtenLength;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read a RLE compressed image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::readRLECompressed(
        std::uint32_t imageWidth,
        std::uint32_t imageHeight,
        std::uint32_t channelsNumber,
        streamReader* pSourceStream,
        std::uint32_t allocatedBits,
        std::uint32_t mask,
        bool /*bInterleaved*/)
{
    IMEBRA_FUNCTION_START();

    // Copy the RLE header into the segmentsOffset array
    //  and adjust the byte endian to the machine architecture
    ///////////////////////////////////////////////////////////
    std::uint32_t segmentsOffset[16];
    ::memset(segmentsOffset, 0, sizeof(segmentsOffset));
    pSourceStream->read((std::uint8_t*)segmentsOffset, 64);
    pSourceStream->adjustEndian((std::uint8_t*)segmentsOffset, 4, streamController::lowByteEndian, sizeof(segmentsOffset) / sizeof(segmentsOffset[0]));

    //
    // Scan all the RLE segments
    //
    ///////////////////////////////////////////////////////////
    std::uint32_t loopsNumber = channelsNumber;
    std::uint32_t loopSize = imageWidth * imageHeight;

    std::uint32_t currentSegmentOffset = sizeof(segmentsOffset);
    std::uint8_t segmentNumber = 0;
    for(std::uint32_t channel = 0; channel<loopsNumber; ++channel)
    {
        for(std::int32_t leftShift = (std::int32_t)(((allocatedBits + 7) & 0xfffffff8) - 8); leftShift >= 0; leftShift -= 8)
        {
            // Prepare to scan all the RLE segment
            ///////////////////////////////////////////////////////////
            std::uint32_t segmentOffset = segmentsOffset[++segmentNumber]; // Get the offset
            pSourceStream->seekForward(segmentOffset - currentSegmentOffset);
            currentSegmentOffset = segmentOffset;

            std::uint8_t  rleByte = 0;         // RLE code
            std::uint8_t  copyBytes = 0;       // Number of bytes to copy
            std::uint8_t  runByte = 0;         // Byte to use in run-lengths
            std::uint8_t  runLength = 0;       // Number of bytes with the same information (runByte)
            std::uint8_t  copyBytesBuffer[0x81];

            std::int32_t* pChannelMemory = m_channels[channel]->m_pBuffer;
            std::uint32_t channelSize = loopSize;
            std::uint8_t* pScanCopyBytes;

            // Read the RLE segment
            ///////////////////////////////////////////////////////////
            pSourceStream->read(&rleByte, 1);
            ++currentSegmentOffset;
            while(channelSize != 0)
            {
                if(rleByte==0x80)
                {
                    pSourceStream->read(&rleByte, 1);
                    ++currentSegmentOffset;
                    continue;
                }

                // Copy the specified number of bytes
                ///////////////////////////////////////////////////////////
                if(rleByte<0x80)
                {
                    copyBytes = ++rleByte;
                    if(copyBytes < channelSize)
                    {
                        pSourceStream->read(copyBytesBuffer, copyBytes + 1);
                        currentSegmentOffset += copyBytes + 1;
                        rleByte = copyBytesBuffer[copyBytes];
                    }
                    else
                    {
                        pSourceStream->read(copyBytesBuffer, copyBytes);
                        currentSegmentOffset += copyBytes;
                    }
                    pScanCopyBytes = copyBytesBuffer;
                    while(copyBytes-- && channelSize != 0)
                    {
                        *pChannelMemory |= ((*pScanCopyBytes++) << leftShift) & mask;
                        ++pChannelMemory;
                        --channelSize;
                    }
                    continue;
                }

                // Copy the same byte several times
                ///////////////////////////////////////////////////////////
                runLength = (std::uint8_t)(1-rleByte);
                if(runLength < channelSize)
                {
                    pSourceStream->read(copyBytesBuffer, 2);
                    currentSegmentOffset += 2;
                    runByte = copyBytesBuffer[0];
                    rleByte = copyBytesBuffer[1];
                }
                else
                {
                    pSourceStream->read(&runByte, 1);
                    ++currentSegmentOffset;
                }
                while(runLength-- && channelSize != 0)
                {
                    *pChannelMemory |= (runByte << leftShift) & mask;
                    ++pChannelMemory;
                    --channelSize;
                }

            } // ...End of the segment scanning loop

        } // ...End of the leftshift calculation

    } // ...Channels scanning loop

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read a single component from a DICOM raw image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::readPixel(
        streamReader* pSourceStream,
        std::int32_t* pDest,
        std::uint32_t numPixels,
        std::uint8_t* pBitPointer,
        std::uint8_t* pReadBuffer,
        std::uint32_t wordSizeBytes,
        std::uint32_t allocatedBits,
        std::uint32_t mask)
{
    IMEBRA_FUNCTION_START();

    if(allocatedBits == 8 || allocatedBits == 16 || allocatedBits == 32)
    {
        pSourceStream->read(pReadBuffer, numPixels * (allocatedBits >> 3));
        if(allocatedBits == 8)
        {
            std::uint8_t* pSource(pReadBuffer);
            while(numPixels-- != 0)
            {
                *pDest++ = (std::int32_t)((std::uint32_t)(*pSource++) & mask);
            }
            return;
        }
        pSourceStream->adjustEndian(pReadBuffer, allocatedBits >> 3, streamController::lowByteEndian, numPixels);
        if(allocatedBits == 16)
        {
            std::uint16_t* pSource((std::uint16_t*)(pReadBuffer));
            while(numPixels-- != 0)
            {
                *pDest++ = (std::int32_t)((std::uint32_t)(*pSource++) & mask);
            }
            return;
        }
        std::uint32_t* pSource((std::uint32_t*)(pReadBuffer));
        while(numPixels-- != 0)
        {
            *pDest++ = (std::int32_t)((*pSource++) & mask);
        }
        return;

    }

    while(numPixels-- != 0)
    {
        *pDest = 0;
        for(std::uint8_t bitsToRead = allocatedBits; bitsToRead != 0;)
        {
            if(*pBitPointer == 0)
            {
                if(wordSizeBytes == 2)
                {
                    pSourceStream->read((std::uint8_t*)&m_ioWord, sizeof(m_ioWord));
                    *pBitPointer = 16;
                }
                else
                {
                    pSourceStream->read(&m_ioByte, 1);
                    m_ioWord = (std::uint16_t)m_ioByte;
                    *pBitPointer = 8;
                }
            }

            if(*pBitPointer <= bitsToRead)
            {
                *pDest |= m_ioWord << (allocatedBits - bitsToRead);
                bitsToRead = (std::uint8_t)(bitsToRead - *pBitPointer);
                *pBitPointer = 0;
                continue;
            }

            *pDest |= (m_ioWord & (std::uint16_t)(((std::uint32_t) 1 << bitsToRead) - 1)) << (allocatedBits - bitsToRead);
            m_ioWord = (std::uint16_t)(m_ioWord >> bitsToRead);
            *pBitPointer = (std::uint8_t)(*pBitPointer - bitsToRead);
            bitsToRead = 0;
        }
        *pDest++ &= mask;
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read a single component from a DICOM raw image
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::writePixel(
        streamWriter* pDestStream,
        std::int32_t pixelValue,
        std::uint8_t*  pBitPointer,
        std::uint32_t wordSizeBytes,
        std::uint32_t allocatedBits,
        std::uint32_t mask)
{
    IMEBRA_FUNCTION_START();

    pixelValue &= mask;

    if(allocatedBits == 8)
    {
        m_ioByte = (std::uint8_t)pixelValue;
        pDestStream->write(&m_ioByte, sizeof(m_ioByte));
        return;
    }

    if(allocatedBits == 16)
    {
        if(wordSizeBytes == 1)
        {
            m_ioWord = pDestStream->adjustEndian((std::uint16_t)pixelValue, streamController::lowByteEndian);
        }
        else
        {
            m_ioWord = (std::uint16_t)pixelValue;
        }
        pDestStream->write((std::uint8_t*)&m_ioWord, sizeof(m_ioWord));
        return;
    }

    if(allocatedBits == 32)
    {
        if(wordSizeBytes == 1)
        {
            m_ioDWord = pDestStream->adjustEndian((std::uint32_t)pixelValue, streamController::lowByteEndian);
        }
        else
        {
            m_ioDWord = (std::uint32_t)pixelValue;
        }
        pDestStream->write((std::uint8_t*)&m_ioDWord, sizeof(m_ioDWord));
        return;
    }

    std::uint32_t maxBits = wordSizeBytes << 3;

    for(std::uint32_t writeBits = allocatedBits; writeBits != 0;)
    {
        std::uint32_t freeBits = (maxBits - *pBitPointer);
        if(freeBits == maxBits)
        {
            m_ioWord = 0;
        }
        if( freeBits <= writeBits )
        {
            m_ioWord = (std::uint16_t)(m_ioWord | ((pixelValue & (((std::int32_t)1 << freeBits) -1 )) << *pBitPointer));
            *pBitPointer = maxBits;
            writeBits = writeBits - freeBits;
            pixelValue >>= freeBits;
        }
        else
        {
            m_ioWord = (std::uint16_t)(m_ioWord | ((pixelValue & (((std::int32_t)1 << writeBits) -1 ))<< *pBitPointer));
            *pBitPointer = (std::uint8_t)(*pBitPointer + writeBits);
            writeBits = 0;
        }

        if(*pBitPointer == maxBits)
        {
            if(wordSizeBytes == 2)
            {
                pDestStream->write((std::uint8_t*)&m_ioWord, 2);
            }
            else
            {
                m_ioByte = (std::uint8_t)m_ioWord;
                pDestStream->write(&m_ioByte, 1);
            }
            *pBitPointer = 0;
        }
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Used by the writing routines to commit the unwritten
//  bits
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::flushUnwrittenPixels(streamWriter* pDestStream, std::uint8_t* pBitPointer, std::uint32_t wordSizeBytes)
{
    IMEBRA_FUNCTION_START();

    if(*pBitPointer == 0)
    {
        return;
    }
    if(wordSizeBytes == 2)
    {
        pDestStream->write((std::uint8_t*)&m_ioWord, 2);
    }
    else if(wordSizeBytes == 4)
    {
        pDestStream->write((std::uint8_t*)&m_ioDWord, 4);
    }
    else
    {
        m_ioByte = (std::uint8_t)m_ioWord;
        pDestStream->write(&m_ioByte, 1);
    }
    *pBitPointer = 0;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Insert an image into a Dicom structure
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void dicomCodec::setImage(
        std::shared_ptr<streamWriter> pDestStream,
        std::shared_ptr<image> pImage,
        const std::string& transferSyntax,
        imageQuality_t /*imageQuality*/,
        tagVR_t dataType,
        std::uint32_t allocatedBits,
        bool bSubSampledX,
        bool bSubSampledY,
        bool bInterleaved,
        bool /*b2Complement*/)
{
    IMEBRA_FUNCTION_START();

    // First calculate the attributes we want to use.
    // Return an exception if they are different from the
    //  old ones and bDontChangeAttributes is true
    ///////////////////////////////////////////////////////////
    std::uint32_t imageWidth, imageHeight;
    pImage->getSize(&imageWidth, &imageHeight);

    std::string colorSpace = pImage->getColorSpace();
    std::uint32_t highBit = pImage->getHighBit();
    bool bRleCompressed = (transferSyntax == "1.2.840.10008.1.2.5");

    std::shared_ptr<handlers::readingDataHandlerNumericBase> imageHandler = pImage->getReadingDataHandler();
    std::uint32_t channelsNumber = pImage->getChannelsNumber();

    // Copy the image into the dicom channels
    ///////////////////////////////////////////////////////////
    allocChannels(channelsNumber, imageWidth, imageHeight, bSubSampledX, bSubSampledY);
    std::uint32_t maxSamplingFactorX = bSubSampledX ? 2 : 1;
    std::uint32_t maxSamplingFactorY = bSubSampledY ? 2 : 1;
    for(std::uint32_t copyChannels = 0; copyChannels < channelsNumber; ++copyChannels)
    {
        ptrChannel dicomChannel = m_channels[copyChannels];
        imageHandler->copyToInt32Interleaved(
                    dicomChannel->m_pBuffer,
                    maxSamplingFactorX /dicomChannel->m_samplingFactorX,
                    maxSamplingFactorY /dicomChannel->m_samplingFactorY,
                    0, 0,
                    dicomChannel->m_width * maxSamplingFactorX / dicomChannel->m_samplingFactorX,
                    dicomChannel->m_height * maxSamplingFactorY / dicomChannel->m_samplingFactorY,
                    copyChannels,
                    imageWidth,
                    imageHeight,
                    channelsNumber);
    }

    std::uint32_t mask = (std::uint32_t)(((std::uint64_t)1 << (highBit + 1)) - 1);

    if(bRleCompressed)
    {
        writeRLECompressed(
                    imageWidth,
                    imageHeight,
                    channelsNumber,
                    pDestStream.get(),
                    allocatedBits,
                    mask);
        return;
    }

    std::uint8_t wordSizeBytes = ((dataType == tagVR_t::OW) || (dataType == tagVR_t::SS) || (dataType == tagVR_t::US)) ? 2 : 1;

    if(bInterleaved || channelsNumber == 1)
    {
        writeUncompressedInterleaved(
                    channelsNumber,
                    bSubSampledX, bSubSampledY,
                    pDestStream.get(),
                    wordSizeBytes,
                    allocatedBits,
                    mask);
        return;
    }

    writeUncompressedNotInterleaved(
                channelsNumber,
                pDestStream.get(),
                wordSizeBytes,
                allocatedBits,
                mask);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Returns true if the codec can handle the transfer
//  syntax
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
bool dicomCodec::canHandleTransferSyntax(const std::string& transferSyntax) const
{
    IMEBRA_FUNCTION_START();

    return(
                transferSyntax == "1.2.840.10008.1.2" ||      // Implicit VR little endian
                transferSyntax == "1.2.840.10008.1.2.1" ||    // Explicit VR little endian
                // transferSyntax=="1.2.840.10008.1.2.1.99" || // Deflated explicit VR little endian
                transferSyntax == "1.2.840.10008.1.2.2" ||    // Explicit VR big endian
                transferSyntax == "1.2.840.10008.1.2.5");     // RLE compression

    IMEBRA_FUNCTION_END();
}


////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
//
//
// Returns true if the transfer syntax has to be
//  encapsulated
//
//
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
bool dicomCodec::encapsulated(const std::string& transferSyntax) const
{
    IMEBRA_FUNCTION_START();

    if(!canHandleTransferSyntax(transferSyntax))
    {
        IMEBRA_THROW(CodecWrongTransferSyntaxError, "Cannot handle the transfer syntax");
    }
    return (transferSyntax == "1.2.840.10008.1.2.5");

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Suggest the number of allocated bits
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t dicomCodec::suggestAllocatedBits(const std::string& transferSyntax, std::uint32_t highBit) const
{
    IMEBRA_FUNCTION_START();

    if(transferSyntax == "1.2.840.10008.1.2.5")
    {
        return (highBit + 8) & 0xfffffff8;
    }

    return highBit + 1;

    IMEBRA_FUNCTION_END();

}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read a single tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint32_t dicomCodec::readTag(
        std::shared_ptr<streamReader> pStream,
        std::shared_ptr<dataSet> pDataSet,
        std::uint32_t tagLengthDWord,
        std::uint16_t tagId,
        std::uint16_t order,
        std::uint16_t tagSubId,
        tagVR_t tagType,
        streamController::tByteOrdering endianType,
        size_t wordSize,
        std::uint32_t bufferId,
        std::uint32_t maxSizeBufferLoad /* = 0xffffffff */
        )
{
    IMEBRA_FUNCTION_START();

    // If the tag's size is bigger than the maximum loadable
    //  size then just specify in which file it resides
    ///////////////////////////////////////////////////////////
    if(tagLengthDWord > maxSizeBufferLoad)
    {
        size_t bufferPosition(pStream->position());
        size_t streamPosition(pStream->getControlledStreamPosition());
        pStream->seekForward(tagLengthDWord);
        size_t bufferLength(pStream->position() - bufferPosition);

        if(bufferLength != tagLengthDWord)
        {
            IMEBRA_THROW(CodecCorruptedFileError, "dicomCodec::readTag detected a corrupted tag");
        }

        std::shared_ptr<data> writeData (pDataSet->getTagCreate(tagId, order, tagSubId, tagType));
        std::shared_ptr<buffer> newBuffer(
                    std::make_shared<buffer>(
                        pStream->getControlledStream(),
                        streamPosition,
                        bufferLength,
                        wordSize,
                        endianType));

        writeData->setBuffer(bufferId, newBuffer);

        return (std::uint32_t)bufferLength;
    }

    // Allocate the tag's buffer
    ///////////////////////////////////////////////////////////
    std::shared_ptr<handlers::writingDataHandlerRaw> handler(pDataSet->getWritingDataHandlerRaw(tagId, order, tagSubId, bufferId, tagType));

    // Do nothing if the tag's size is 0
    ///////////////////////////////////////////////////////////
    if(tagLengthDWord == 0)
    {
        return 0;
    }

    // In order to deal with damaged tags asking for an
    //  incredible amount of memory, this function reads the
    //  tag using a lot of small buffers (32768 bytes max)
    //  and then the tag's buffer is rebuilt at the end of the
    //  function.
    // This method saves a lot of time if a huge amount of
    //  memory is asked by a damaged tag, since only the amount
    //  of memory actually stored in the source file is
    //  allocated
    ///////////////////////////////////////////////////////////

    // If the buffer size is bigger than the following const
    //  variable, then read the buffer in small chunks
    ///////////////////////////////////////////////////////////
    const std::uint32_t smallBuffersSize(32768);

    if(tagLengthDWord <= smallBuffersSize) // Read in one go
    {
        handler->setSize(tagLengthDWord);
        pStream->read(handler->getMemoryBuffer(), tagLengthDWord);
    }
    else // Read in small chunks
    {
        std::list<std::vector<std::uint8_t> > buffers;

        // Used to keep track of the read bytes
        ///////////////////////////////////////////////////////////
        std::uint32_t remainingBytes(tagLengthDWord);

        // Fill all the small buffers
        ///////////////////////////////////////////////////////////
        while(remainingBytes != 0)
        {
            // Calculate the small buffer's size and allocate it
            ///////////////////////////////////////////////////////////
            std::uint32_t thisBufferSize( (remainingBytes > smallBuffersSize) ? smallBuffersSize : remainingBytes);
            buffers.push_back(std::vector<std::uint8_t>());
            buffers.back().resize(thisBufferSize);

            // Fill the buffer
            ///////////////////////////////////////////////////////////
            pStream->read(&buffers.back()[0], thisBufferSize);

            // Decrease the number of the remaining bytes
            ///////////////////////////////////////////////////////////
            remainingBytes -= thisBufferSize;
        }

        // Copy the small buffers into the tag object
        ///////////////////////////////////////////////////////////
        handler->setSize(tagLengthDWord);
        std::uint8_t* pHandlerBuffer(handler->getMemoryBuffer());

        // Scan all the small buffers and copy their content into
        //  the final buffer
        ///////////////////////////////////////////////////////////
        std::list<std::vector<std::uint8_t> >::iterator smallBuffersIterator;
        remainingBytes = tagLengthDWord;
        for(smallBuffersIterator=buffers.begin(); smallBuffersIterator != buffers.end(); ++smallBuffersIterator)
        {
            std::uint32_t copySize=(remainingBytes>smallBuffersSize) ? smallBuffersSize : remainingBytes;
            ::memcpy(pHandlerBuffer, &(*smallBuffersIterator)[0], copySize);
            pHandlerBuffer += copySize;
            remainingBytes -= copySize;
        }
    } // end of reading from stream

    // All the bytes have been read, now rebuild the tag's
    //  buffer. Don't rebuild the tag if it is 0xfffc,0xfffc
    //  (end of the stream)
    ///////////////////////////////////////////////////////////
    if(tagId == 0xfffc && tagSubId == 0xfffc)
    {
        return (std::uint32_t)tagLengthDWord;
    }

    // Adjust the buffer's byte endian
    ///////////////////////////////////////////////////////////
    if(wordSize != 0)
    {
        pStream->adjustEndian(handler->getMemoryBuffer(), wordSize, endianType, tagLengthDWord / wordSize);
    }

    // Return the tag's length in bytes
    ///////////////////////////////////////////////////////////
    return (std::uint32_t)tagLengthDWord;

    IMEBRA_FUNCTION_END();
}

} // namespace codecs

} // namespace implementation

} // namespace imebra

