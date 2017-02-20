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

/*! \file jpegCodec.h
    \brief Declaration of the class jpegCodec.

*/

#if !defined(imebraJpegCodec_7F63E846_8824_42c6_A048_DD59C657AED4__INCLUDED_)
#define imebraJpegCodec_7F63E846_8824_42c6_A048_DD59C657AED4__INCLUDED_

#include "codecImpl.h"
#include <map>
#include <list>


namespace imebra
{

class huffmanTable;

namespace implementation
{

namespace codecs
{

/// \addtogroup group_codecs
///
/// @{

// The following classes are used in the jpegCodec
//  declaration
///////////////////////////////////////////////////////////
namespace jpeg
{
	class tag;
	class jpegChannel;
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief The Jpeg codec.
///
/// This class is used to decode and encode a Jpeg stream.
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class jpegCodec : public codec
{
	friend class jpeg::tag;

public:
	// Constructor
	///////////////////////////////////////////////////////////
	jpegCodec();

	// Allocate the image's channels
	///////////////////////////////////////////////////////////
	void allocChannels();

	// Find the mcu's size
	///////////////////////////////////////////////////////////
	void findMcuSize();

	// Recalculate the tables for dequantization/quantization
	void recalculateQuantizationTables(int table);


	// Erase the allocated channels
	///////////////////////////////////////////////////////////
	void eraseChannels();

	// Retrieve the image from a dataset
	///////////////////////////////////////////////////////////
    virtual std::shared_ptr<image> getImage(const dataSet& sourceDataSet, std::shared_ptr<streamReader> pStream, tagVR_t dataType);

	// Insert a jpeg compressed image into a dataset
	///////////////////////////////////////////////////////////
	virtual void setImage(
		std::shared_ptr<streamWriter> pDestStream,
		std::shared_ptr<image> pImage,
        const std::string& transferSyntax,
        imageQuality_t imageQuality,
        tagVR_t dataType,
        std::uint32_t allocatedBits,
		bool bSubSampledX,
		bool bSubSampledY,
		bool bInterleaved,
		bool b2Complement);

	// Return true if the codec can handle the transfer
	///////////////////////////////////////////////////////////
    virtual bool canHandleTransferSyntax(const std::string& transferSyntax) const;

	// Returns true if the transfer syntax has to be
	//  encapsulated
	//
	///////////////////////////////////////////////////////////
    virtual bool encapsulated(const std::string& transferSyntax) const;

	// Return the suggested allocated bits
	///////////////////////////////////////////////////////////
    virtual std::uint32_t suggestAllocatedBits(const std::string& transferSyntax, std::uint32_t highBit) const;

	// Create another jpeg codec
	///////////////////////////////////////////////////////////
	virtual std::shared_ptr<codec> createCodec();

protected:
	// Read a jpeg stream and build a Dicom dataset
	///////////////////////////////////////////////////////////
	virtual void readStream(std::shared_ptr<streamReader> pSourceStream, std::shared_ptr<dataSet> pDataSet, std::uint32_t maxSizeBufferLoad = 0xffffffff);

	// Write a Dicom dataset as a Jpeg stream
	///////////////////////////////////////////////////////////
	virtual void writeStream(std::shared_ptr<streamWriter> pSourceStream, std::shared_ptr<dataSet> pDataSet);

	///////////////////////////////////////////////////////////
	//
	// Image's attributes. Used while reading/writing an image
	//
	///////////////////////////////////////////////////////////
public:
	// The image's size, in pixels
	///////////////////////////////////////////////////////////
	std::uint32_t m_imageWidth;
	std::uint32_t m_imageHeight;

	// Encoding process
	///////////////////////////////////////////////////////////
	std::uint8_t  m_process;

	// The bits per color component
	///////////////////////////////////////////////////////////
    std::uint32_t m_precision;
	std::int32_t m_valuesMask;

	// true when the end of the image has been reached
	///////////////////////////////////////////////////////////
	bool m_bEndOfImage;

	// The allocated channels
	///////////////////////////////////////////////////////////
	typedef std::shared_ptr<jpeg::jpegChannel> ptrChannel;
	typedef std::map<std::uint8_t, ptrChannel> tChannelsMap;
	tChannelsMap m_channelsMap;

	// The list of the channels in the active scan, zero
	//  terminated
	///////////////////////////////////////////////////////////
	jpeg::jpegChannel* m_channelsList[257]; // 256 channels + terminator

	// Huffman tables
	///////////////////////////////////////////////////////////
	std::shared_ptr<huffmanTable> m_pHuffmanTableDC[16];
	std::shared_ptr<huffmanTable> m_pHuffmanTableAC[16];

	//
	// Quantization tables
	//
	///////////////////////////////////////////////////////////
	std::uint32_t m_quantizationTable[16][64];

	// The number of MCUs per restart interval
	///////////////////////////////////////////////////////////
	std::uint16_t m_mcuPerRestartInterval;

	// The number of processed MCUs
	///////////////////////////////////////////////////////////
	std::uint32_t m_mcuProcessed;
	std::uint32_t m_mcuProcessedX;
	std::uint32_t m_mcuProcessedY;

	// The length of the EOB run
	///////////////////////////////////////////////////////////
	std::uint32_t m_eobRun;

	// The last found restart interval
	///////////////////////////////////////////////////////////
	std::uint32_t m_mcuLastRestart;

	// Spectral index and progressive bits reading
	///////////////////////////////////////////////////////////
	std::uint32_t m_spectralIndexStart;
	std::uint32_t m_spectralIndexEnd;

	// true if we are reading a lossless jpeg image
	///////////////////////////////////////////////////////////
	bool m_bLossless;

	// The maximum sampling factor
	///////////////////////////////////////////////////////////
	std::uint32_t m_maxSamplingFactorX;
	std::uint32_t m_maxSamplingFactorY;

	// The number of MCUs (horizontal, vertical, total)
	///////////////////////////////////////////////////////////
	std::uint32_t m_mcuNumberX;
	std::uint32_t m_mcuNumberY;
	std::uint32_t m_mcuNumberTotal;


	// The image's size, rounded to accomodate all the MCUs
	///////////////////////////////////////////////////////////
	std::uint32_t m_jpegImageWidth;
	std::uint32_t m_jpegImageHeight;


	// FDCT/IDCT
	///////////////////////////////////////////////////////////
public:
	void FDCT(std::int32_t* pIOMatrix, float* pDescaleFactors);
	void IDCT(std::int32_t* pIOMatrix, long long* pScaleFactors);

protected:
	/// \internal
	/// \brief This enumeration contains the tags used by
	///         the jpeg codec
	///////////////////////////////////////////////////////////
	enum tTagId
	{
		unknown = 0xff,

		sof0 = 0xc0,
		sof1 = 0xc1,
		sof2 = 0xc2,
		sof3 = 0xc3,

		dht = 0xc4,

		sof5 = 0xc5,
		sof6 = 0xc6,
		sof7 = 0xc7,

		sof9 = 0xc9,
		sofA = 0xca,
		sofB = 0xcb,

		sofD = 0xcd,
		sofE = 0xce,
		sofF = 0xcf,

		rst0 = 0xd0,
		rst1 = 0xd1,
		rst2 = 0xd2,
		rst3 = 0xd3,
		rst4 = 0xd4,
		rst5 = 0xd5,
		rst6 = 0xd6,
		rst7 = 0xd7,

		eoi = 0xd9,
		sos = 0xda,
		dqt = 0xdb,

		dri = 0xdd
	};

	// Register a tag in the jpeg codec
	///////////////////////////////////////////////////////////
	void registerTag(tTagId tagId, std::shared_ptr<jpeg::tag> pTag);

	// Read a lossy block of pixels
	///////////////////////////////////////////////////////////
	inline void readBlock(streamReader* pStream, std::int32_t* pBuffer, jpeg::jpegChannel* pChannel);

	// Write a lossy block of pixels
	///////////////////////////////////////////////////////////
	inline void writeBlock(streamWriter* pStream, std::int32_t* pBuffer, jpeg::jpegChannel* pChannel, bool bCalcHuffman);

	// Reset the internal variables
	///////////////////////////////////////////////////////////
    void resetInternal(bool bCompression, imageQuality_t compQuality);

    std::shared_ptr<image> copyJpegChannelsToImage(bool b2complement, const std::string& colorSpace);
    void copyImageToJpegChannels(std::shared_ptr<image> sourceImage, bool b2complement, std::uint32_t allocatedBits, bool bSubSampledX, bool bSubSampledY);

	void writeScan(streamWriter* pDestinationStream, bool bCalcHuffman);

	void writeTag(streamWriter* pDestinationStream, tTagId tagId);

	long long m_decompressionQuantizationTable[16][64];
	float m_compressionQuantizationTable[16][64];

	// Map of the available Jpeg tags
	///////////////////////////////////////////////////////////
	typedef std::shared_ptr<jpeg::tag> ptrTag;
	typedef std::map<std::uint8_t, ptrTag> tTagsMap;
	tTagsMap m_tagsMap;

	// temporary matrix used by FDCT
	///////////////////////////////////////////////////////////
	float m_fdctTempMatrix[64];

	// temporary matrix used by IDCT
	///////////////////////////////////////////////////////////
	long long m_idctTempMatrix[64];
};


/// @}

namespace jpeg
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// CImbsJpegCodecChannel
// An image's channel
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class jpegChannel : public channel
{
public:
	// Constructor
	///////////////////////////////////////////////////////////
	jpegChannel():
		m_quantTable(0),
		m_blockMcuX(0),
		m_blockMcuY(0),
		m_blockMcuXY(0),
		m_lastDCValue(0),
		m_defaultDCValue(0),
		m_losslessPositionX(0),
		m_losslessPositionY(0),
		m_unprocessedAmplitudesCount(0),
		m_unprocessedAmplitudesPredictor(0),
		m_huffmanTableDC(0),
		m_huffmanTableAC(0),
		m_pActiveHuffmanTableDC(0),
		m_pActiveHuffmanTableAC(0),
		m_valuesMask(0){}

	// Quantization table's id
	///////////////////////////////////////////////////////////
	int m_quantTable;

	// Blocks per MCU
	///////////////////////////////////////////////////////////
    std::uint32_t m_blockMcuX;
    std::uint32_t m_blockMcuY;
    std::uint32_t m_blockMcuXY;

	// Last DC value
	///////////////////////////////////////////////////////////
	std::int32_t m_lastDCValue;

	// Default DC value
	///////////////////////////////////////////////////////////
	std::int32_t m_defaultDCValue;

	// Lossless position
	///////////////////////////////////////////////////////////
	std::uint32_t m_losslessPositionX;
	std::uint32_t m_losslessPositionY;

	std::int32_t m_unprocessedAmplitudesBuffer[1024];
	std::uint32_t m_unprocessedAmplitudesCount;
	std::uint32_t m_unprocessedAmplitudesPredictor;

	// Huffman tables' id
	///////////////////////////////////////////////////////////
	int m_huffmanTableDC;
	int m_huffmanTableAC;
	huffmanTable* m_pActiveHuffmanTableDC;
	huffmanTable* m_pActiveHuffmanTableAC;

	std::int32_t m_valuesMask;

	inline void addUnprocessedAmplitude(std::int32_t unprocessedAmplitude, std::uint32_t predictor, bool bMcuRestart)
	{
		if(bMcuRestart ||
			predictor != m_unprocessedAmplitudesPredictor ||
			m_unprocessedAmplitudesCount == sizeof(m_unprocessedAmplitudesBuffer) / sizeof(m_unprocessedAmplitudesBuffer[0]))
		{
			processUnprocessedAmplitudes();
			if(bMcuRestart)
			{
				m_unprocessedAmplitudesPredictor = 0;
				m_unprocessedAmplitudesBuffer[0] = unprocessedAmplitude + m_defaultDCValue;
			}
			else
			{
				m_unprocessedAmplitudesPredictor = predictor;
				m_unprocessedAmplitudesBuffer[0] = unprocessedAmplitude;
			}
			++m_unprocessedAmplitudesCount;
			return;
		}
		m_unprocessedAmplitudesBuffer[m_unprocessedAmplitudesCount++] = unprocessedAmplitude;
	}

	void processUnprocessedAmplitudes();
};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// jpegCodecTag
//
// The base class for all the jpeg tags
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tag
{
public:
	typedef std::shared_ptr<jpeg::jpegChannel> ptrChannel;

public:
	// Write the tag's content.
	// The function should call WriteLength first.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec)=0;

	// Read the tag's content. The function should call
	//  ReadLength first.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry)=0;

protected:
	// Write the tag's length
	///////////////////////////////////////////////////////////
	void writeLength(streamWriter* pStream, std::uint16_t length);

	// Read the tag's length
	///////////////////////////////////////////////////////////
    std::uint32_t readLength(streamReader* pStream);
};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// jpegCodecTagUnknown
//
// Read/write an unknown tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagUnknown: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// jpegCodecTagSOF
//
// Read/write a SOF tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagSOF: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read/write a DHT tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagDHT: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read/write a SOS tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagSOS: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read/write a DQT tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagDQT: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read/write a DRI tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagDRI: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read/write a RST tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagRST: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Read/write an EOI tag
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class tagEOI: public tag
{
public:
	// Write the tag's content.
	///////////////////////////////////////////////////////////
	virtual void writeTag(streamWriter* pStream, jpegCodec* pCodec);

	// Read the tag's content.
	///////////////////////////////////////////////////////////
	virtual void readTag(streamReader* pStream, jpegCodec* pCodec, std::uint8_t tagEntry);

};

} // namespace jpegCodec

} // namespace codecs

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraJpegCodec_7F63E846_8824_42c6_A048_DD59C657AED4__INCLUDED_)
