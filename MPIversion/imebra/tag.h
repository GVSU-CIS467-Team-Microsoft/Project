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

/*! \file tag.h
    \brief Declaration of the class Tag.

*/

#if !defined(imebraTagContent__INCLUDED_)
#define imebraTagContent__INCLUDED_

#include <string>
#include <cstdint>
#include <memory>
#include <map>
#include "image.h"
#include "readingDataHandlerNumeric.h"
#include "writingDataHandlerNumeric.h"
#include "definitions.h"
#include "streamReader.h"
#include "streamWriter.h"
#include "dataSet.h"

#ifndef SWIG

namespace imebra
{
namespace implementation
{
class data;
}
}

#endif

namespace imebra
{

///
/// \brief This class represents a DICOM tag.
///
///////////////////////////////////////////////////////////////////////////////
class IMEBRA_API Tag
{
    Tag(const Tag&) = delete;
    Tag& operator=(const Tag&) = delete;

#ifndef SWIG
    friend class DataSet;
private:
    Tag(std::shared_ptr<imebra::implementation::data> pData);
#endif

public:
    virtual ~Tag();

    /// \brief Returns the number of buffers in the tag.
    ///
    /// \return the number of buffers in the tag
    ///
    ///////////////////////////////////////////////////////////////////////////////
    size_t getBuffersCount() const;

    /// \brief Returns true if the specified buffer exists, otherwise it returns
    ///        false.
    ///
    /// \param bufferId the zero-based buffer's id the
    ///                 function has to check for
    /// \return true if the buffer exists, false otherwise
    ///
    ///////////////////////////////////////////////////////////////////////////////
    bool bufferExists(size_t bufferId) const;

    /// \brief Returns the size of a buffer, in bytes.
    ///
    /// If the buffer doesn't exist then throws MissingBufferError.
    ///
    /// \param bufferId the zero-based buffer's id the
    ///                 function has to check for
    /// \return the buffer's size in bytes
    ///
    ///////////////////////////////////////////////////////////////////////////////
    size_t getBufferSize(size_t bufferId) const;

    /// \brief Retrieve a ReadingDataHandler object connected to a specific
    ///        buffer.
    ///
    /// If the specified buffer does not exist then throws or MissingBufferError.
    ///
    /// \param bufferId the buffer to connect to the ReadingDataHandler object.
    ///                 The first buffer has an Id = 0
    /// \return a ReadingDataHandler object connected to the requested buffer
    ///
    ///////////////////////////////////////////////////////////////////////////////
    ReadingDataHandler* getReadingDataHandler(size_t bufferId) const;

    /// \brief Retrieve a WritingDataHandler object connected to a specific
    ///        tag's buffer.
    ///
    /// If the specified Tag does not exist then it creates a new tag with the VR
    ///  specified in the tagVR parameter
    ///
    /// The returned WritingDataHandler is connected to a new buffer which is
    /// updated and stored in the tag when WritingDataHandler is destroyed.
    ///
    /// \param bufferId the position where the new buffer has to be stored into the
    ///                 tag. The first buffer position is 0
    /// \return a WritingDataHandler object connected to a new Tag's buffer
    ///
    ///////////////////////////////////////////////////////////////////////////////
    WritingDataHandler* getWritingDataHandler(size_t bufferId);

    /// \brief Retrieve a ReadingDataHandlerNumeric object connected to the
    ///        Tag's numeric buffer.
    ///
    /// If the tag's VR is not a numeric type then throws std::bad_cast.
    ///
    /// If the specified Tag does not contain the specified buffer then
    ///  throws MissingBufferError.
    ///
    /// \param bufferId the buffer to connect to the ReadingDataHandler object.
    ///                 The first buffer has an Id = 0
    /// \return a ReadingDataHandlerNumeric object connected to the Tag's buffer
    ///
    ///////////////////////////////////////////////////////////////////////////////
    ReadingDataHandlerNumeric* getReadingDataHandlerNumeric(size_t bufferId) const;

    /// \brief Retrieve a ReadingDataHandlerNumeric object connected to the
    ///        Tag's raw data buffer (8 bit unsigned integers).
    ///
    /// If the tag's VR is not a numeric type then throws std::bad_cast.
    ///
    /// If the specified Tag does not contain the specified buffer then
    ///  throws MissingBufferError.
    ///
    /// \param bufferId the buffer to connect to the ReadingDataHandler object.
    ///                 The first buffer has an Id = 0
    /// \return a ReadingDataHandlerNumeric object connected to the Tag's buffer
    ///         (raw content represented by 8 bit unsigned integers)
    ///
    ///////////////////////////////////////////////////////////////////////////////
    ReadingDataHandlerNumeric* getReadingDataHandlerRaw(size_t bufferId) const;

    /// \brief Retrieve a WritingDataHandlerNumeric object connected to the
    ///        Tag's buffer.
    ///
    /// If the tag's VR is not a numeric type then throws std::bad_cast.
    ///
    /// The returned WritingDataHandlerNumeric is connected to a new buffer which
    /// is updated and stored into the tag when WritingDataHandlerNumeric is
    /// destroyed.
    ///
    /// \param bufferId the position where the new buffer has to be stored in the
    ///                 tag. The first buffer position is 0
    /// \return a WritingDataHandlerNumeric object connected to a new Tag's buffer
    ///
    ///////////////////////////////////////////////////////////////////////////////
    WritingDataHandlerNumeric* getWritingDataHandlerNumeric(size_t bufferId);

    /// \brief Retrieve a WritingDataHandlerNumeric object connected to the
    ///        Tag's raw data buffer (8 bit unsigned integers).
    ///
    /// If the tag's VR is not a numeric type then throws std::bad_cast.
    ///
    /// The returned WritingDataHandlerNumeric is connected to a new buffer which
    /// is updated and stored into the tag when WritingDataHandlerNumeric is
    /// destroyed.
    ///
    /// \param bufferId the position where the new buffer has to be stored in the
    ///                 tag. The first buffer position is 0
    /// \return a WritingDataHandlerNumeric object connected to a new Tag's buffer
    ///         (the buffer contains raw data of 8 bit integers)
    ///
    ///////////////////////////////////////////////////////////////////////////////
    WritingDataHandlerNumeric* getWritingDataHandlerRaw(size_t bufferId);

    /// \brief Get a StreamReader connected to a buffer's data.
    ///
    /// \param bufferId   the id of the buffer for which the StreamReader is
    ///                    required. This parameter is usually 0
    /// \return           the streamReader connected to the buffer's data.
    ///
    ///////////////////////////////////////////////////////////////////////////////
    StreamReader* getStreamReader(size_t bufferId);

    /// \brief Get a StreamWriter connected to a buffer's data.
    ///
    /// @param bufferId   the id of the buffer for which the StreamWriter is
    ///                    required. This parameter is usually 0
    /// @return           the StreamWriter connected to the buffer's data.
    ///
    ///////////////////////////////////////////////////////////////////////////////
    StreamWriter* getStreamWriter(size_t bufferId);

    /// \brief Retrieve an embedded DataSet.
    ///
    /// Sequence tags (VR=SQ) store embedded dicom structures.
    ///
    /// @param dataSetId  the ID of the sequence item to retrieve. Several sequence
    ///                   items can be embedded in one tag.
    /// @return           the sequence DataSet
    ///
    ///////////////////////////////////////////////////////////////////////////////
    DataSet* getSequenceItem(size_t dataSetId) const;

    /// \brief Check for the existance of a sequence item.
    ///
    /// \param dataSetId the ID of the sequence item to check for
    /// \return true if the sequence item exists, false otherwise
    ///
    ///////////////////////////////////////////////////////////////////////////////
    bool sequenceItemExists(size_t dataSetId) const;

    /// \brief Insert a sequence item into the Tag.
    ///
    /// Several sequence items can be nested one inside each other.
    /// When a sequence item is embedded into a Tag, then the tag will have a
    /// sequence VR (VR = SQ).
    ///
    /// \param dataSetId  the ID of the sequence item
    /// \param dataSet    the DataSet containing the sequence item data
    ///
    ///////////////////////////////////////////////////////////////////////////////
    void setSequenceItem(size_t dataSetId, const DataSet& dataSet);

    /// \brief Append a sequence item into the Tag.
    ///
    /// Several sequence items can be nested one inside each other.
    /// When a sequence item is embedded into a Tag, then the tag will have a
    /// sequence VR (VR = SQ).
    ///
    /// \param dataSet    the DataSet containing the sequence item data
    ///
    ///////////////////////////////////////////////////////////////////////////////
    void appendSequenceItem(const DataSet& dataSet);

    /// \brief Get the tag's data type.
    ///
    /// @return the tag's data type
    ///
    ///////////////////////////////////////////////////////////////////////////////
    tagVR_t getDataType() const;

#ifndef SWIG
protected:
    std::shared_ptr<imebra::implementation::data> m_pData;
#endif
};

}

#endif // !defined(imebraTagContent__INCLUDED_)
