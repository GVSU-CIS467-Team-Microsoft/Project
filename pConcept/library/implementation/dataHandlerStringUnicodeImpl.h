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

/*! \file dataHandlerStringUnicode.h
    \brief Declaration of the base class used by the string handlers that need to work
	        with different charsets.

*/

#if !defined(imebraDataHandlerStringUnicode_367AAE47_6FD7_4107_AB5B_25A355C5CB6E__INCLUDED_)
#define imebraDataHandlerStringUnicode_367AAE47_6FD7_4107_AB5B_25A355C5CB6E__INCLUDED_

#include "charsetConversionImpl.h"
#include "dataHandlerImpl.h"
#include "charsetsListImpl.h"
#include <memory>
#include <vector>
#include <string>


namespace imebra
{

namespace implementation
{

namespace handlers
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief This is the base class for all the data handlers
///         that manage strings.
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class readingDataHandlerStringUnicode : public readingDataHandler
{
public:
    readingDataHandlerStringUnicode(const memory& parseMemory, const charsetsList::tCharsetsList& charsets, tagVR_t dataType, const wchar_t separator, const std::uint8_t paddingByte);

    // Get the data element as a signed long
    ///////////////////////////////////////////////////////////
    virtual std::int32_t getSignedLong(const size_t index) const;

    // Get the data element as an unsigned long
    ///////////////////////////////////////////////////////////
    virtual std::uint32_t getUnsignedLong(const size_t index) const;

    // Get the data element as a double
    ///////////////////////////////////////////////////////////
    virtual double getDouble(const size_t index) const;

    // Get the data element as a string
    ///////////////////////////////////////////////////////////
    virtual std::string getString(const size_t index) const;

    // Get the data element as an unicode string
    ///////////////////////////////////////////////////////////
    virtual std::wstring getUnicodeString(const size_t index) const;

    // Retrieve the data element as a string
    ///////////////////////////////////////////////////////////
    virtual size_t getSize() const;

protected:

    std::vector<std::wstring> m_strings;
};


class writingDataHandlerStringUnicode : public writingDataHandler
{
public:
    writingDataHandlerStringUnicode(const std::shared_ptr<buffer>& pBuffer, const charsetsList::tCharsetsList& charsets, tagVR_t dataType, const wchar_t separator, const size_t unitSize, const size_t maxSize, const std::uint8_t paddingByte);

    ~writingDataHandlerStringUnicode();

    // Set the data element as a signed long
    ///////////////////////////////////////////////////////////
    virtual void setSignedLong(const size_t index, const std::int32_t value);

    // Set the data element as an unsigned long
    ///////////////////////////////////////////////////////////
    virtual void setUnsignedLong(const size_t index, const std::uint32_t value);

    // Set the data element as a double
    ///////////////////////////////////////////////////////////
    virtual void setDouble(const size_t index, const double value);

    // Set the buffer's size, in data elements
    ///////////////////////////////////////////////////////////
    virtual void setSize(const size_t elementsNumber);

    virtual size_t getSize() const;

    virtual void setString(const size_t index, const std::string& value);

    virtual void setUnicodeString(const size_t index, const std::wstring& value);

    // Throw an exception if the content is not valid
    ///////////////////////////////////////////////////////////
    virtual void validate() const;

protected:
    std::vector<std::wstring> m_strings;

    charsetsList::tCharsetsList m_charsets;

    wchar_t m_separator;
    size_t m_unitSize;
    size_t m_maxSize;


};


} // namespace handlers

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraDataHandlerStringUnicode_367AAE47_6FD7_4107_AB5B_25A355C5CB6E__INCLUDED_)
