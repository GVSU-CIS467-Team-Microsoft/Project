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

/*! \file dataHandlerStringUnicode.cpp
    \brief Implementation of the base class used by the string handlers that need
	        to handle several charsets.

*/

#include "exceptionImpl.h"
#include "dataHandlerStringUnicodeImpl.h"
#include "memoryImpl.h"
#include "bufferImpl.h"

namespace imebra
{

namespace implementation
{

namespace handlers
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// dataHandlerStringUnicode
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

readingDataHandlerStringUnicode::readingDataHandlerStringUnicode(const memory& parseMemory, const charsetsList::tCharsetsList& charsets, tagVR_t dataType, const wchar_t separator, const std::uint8_t paddingByte):
    readingDataHandler(dataType)
{
    IMEBRA_FUNCTION_START();

    std::string asciiString((const char*)parseMemory.data(), parseMemory.size());
    std::wstring parseString(dicomConversion::convertToUnicode(asciiString, charsets));

    while(!parseString.empty() && parseString.back() == (wchar_t)paddingByte)
    {
        parseString.pop_back();
    }

    if(separator == 0)
    {
        m_strings.push_back(parseString);
        return;
    }

    for(size_t firstPosition(0); ; )
    {
        size_t nextPosition = parseString.find(separator, firstPosition);
        if(nextPosition == std::string::npos)
        {
            m_strings.push_back(parseString.substr(firstPosition));
            return;
        }
        m_strings.push_back(parseString.substr(firstPosition, nextPosition - firstPosition));
        firstPosition = ++nextPosition;
    }

    IMEBRA_FUNCTION_END();

}

// Get the data element as a signed long
///////////////////////////////////////////////////////////
std::int32_t readingDataHandlerStringUnicode::getSignedLong(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    std::wistringstream conversion(getUnicodeString(index));
    std::int32_t value;
    if(!(conversion >> value))
    {
        IMEBRA_THROW(DataHandlerConversionError, "The string is not a number");
    }
    return value;

    IMEBRA_FUNCTION_END();
}

// Get the data element as an unsigned long
///////////////////////////////////////////////////////////
std::uint32_t readingDataHandlerStringUnicode::getUnsignedLong(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    std::wistringstream conversion(getUnicodeString(index));
    std::uint32_t value;
    if(!(conversion >> value))
    {
        IMEBRA_THROW(DataHandlerConversionError, "The string is not a number");
    }
    return value;

    IMEBRA_FUNCTION_END();
}

// Get the data element as a double
///////////////////////////////////////////////////////////
double readingDataHandlerStringUnicode::getDouble(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    std::wistringstream conversion(getUnicodeString(index));
    double value;
    if(!(conversion >> value))
    {
        IMEBRA_THROW(DataHandlerConversionError, "The string is not a number");
    }
    return value;

    IMEBRA_FUNCTION_END();
}

// Get the data element as a string
///////////////////////////////////////////////////////////
std::string readingDataHandlerStringUnicode::getString(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    charsetsList::tCharsetsList charsets;
    charsets.push_back("ISO_IR 192");
    return dicomConversion::convertFromUnicode(getUnicodeString(index), &charsets);

    IMEBRA_FUNCTION_END();
}

// Get the data element as an unicode string
///////////////////////////////////////////////////////////
std::wstring readingDataHandlerStringUnicode::getUnicodeString(const size_t index) const
{
    IMEBRA_FUNCTION_START();

    if(index >= getSize())
    {
        IMEBRA_THROW(MissingItemError, "Missing item " << index);
    }
    return m_strings.at(index);

    IMEBRA_FUNCTION_END();
}

// Retrieve the data element as a string
///////////////////////////////////////////////////////////
size_t readingDataHandlerStringUnicode::getSize() const
{
    IMEBRA_FUNCTION_START();

    return m_strings.size();

    IMEBRA_FUNCTION_END();
}

writingDataHandlerStringUnicode::writingDataHandlerStringUnicode(const std::shared_ptr<buffer> &pBuffer, const charsetsList::tCharsetsList& charsets, tagVR_t dataType, const wchar_t separator, const size_t unitSize, const size_t maxSize, const uint8_t paddingByte):
    writingDataHandler(pBuffer, dataType, paddingByte), m_charsets(charsets), m_separator(separator), m_unitSize(unitSize), m_maxSize(maxSize)
{
}

writingDataHandlerStringUnicode::~writingDataHandlerStringUnicode()
{
    IMEBRA_FUNCTION_START();

    std::wstring completeString;
    for(size_t stringsIterator(0); stringsIterator != m_strings.size(); ++stringsIterator)
    {
        if(stringsIterator != 0)
        {
            completeString += m_separator;
        }
        completeString += m_strings.at(stringsIterator);
    }

    std::string asciiString = dicomConversion::convertFromUnicode(completeString, &m_charsets);

    std::shared_ptr<memory> commitMemory = std::make_shared<memory>(asciiString.size());
    commitMemory->assign((std::uint8_t*)asciiString.data(), asciiString.size());

    // The buffer's size must be an even number
    ///////////////////////////////////////////////////////////
    size_t memorySize = commitMemory->size();
    if((memorySize & 0x1) != 0)
    {
        commitMemory->resize(++memorySize);
        *(commitMemory->data() + (memorySize - 1)) = m_paddingByte;
    }

    m_buffer->commit(commitMemory, m_charsets);

    IMEBRA_FUNCTION_END();
}

// Set the data element as a signed long
///////////////////////////////////////////////////////////
void writingDataHandlerStringUnicode::setSignedLong(const size_t index, const std::int32_t value)
{
    IMEBRA_FUNCTION_START();

    std::wostringstream conversion;
    conversion << value;
    setUnicodeString(index, conversion.str());

    IMEBRA_FUNCTION_END();
}

// Set the data element as an unsigned long
///////////////////////////////////////////////////////////
void writingDataHandlerStringUnicode::setUnsignedLong(const size_t index, const std::uint32_t value)
{
    IMEBRA_FUNCTION_START();

    std::wostringstream conversion;
    conversion << value;
    setUnicodeString(index, conversion.str());

    IMEBRA_FUNCTION_END();
}

// Set the data element as a double
///////////////////////////////////////////////////////////
void writingDataHandlerStringUnicode::setDouble(const size_t index, const double value)
{
    IMEBRA_FUNCTION_START();

    std::wostringstream conversion;
    conversion << value;
    setUnicodeString(index, conversion.str());

    IMEBRA_FUNCTION_END();
}

// Set the buffer's size, in data elements
///////////////////////////////////////////////////////////
void writingDataHandlerStringUnicode::setSize(const size_t elementsNumber)
{
    IMEBRA_FUNCTION_START();

    m_strings.resize(elementsNumber);

    IMEBRA_FUNCTION_END();
}

size_t writingDataHandlerStringUnicode::getSize() const
{
    IMEBRA_FUNCTION_START();

    return m_strings.size();

    IMEBRA_FUNCTION_END();
}

void writingDataHandlerStringUnicode::setString(const size_t index, const std::string& value)
{
    IMEBRA_FUNCTION_START();

    charsetsList::tCharsetsList charsets;
    charsets.push_back("ISO_IR 192");
    setUnicodeString(index, dicomConversion::convertToUnicode(value, charsets));

    IMEBRA_FUNCTION_END();
}

void writingDataHandlerStringUnicode::setUnicodeString(const size_t index, const std::wstring& value)
{
    IMEBRA_FUNCTION_START();

    if(m_separator == 0 && index != 0)
    {
        IMEBRA_THROW(DataHandlerInvalidDataError, "Cannot insert more than one item in this string tag");
    }
    if(index >= getSize())
    {
        setSize(index + 1);
    }
    m_strings[index] = value;

    validate();

    IMEBRA_FUNCTION_END();
}

void writingDataHandlerStringUnicode::validate() const
{
    IMEBRA_FUNCTION_START();

    validateStringContainer(m_strings, m_maxSize, m_unitSize, m_separator != 0);

    IMEBRA_FUNCTION_END();
}




} // namespace handlers

} // namespace implementation

} // namespace imebra
