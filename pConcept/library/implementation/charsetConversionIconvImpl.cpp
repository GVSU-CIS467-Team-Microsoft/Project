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

/*! \file charsetConversionIconv.cpp
    \brief Implementation of the charsetConversion class using Iconv.

*/

#include "configurationImpl.h"

#if defined(IMEBRA_USE_ICONV)

#include "exceptionImpl.h"
#include "charsetConversionIconvImpl.h"
#include "../include/imebra/exceptions.h"

#include <memory>

namespace imebra
{

///////////////////////////////////////////////////////////
//
// Constructor
//
///////////////////////////////////////////////////////////
charsetConversionIconv::charsetConversionIconv(const std::string& dicomName)
{
    IMEBRA_FUNCTION_START();

    const charsetInformation& info(getDictionary().getCharsetInformation(dicomName));

    std::string toCodeIgnore(info.m_isoRegistration);
    toCodeIgnore += "//IGNORE";

    // Check little endian/big endian
    ///////////////////////////////////////////////////////////
    std::uint16_t m_endianCheck=0x00ff;
    const char* utfCode;
    static_assert(sizeof(wchar_t) == 2 || sizeof(wchar_t) == 4, "Unsupported UTF char size");
    if(sizeof(wchar_t) == 2)
    {
        utfCode = (*((std::uint8_t*)&m_endianCheck) == 0xff) ? "UTF-16LE" : "UTF-16BE";
    }
    else
    {
        utfCode = (*((std::uint8_t*)&m_endianCheck) == 0xff) ? "UTF-32LE" : "UTF-32BE";
    }

    m_iconvToUnicode = iconv_open(utfCode, info.m_isoRegistration.c_str());
    m_iconvFromUnicode = iconv_open(toCodeIgnore.c_str(), utfCode);
    if(m_iconvToUnicode == (iconv_t)-1 || m_iconvFromUnicode == (iconv_t)-1)
    {
        IMEBRA_THROW(CharsetConversionNoSupportedTableError, "Table " << dicomName << " not supported by the system");
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Destructor
//
///////////////////////////////////////////////////////////
charsetConversionIconv::~charsetConversionIconv()
{
    iconv_close(m_iconvToUnicode);
    iconv_close(m_iconvFromUnicode);
}

///////////////////////////////////////////////////////////
//
// Convert a string from unicode to multibyte
//
///////////////////////////////////////////////////////////
std::string charsetConversionIconv::fromUnicode(const std::wstring& unicodeString) const
{
    IMEBRA_FUNCTION_START();

	if(unicodeString.empty())
	{
		return std::string();
	}

	return myIconv(m_iconvFromUnicode, (char*)unicodeString.c_str(), unicodeString.length() * sizeof(wchar_t));

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Convert a string from multibyte to unicode
//
///////////////////////////////////////////////////////////
std::wstring charsetConversionIconv::toUnicode(const std::string& asciiString) const
{
    IMEBRA_FUNCTION_START();

	if(asciiString.empty())
	{
		return std::wstring();
	}

	std::string convertedString(myIconv(m_iconvToUnicode, (char*)asciiString.c_str(), asciiString.length()));
	std::wstring returnString((wchar_t*)convertedString.c_str(), convertedString.size() / sizeof(wchar_t));

    return returnString;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// In Posix systems executes an iconv
//
///////////////////////////////////////////////////////////
#if defined(IMEBRA_WINDOWS)
std::string charsetConversionIconv::myIconv(iconv_t context, const char* inputString, size_t inputStringLengthBytes) const
#else
std::string charsetConversionIconv::myIconv(iconv_t context, char* inputString, size_t inputStringLengthBytes) const
#endif
{
    IMEBRA_FUNCTION_START();

	std::string finalString;

	// Reset the state
	///////////////////////////////////////////////////////////
	iconv(context, 0, 0, 0, 0);

	// Buffer for the conversion
	///////////////////////////////////////////////////////////
	char conversionBuffer[1024];

	// Convert the string
	///////////////////////////////////////////////////////////
	for(size_t iconvReturn = (size_t)-1; iconvReturn == (size_t)-1;)
	{
		// Executes the conversion
		///////////////////////////////////////////////////////////
		size_t progressiveOutputBufferSize = sizeof(conversionBuffer);
		char* progressiveOutputBuffer(conversionBuffer);
		iconvReturn = iconv(context, &inputString, &inputStringLengthBytes, &progressiveOutputBuffer, &progressiveOutputBufferSize);

		// Update buffer's size
		///////////////////////////////////////////////////////////
		size_t outputLengthBytes = sizeof(conversionBuffer) - progressiveOutputBufferSize;

		finalString.append(conversionBuffer, outputLengthBytes);

		// Throw if an invalid sequence is found
		///////////////////////////////////////////////////////////
		if(iconvReturn == (size_t)-1 && errno != E2BIG)
		{
		    return std::string();
		}
	}

	return finalString;

	IMEBRA_FUNCTION_END();
}

} // namespace imebra

#endif
