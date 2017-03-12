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

/*! \file charsetConversionICU.cpp
    \brief Implementation of the charsetConversion class using the ICU library.

*/

#include "configurationImpl.h"
#if defined(IMEBRA_USE_ICU)

#include "exceptionImpl.h"
#include "charsetConversionICUImpl.h"
#include "../include/imebra/exceptions.h"
#include <memory>

namespace imebra
{

///////////////////////////////////////////////////////////
//
// Constructor
//
///////////////////////////////////////////////////////////
charsetConversionICU::charsetConversionICU(const std::string& dicomName)
{
    IMEBRA_FUNCTION_START();

    UErrorCode errorCode(U_ZERO_ERROR);
    const charsetInformation& info = getDictionary().getCharsetInformation(dicomName);

    m_pIcuConverter = ucnv_open(info.m_isoRegistration.c_str(), &errorCode);
    if(U_FAILURE(errorCode))
    {
        IMEBRA_THROW(CharsetConversionNoSupportedTableError, "ICU library returned error " << errorCode << " for table " << dicomName);
    }
    ucnv_setSubstChars(m_pIcuConverter, "?", 1, &errorCode);
    if(U_FAILURE(errorCode))
    {
        IMEBRA_THROW(CharsetConversionNoSupportedTableError, "ICU library returned error " << errorCode << " while setting the substitution char for table " << dicomName);
    }

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Destructor
//
///////////////////////////////////////////////////////////
charsetConversionICU::~charsetConversionICU()
{
    ucnv_close(m_pIcuConverter);
}



///////////////////////////////////////////////////////////
//
// Convert a string from unicode to multibyte
//
///////////////////////////////////////////////////////////
std::string charsetConversionICU::fromUnicode(const std::wstring& unicodeString) const
{
    IMEBRA_FUNCTION_START();

	if(unicodeString.empty())
	{
		return std::string();
	}

    UnicodeString unicodeStringConversion;
    switch(sizeof(wchar_t))
    {
    case 2:
        unicodeStringConversion = UnicodeString((UChar*)&(unicodeString[0]), (std::int32_t)unicodeString.size());
        break;
    case 4:
        unicodeStringConversion = UnicodeString::fromUTF32((UChar32*)&(unicodeString[0]), (std::int32_t)unicodeString.size());
        break;
    }
    UErrorCode errorCode(U_ZERO_ERROR);
    int32_t conversionLength = unicodeStringConversion.extract(0, 0, m_pIcuConverter, errorCode);
    errorCode = U_ZERO_ERROR;
    std::string returnString((size_t)conversionLength, char(0));
    unicodeStringConversion.extract(&(returnString[0]), conversionLength, m_pIcuConverter, errorCode);
    if(U_FAILURE(errorCode))
    {
        IMEBRA_THROW(CharsetConversionError, "ICU library returned error " << errorCode);
    }
    if(returnString == "?" && unicodeString != L"?")
    {
        return "";
    }
    return returnString;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Convert a string from multibyte to unicode
//
///////////////////////////////////////////////////////////
std::wstring charsetConversionICU::toUnicode(const std::string& asciiString) const
{
    IMEBRA_FUNCTION_START();

	if(asciiString.empty())
	{
		return std::wstring();
	}

    UErrorCode errorCode(U_ZERO_ERROR);
    UnicodeString unicodeString(&(asciiString[0]), (std::int32_t)asciiString.size(), m_pIcuConverter, errorCode);
    switch(sizeof(wchar_t))
    {
    case 2:
    {
        std::wstring returnString((size_t)unicodeString.length(), wchar_t(0));
        unicodeString.extract((UChar*)&(returnString[0]), unicodeString.length(), errorCode);
        return returnString;
    }
    case 4:
    {
        int32_t conversionLength = unicodeString.toUTF32((UChar32*)0, (int32_t)0, errorCode);
        errorCode = U_ZERO_ERROR;
        std::wstring returnString((size_t)conversionLength, wchar_t(0));
        unicodeString.toUTF32((UChar32*)&(returnString[0]), conversionLength, errorCode);
        return returnString;
    }
    }

	IMEBRA_FUNCTION_END();
}



} // namespace imebra



#endif
