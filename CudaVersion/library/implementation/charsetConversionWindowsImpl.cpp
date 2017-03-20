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

/*! \file charsetConversionWindows.cpp
    \brief Implementation of the charsetConversion class on Windows.

*/

#include "configurationImpl.h"

#if defined(IMEBRA_USE_WINDOWS_CHARSET)

#include "exceptionImpl.h"
#include "charsetConversionWindowsImpl.h"
#include <memory>

namespace imebra
{

///////////////////////////////////////////////////////////
//
// Constructor
//
///////////////////////////////////////////////////////////
charsetConversionWindows::charsetConversionWindows(const std::string& dicomName)
{
    IMEBRA_FUNCTION_START();

    const charsetInformation& info(getDictionary().getCharsetInformation(dicomName));

    m_codePage = info.m_codePage;
    m_bZeroFlag = info.m_bZeroFlag;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Convert a string from unicode to multibyte
//
///////////////////////////////////////////////////////////
std::string charsetConversionWindows::fromUnicode(const std::wstring& unicodeString) const
{
    IMEBRA_FUNCTION_START();

	if(unicodeString.empty())
	{
		return std::string();
	}

	BOOL bUsedDefault = false;
	int requiredChars = ::WideCharToMultiByte(m_codePage, m_bZeroFlag ? 0 : WC_NO_BEST_FIT_CHARS | WC_COMPOSITECHECK | WC_DEFAULTCHAR, unicodeString.c_str(), (int)(unicodeString.length()), 0, 0, 0, m_bZeroFlag ? 0 : &bUsedDefault);
    if(requiredChars <= 0)
    {
        return "";
    }
    std::string returnString((size_t)requiredChars, char(0));
    ::WideCharToMultiByte(m_codePage, m_bZeroFlag ? 0 : WC_NO_BEST_FIT_CHARS | WC_COMPOSITECHECK | WC_DEFAULTCHAR, unicodeString.c_str(), (int)(unicodeString.length()), &(returnString[0]), requiredChars, 0, m_bZeroFlag ? 0 : &bUsedDefault);
	if(bUsedDefault)
	{
		return std::string();
	}
	return returnString;

	IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
//
// Convert a string from multibyte to unicode
//
///////////////////////////////////////////////////////////
std::wstring charsetConversionWindows::toUnicode(const std::string& asciiString) const
{
    IMEBRA_FUNCTION_START();

	if(asciiString.empty())
	{
		return std::wstring();
	}

	int requiredWChars = ::MultiByteToWideChar(m_codePage, 0, asciiString.c_str(), (int)(asciiString.length()), 0, 0);
    if(requiredWChars <= 0)
    {
        return L"";
    }
    std::wstring returnString((size_t)requiredWChars, char(0));
    ::MultiByteToWideChar(m_codePage, 0, asciiString.c_str(), (int)(asciiString.length()), &(returnString[0]), requiredWChars);
    return returnString;

	IMEBRA_FUNCTION_END();
}

} // namespace imebra

#endif
