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

/*! \file dataHandlerStringPN.cpp
    \brief Implementation of the class dataHandlerStringPN.

*/

#include "dataHandlerStringPNImpl.h"

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
// dataHandlerStringPN
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

readingDataHandlerStringPN::readingDataHandlerStringPN(const memory& parseMemory, const charsetsList::tCharsetsList& charsets):
    readingDataHandlerStringUnicode(parseMemory, charsets, tagVR_t::PN, L'=', 0x20)
{
}

writingDataHandlerStringPN::writingDataHandlerStringPN(const std::shared_ptr<buffer>& pBuffer, const charsetsList::tCharsetsList& charsets):
    writingDataHandlerStringUnicode(pBuffer, charsets, tagVR_t::PN, L'=', 0, 0, 0x20)
{
}

void writingDataHandlerStringPN::validate() const
{
    IMEBRA_FUNCTION_START();

    if(m_strings.size() > 3)
    {
        IMEBRA_THROW(DataHandlerInvalidDataError, "A patient name can contain maximum 3 groups");
    }
    for(size_t scanGroups(0); scanGroups != m_strings.size(); ++scanGroups)
    {
        if(m_strings[scanGroups].size() > 64)
        {
            IMEBRA_THROW(DataHandlerInvalidDataError, "A patient name group can contain maximum 64 chars");
        }
    }

    writingDataHandlerStringUnicode::validate();

    IMEBRA_FUNCTION_END();
}

} // namespace handlers

} // namespace implementation

} // namespace imebra
