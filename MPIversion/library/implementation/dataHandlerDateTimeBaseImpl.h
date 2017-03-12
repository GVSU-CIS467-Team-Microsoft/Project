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

/*! \file dataHandlerDateTimeBase.h
    \brief Declaration of the base class for the time/date handlers.

*/

#if !defined(imebraDataHandlerDateTimeBase_85665C7B_8DDF_479e_8CC0_83E95CB625DC__INCLUDED_)
#define imebraDataHandlerDateTimeBase_85665C7B_8DDF_479e_8CC0_83E95CB625DC__INCLUDED_

#include "dataHandlerStringImpl.h"


namespace imebra
{

namespace implementation
{

namespace handlers
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief This class is used as base class by the handlers
///         that manage the date and the time
///
/// This class supplies the methods setSignedLong(), 
///  setUnsignedLong(), setDouble(), getSignedLong(),
///  getUnsignedLong(), getDouble(). Those methods work
///  with time_t structure
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class readingDataHandlerDateTimeBase : public readingDataHandlerString
{

public:
    readingDataHandlerDateTimeBase(const memory& parseMemory, tagVR_t dataType);

    virtual std::int32_t getSignedLong(const size_t index) const;
    virtual std::uint32_t getUnsignedLong(const size_t index) const;
    virtual double getDouble(const size_t index) const;

protected:
	void parseDate(
        const std::string& dateString,
        std::uint32_t* pYear,
        std::uint32_t* pMonth,
        std::uint32_t* pDay) const;

	void parseTime(
        const std::string& timeString,
        std::uint32_t* pHour,
        std::uint32_t* pMinutes,
        std::uint32_t* pSeconds,
        std::uint32_t* pNanoseconds,
		std::int32_t* pOffsetHours,
		std::int32_t* pOffsetMinutes) const;
};

class writingDataHandlerDateTimeBase: public writingDataHandlerString
{
public:
    writingDataHandlerDateTimeBase(const std::shared_ptr<buffer>& pBuffer, tagVR_t dataType, const size_t unitSize, const size_t maxSize);

    virtual void setSignedLong(const size_t index, const std::int32_t value);
    virtual void setUnsignedLong(const size_t index, const std::uint32_t value);
    virtual void setDouble(const size_t index, const double value);

protected:
    std::string buildDate(
        std::uint32_t year,
        std::uint32_t month,
        std::uint32_t day) const;

    std::string buildTime(
        std::uint32_t hour,
        std::uint32_t minutes,
        std::uint32_t seconds,
        std::uint32_t nanoseconds,
        std::int32_t offsetHours,
        std::int32_t offsetMinutes
        ) const;

    std::string buildTimeSimple(
        std::uint32_t hour,
        std::uint32_t minutes,
        std::uint32_t seconds,
        std::uint32_t nanoseconds
        ) const;

    std::string padLeft(const std::string& source, const char fillChar, const size_t length) const;

};

} // namespace handlers

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraDataHandlerDateTimeBase_85665C7B_8DDF_479e_8CC0_83E95CB625DC__INCLUDED_)
