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

/*! \file dataHandlerStringAS.h
    \brief Declaration of the class dataHandlerStringAS.

*/

#if !defined(imebraDataHandlerStringAS_367AAE47_6FD7_4107_AB5B_25A355C5CB6E__INCLUDED_)
#define imebraDataHandlerStringAS_367AAE47_6FD7_4107_AB5B_25A355C5CB6E__INCLUDED_

#include "dataHandlerStringImpl.h"
#include "../include/imebra/definitions.h"


namespace imebra
{

namespace implementation
{

namespace handlers
{


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief Handles the Dicom data type "AS" (age string)
///
/// The handler supplies two additional functions designed
///  to set/get the age (setAge() and getAge()) and 
///  overwrite the functions getSignedLong(), 
///  getUnsignedLong(), getDouble(), setSignedLong(),
///  setUnsignedLong(), setDouble() to make them work with
///  the years.
///
/// setDouble() and getDouble() work also with fraction
///  of a year, setting the age unit correctly (days, 
///  weeks, months or years).
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class readingDataHandlerStringAS : public readingDataHandlerString
{
public:
    readingDataHandlerStringAS(const memory& parseMemory);

	/// \brief Retrieve the age value and its unit from the
	///         buffer handled by this handler.
	///
	/// @param index the zero based index of the age value to
	///               modify
    /// @param pUnit a pointer to a ageUnit_t that will be
	///               filled with the unit information related
	///               to the returned value
	/// @return the age, expressed in the unit written in the
	///               location referenced by the parameter 
	///               pUnit
	///
	///////////////////////////////////////////////////////////
    virtual std::uint32_t getAge(const size_t index, ageUnit_t* pUnit) const;

	/// \brief Retrieve the age from the handled buffer.
	///
	/// The returned value is always expressed in years.
	///
	/// @param index the zero based index of the age value to
	///               read
	/// @return the age contained in the buffer converted into
	///          years
	///
	///////////////////////////////////////////////////////////
	virtual std::int32_t getSignedLong(const size_t index) const;

	/// \brief Retrieve the age from the handled buffer.
	///
	/// The returned value is always expressed in years.
	///
	/// @param index the zero based index of the age value to
	///               read
	/// @return the age contained in the buffer converted into
	///          years
	///
	///////////////////////////////////////////////////////////
	virtual std::uint32_t getUnsignedLong(const size_t index) const;

	/// \brief Retrieve the age from the handled buffer.
	///
	/// The returned value is always expressed in years.
	/// The function can return fraction of a year if the
	///  age contained in the buffer is expressed in days,
	///  weeks or months.
	///
	/// @param index the zero based index of the age value to
	///               read
	/// @return the age contained in the buffer converted into
	///          years
	///
	///////////////////////////////////////////////////////////
	virtual double getDouble(const size_t index) const;
};


class writingDataHandlerStringAS: public writingDataHandlerString
{
public:
    writingDataHandlerStringAS(const std::shared_ptr<buffer>& pBuffer);

    /// \brief Set the value of the age string and specify
    ///         its unit.
    ///
    /// @param index the zero based index of the age value to
    ///               read
    /// @param age   the age to be written in the buffer
    /// @param unit  the units used for the parameter age
    ///
    ///////////////////////////////////////////////////////////
    virtual void setAge(const size_t index, const std::uint32_t age, const ageUnit_t unit);

    /// \brief Write the specified age into the handled buffer.
    ///
    /// @param index the zero based index of the age value to
    ///               modify
    /// @param value the age to be written, in years
    ///
    ///////////////////////////////////////////////////////////
    virtual void setSignedLong(const size_t index, const std::int32_t value);

    /// \brief Write the specified age into the handled buffer.
    ///
    /// @param index the zero based index of the age value to
    ///               modify
    /// @param value the age to be written, in years
    ///
    ///////////////////////////////////////////////////////////
    virtual void setUnsignedLong(const size_t index, const std::uint32_t value);

    /// \brief Write the specified age into the handled buffer.
    ///
    /// If a fraction of a year is specified, then the function
    ///  set the age in days, weeks or months according to
    ///  the value of the fraction.
    ///
    /// @param index the zero based index of the age value to
    ///               modify
    /// @param value the age to be written, in years
    ///
    ///////////////////////////////////////////////////////////
    virtual void setDouble(const size_t index, const double value);

    virtual void validate() const;

};

} // namespace handlers

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraDataHandlerStringAS_367AAE47_6FD7_4107_AB5B_25A355C5CB6E__INCLUDED_)
