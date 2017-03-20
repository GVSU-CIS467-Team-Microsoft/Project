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

/*! \file baseStream.h
    \brief Declaration of the the base class for the streams (memory, file, ...)
            used by the imebra library.

*/

#if !defined(imebraBaseStream_3146DA5A_5276_4804_B9AB_A3D54C6B123A__INCLUDED_)
#define imebraBaseStream_3146DA5A_5276_4804_B9AB_A3D54C6B123A__INCLUDED_

#include <memory>
#include "exceptionImpl.h"
#include <vector>
#include <map>
#include <stdexcept>


namespace imebra
{

namespace implementation
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief This class represents an input stream.
///
/// Specialized classes derived from this class can
///  read from files stored on the computer's disks, on the
///  network or in memory.
///
/// The application can read or write into the stream
///  by using the streamReader or the streamWriter.
///
/// While this class can be used across several threads,
///  the streamReader and the streamWriter can be used in
///  one thread only. This is not a big deal, since one
///  stream can be connected to several streamReaders and
///  streamWriters.
///
/// The library supplies two specialized streams derived
///  from this class:
/// - imebra::stream (used to read or write into physical
///    files)
/// - imebra::memoryStream (used to read or write into
///    imebra::memory objects)
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class baseStreamInput
{

public:
	/// \brief Read raw data from the stream.
	///
	/// The function is multithreading-safe and is called by
	///  the streamReader class when its buffer has to be
	///  refilled.
	///
	/// @param startPosition  the position in the file from
	///                        which the data has to be read
	/// @param pBuffer        a pointer to the memory where the
	///                        read data has to be placed
	/// @param bufferLength   the number of bytes to read from
	///                        the file
	/// @return the number of bytes read from the file. When
	///          it is 0 then the end of the file has been
	///          reached
	///
	///////////////////////////////////////////////////////////
    virtual size_t read(size_t startPosition, std::uint8_t* pBuffer, size_t bufferLength) = 0;
};


class baseStreamOutput
{

public:
    /// \brief Writes raw data into the stream.
    ///
    /// The function is multithreading-safe and is called by
    ///  the streamWriter class when its buffer has to be
    ///  flushed.
    ///
    /// @param startPosition  the position in the file where
    ///                        the data has to be written
    /// @param pBuffer        pointer to the data that has to
    ///                        be written
    /// @param bufferLength   number of bytes in the data
    ///                        buffer that has to be written
    ///
    ///////////////////////////////////////////////////////////
    virtual void write(size_t startPosition, const std::uint8_t* pBuffer, size_t bufferLength) = 0;

};

} // namespace implementation

} // namespace imebra


#endif // !defined(imebraBaseStream_3146DA5A_5276_4804_B9AB_A3D54C6B123A__INCLUDED_)
