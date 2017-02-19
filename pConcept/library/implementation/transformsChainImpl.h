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

/*! \file transformsChain.h
    \brief Declaration of the class transformsChain.

*/

#if !defined(imebraTransformsChain_5DB89BFD_F105_45e7_B9D9_3756AC93C821__INCLUDED_)
#define imebraTransformsChain_5DB89BFD_F105_45e7_B9D9_3756AC93C821__INCLUDED_

#include <vector>
#include "transformImpl.h"

namespace imebra
{

namespace implementation
{

namespace transforms
{

/// \addtogroup group_transforms
///
/// @{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief Executes a sequence of transforms.
///
/// Before calling runTransform() specify the sequence
///  by calling addTransform().
/// Each specified transforms take the output of the 
///  previous transform as input.
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class transformsChain: public transform
{
public:
	transformsChain();

	/// \brief Add a transform to the transforms chain.
	///
	/// The added transform will take the output of the 
	///  previously added transform as an input image and will
	///  supply its output to the next added transform or as
	///  an output of the transformsChain if it is the
	///  last added transform.
	///
	/// @param pTransform the transform to be added to
	///                    transformsChain
	///
	///////////////////////////////////////////////////////////
	void addTransform(std::shared_ptr<transform> pTransform);

    virtual void runTransformHandlers(
            std::shared_ptr<handlers::readingDataHandlerNumericBase> inputHandler, bitDepth_t inputDepth, std::uint32_t inputHandlerWidth, const std::string& inputHandlerColorSpace,
            std::shared_ptr<palette> inputPalette,
            std::uint32_t inputHighBit,
            std::uint32_t inputTopLeftX, std::uint32_t inputTopLeftY, std::uint32_t inputWidth, std::uint32_t inputHeight,
            std::shared_ptr<handlers::writingDataHandlerNumericBase> outputHandler, bitDepth_t outputDepth, std::uint32_t outputHandlerWidth, const std::string& outputHandlerColorSpace,
            std::shared_ptr<palette> outputPalette,
            std::uint32_t outputHighBit,
            std::uint32_t outputTopLeftX, std::uint32_t outputTopLeftY) const;

	/// \brief Returns true if the transform doesn't do
	///         anything.
	///
	/// @return false if the transform does something, or true
	///          if the transform doesn't do anything (e.g. an
	///          empty transformsChain object).
	///
	///////////////////////////////////////////////////////////
    virtual bool isEmpty() const;

    virtual std::shared_ptr<image> allocateOutputImage(
            bitDepth_t inputDepth,
            const std::string& inputColorSpace,
            std::uint32_t inputHighBit,
            std::shared_ptr<palette> inputPalette,
            std::uint32_t outputWidth, std::uint32_t outputHeight) const;

protected:
    typedef std::vector<std::shared_ptr<transform> > tTransformsList;
	tTransformsList m_transformsList;

};

/// @}

} // namespace transforms

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraTransformsChain_5DB89BFD_F105_45e7_B9D9_3756AC93C821__INCLUDED_)
