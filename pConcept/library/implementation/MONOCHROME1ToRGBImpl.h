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

/*! \file MONOCHROME1ToRGB.h
    \brief Declaration of the class MONOCHROME1ToRGB.

*/

#if !defined(imebraMONOCHROME1ToRGB_E27C63E7_A907_4899_9BD3_8026AD7D110C__INCLUDED_)
#define imebraMONOCHROME1ToRGB_E27C63E7_A907_4899_9BD3_8026AD7D110C__INCLUDED_

#include "colorTransformImpl.h"


namespace imebra
{

namespace implementation
{

namespace transforms
{

namespace colorTransforms
{

/// \addtogroup group_transforms
///
/// @{

///////////////////////////////////////////////////////////
/// \brief Transforms an image from the colorspace 
///         MONOCHROME1 into the color space RGB.
///
/// The input image has to have the colorspace MONOCHROME1,
///  while the output image is created by the transform
///  and will have the colorspace RGB.
///
///////////////////////////////////////////////////////////
class MONOCHROME1ToRGB: public colorTransform
{
public:
    virtual std::string getInitialColorSpace() const;
    virtual std::string getFinalColorSpace() const;

    DEFINE_RUN_TEMPLATE_TRANSFORM;

    template <class inputType, class outputType>
    void templateTransform(
            const inputType* inputHandlerData,
            outputType* outputHandlerData,
            bitDepth_t /* inputDepth */, std::uint32_t inputHandlerWidth, const std::string& inputHandlerColorSpace,
            std::shared_ptr<palette> /* inputPalette */,
            std::uint32_t inputHighBit,
            std::uint32_t inputTopLeftX, std::uint32_t inputTopLeftY, std::uint32_t inputWidth, std::uint32_t inputHeight,
            bitDepth_t /* outputDepth */, std::uint32_t outputHandlerWidth, const std::string& outputHandlerColorSpace,
            std::shared_ptr<palette> /* outputPalette */,
            std::uint32_t outputHighBit,
            std::uint32_t outputTopLeftX, std::uint32_t outputTopLeftY) const

    {
        IMEBRA_FUNCTION_START();

        checkColorSpaces(inputHandlerColorSpace, outputHandlerColorSpace);
        checkHighBit(inputHighBit, outputHighBit);

        const inputType* pInputMemory(inputHandlerData);
        outputType* pOutputMemory(outputHandlerData);

        pInputMemory += inputTopLeftY * inputHandlerWidth + inputTopLeftX;
        pOutputMemory += (outputTopLeftY * outputHandlerWidth + outputTopLeftX) * 3;

        std::int64_t inputHandlerMinValue = getMinValue<inputType>(inputHighBit);
        std::int64_t outputHandlerMinValue = getMinValue<outputType>(outputHighBit);
        std::int64_t inputHandlerNumValuesMinusOne = ((std::int64_t)1 << (inputHighBit + 1)) - 1;

        outputType outputValue;

        for(; inputHeight != 0; --inputHeight)
        {
            for(std::uint32_t scanPixels(inputWidth); scanPixels != 0; --scanPixels)
            {
                outputValue = (outputType)(outputHandlerMinValue + inputHandlerNumValuesMinusOne - (std::int64_t)*pInputMemory++ + inputHandlerMinValue);
                *pOutputMemory = outputValue;
                *++pOutputMemory = outputValue;
                *++pOutputMemory = outputValue;
                ++pOutputMemory;
            }
            pInputMemory += inputHandlerWidth - inputWidth;
            pOutputMemory += (outputHandlerWidth - inputWidth) * 3;
        }

        IMEBRA_FUNCTION_END();
    }

};

/// @}

} // namespace colorTransforms

} // namespace transforms

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraMONOCHROME1ToRGB_E27C63E7_A907_4899_9BD3_8026AD7D110C__INCLUDED_)
