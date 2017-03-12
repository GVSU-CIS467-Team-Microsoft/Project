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

/*! \file transformHighBit.h
    \brief Declaration of the class transformHighBit.

*/

#if !defined(imebraTransformHighBit_8347C70F_1FC8_4df8_A887_8DE9E968B2CF__INCLUDED_)
#define imebraTransformHighBit_8347C70F_1FC8_4df8_A887_8DE9E968B2CF__INCLUDED_

#include "transformImpl.h"
#include "colorTransformsFactoryImpl.h"
#include "../include/imebra/exceptions.h"

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
/// \brief Shift the image's content to adapt it to the
///         specified high bit.
///
/// In order to use this transform, both the input and
///  the output image must be defined by calling
///  declareInputImage() and declareOutputImage().
///
/// The input image is copied into the output image but
///  all the values are shifted to be fitted in the
///  output image's high bit settings.
///
///////////////////////////////////////////////////////////
class transformHighBit: public transform
{
public:
    virtual std::shared_ptr<image> allocateOutputImage(
            bitDepth_t inputDepth,
            const std::string& inputColorSpace,
            std::uint32_t inputHighBit,
            std::shared_ptr<palette> inputPalette,
            std::uint32_t outputWidth, std::uint32_t outputHeight) const;

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

            if(colorTransforms::colorTransformsFactory::normalizeColorSpace(inputHandlerColorSpace) !=
               colorTransforms::colorTransformsFactory::normalizeColorSpace(outputHandlerColorSpace))
            {
                IMEBRA_THROW(TransformHighBitDifferentColorSpacesError, "The input and output image must have the same color space");
            }

            std::uint32_t numChannels(colorTransforms::colorTransformsFactory::getNumberOfChannels(inputHandlerColorSpace));

            const inputType* pInputMemory(inputHandlerData);
            outputType* pOutputMemory(outputHandlerData);

            pInputMemory += (inputTopLeftY * inputHandlerWidth + inputTopLeftX) * numChannels;
            pOutputMemory += (outputTopLeftY * outputHandlerWidth + outputTopLeftX) * numChannels;

            std::int64_t inputHandlerMinValue = getMinValue<inputType>(inputHighBit);
            std::int64_t outputHandlerMinValue = getMinValue<outputType>(outputHighBit);

            if(inputHighBit > outputHighBit)
            {
                std::uint32_t rightShift = inputHighBit - outputHighBit;
                for(; inputHeight != 0; --inputHeight)
                {
                    for(std::uint32_t scanPixels(inputWidth * numChannels); scanPixels != 0; --scanPixels)
                    {
                        *pOutputMemory++ = (outputType)((((std::int32_t)*(pInputMemory++) - inputHandlerMinValue) >> rightShift) + outputHandlerMinValue);
                    }
                    pInputMemory += (inputHandlerWidth - inputWidth) * numChannels;
                    pOutputMemory += (outputHandlerWidth - inputWidth) * numChannels;
                }
            }
            else
            {
                std::uint32_t leftShift = outputHighBit - inputHighBit;
                for(; inputHeight != 0; --inputHeight)
                {
                    for(std::uint32_t scanPixels(inputWidth * numChannels); scanPixels != 0; --scanPixels)
                    {
                        *pOutputMemory++ = (outputType)((((std::int32_t)*(pInputMemory++) - inputHandlerMinValue) << leftShift) + outputHandlerMinValue);
                    }
                    pInputMemory += (inputHandlerWidth - inputWidth) * numChannels;
                    pOutputMemory += (outputHandlerWidth - inputWidth) * numChannels;
                }
            }

            IMEBRA_FUNCTION_END();
        }

};

/// @}

} // namespace transforms

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraTransformHighBit_8347C70F_1FC8_4df8_A887_8DE9E968B2CF__INCLUDED_)
