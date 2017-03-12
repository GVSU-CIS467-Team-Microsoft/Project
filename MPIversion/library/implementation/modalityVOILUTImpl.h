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

/*! \file modalityVOILUT.h
    \brief Declaration of the class modalityVOILUT.

*/

#if !defined(imebraModalityVOILUT_8347C70F_1FC8_4df8_A887_8DE9E968B2CF__INCLUDED_)
#define imebraModalityVOILUT_8347C70F_1FC8_4df8_A887_8DE9E968B2CF__INCLUDED_

#include "transformImpl.h"
#include "imageImpl.h"
#include "dataSetImpl.h"
#include "LUTImpl.h"
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
///////////////////////////////////////////////////////////
/// \brief This class transforms the pixel values of the
///         image retrieved from the dataset into values
///         that are meaningful to th application.
///
/// For instance, the original pixel values could store
///  a device specific value that has a meaning only when
///  used by the device that generated it: this transform
///  uses the modality VOI/LUT defined in the dataset to
///  convert the original values into optical density
///  or other known measure units.
///
/// If the dataset doesn't define any modality VOI/LUT
///  transformation, then the input image is simply copied
///  into the output image.
///
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class modalityVOILUT: public transform
{
public:
	/// \brief Constructor.
	///
	/// @param pDataSet the dataSet from which the input
	///         images come from
	///
	///////////////////////////////////////////////////////////
    modalityVOILUT(std::shared_ptr<const dataSet> pDataSet);

	DEFINE_RUN_TEMPLATE_TRANSFORM;

	template <class inputType, class outputType>
			void templateTransform(
                    const inputType* inputHandlerData,
                    outputType* outputHandlerData,
                    bitDepth_t /* inputDepth */, std::uint32_t inputHandlerWidth, const std::string& inputHandlerColorSpace,
                    std::shared_ptr<palette> /* inputPalette */,
                    std::uint32_t /* inputHighBit */,
                    std::uint32_t inputTopLeftX, std::uint32_t inputTopLeftY, std::uint32_t inputWidth, std::uint32_t inputHeight,
                    bitDepth_t /* outputDepth */, std::uint32_t outputHandlerWidth, const std::string& outputHandlerColorSpace,
                    std::shared_ptr<palette> /* outputPalette */,
                    std::uint32_t /* outputHighBit */,
                    std::uint32_t outputTopLeftX, std::uint32_t outputTopLeftY) const
	{
        IMEBRA_FUNCTION_START();

		if(!colorTransforms::colorTransformsFactory::isMonochrome(inputHandlerColorSpace) || !colorTransforms::colorTransformsFactory::isMonochrome(outputHandlerColorSpace))
		{
            IMEBRA_THROW(ModalityVOILUTError, "modalityVOILUT can process only monochromatic images");
		}
        const inputType* pInputMemory(inputHandlerData);
		outputType* pOutputMemory(outputHandlerData);

		pInputMemory += inputTopLeftY * inputHandlerWidth + inputTopLeftX;
		pOutputMemory += outputTopLeftY * outputHandlerWidth + outputTopLeftX;

		//
		// Modality LUT found
		//
		///////////////////////////////////////////////////////////
        if(m_voiLut != 0 && m_voiLut->getSize() != 0)
		{
			for(; inputHeight != 0; --inputHeight)
			{
                for(std::uint32_t scanPixels(inputWidth); scanPixels != 0; --scanPixels)
				{
                    *(pOutputMemory++) = (outputType) ( m_voiLut->getMappedValue((std::int32_t)*(pInputMemory++)));
				}
				pInputMemory += (inputHandlerWidth - inputWidth);
				pOutputMemory += (outputHandlerWidth - inputWidth);
			}
			return;
		}

		//
		// Modality LUT not found
		//
		///////////////////////////////////////////////////////////

		// Retrieve the intercept/scale pair
		///////////////////////////////////////////////////////////
		for(; inputHeight != 0; --inputHeight)
		{
            for(std::uint32_t scanPixels(inputWidth); scanPixels != 0; --scanPixels)
			{
                *(pOutputMemory++) = (outputType)((double)(*(pInputMemory++)) * m_rescaleSlope + m_rescaleIntercept);
			}
			pInputMemory += (inputHandlerWidth - inputWidth);
			pOutputMemory += (outputHandlerWidth - inputWidth);
		}
		IMEBRA_FUNCTION_END();
	}

    virtual bool isEmpty() const;

    virtual std::shared_ptr<image> allocateOutputImage(
            bitDepth_t inputDepth,
            const std::string& inputColorSpace,
            std::uint32_t inputHighBit,
            std::shared_ptr<palette> inputPalette,
            std::uint32_t outputWidth, std::uint32_t outputHeight) const;

private:
    std::shared_ptr<const dataSet> m_pDataSet;
    std::shared_ptr<const lut> m_voiLut;
    double m_rescaleIntercept;
    double m_rescaleSlope;
	bool m_bEmpty;
};


/// @}

} // namespace transforms

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraModalityVOILUT_8347C70F_1FC8_4df8_A887_8DE9E968B2CF__INCLUDED_)
