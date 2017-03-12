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

/*! \file colorTransformsFactory.cpp
    \brief Implementation of the class ColorTransformsFactory.
*/

#include "../include/imebra/colorTransformsFactory.h"
#include "../implementation/colorTransformsFactoryImpl.h"
#include "../include/imebra/exceptions.h"
namespace imebra
{

std::string ColorTransformsFactory::normalizeColorSpace(const std::string& colorSpace)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::normalizeColorSpace(colorSpace);
}

bool ColorTransformsFactory::isMonochrome(const std::string& colorSpace)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::isMonochrome(colorSpace);
}

bool ColorTransformsFactory::isSubsampledX(const std::string& colorSpace)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::isSubsampledX(colorSpace);
}

bool ColorTransformsFactory::isSubsampledY(const std::string& colorSpace)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::isSubsampledY(colorSpace);
}

bool ColorTransformsFactory::canSubsample(const std::string& colorSpace)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::canSubsample(colorSpace);
}

std::string ColorTransformsFactory::makeSubsampled(const std::string& colorSpace, bool bSubsampleX, bool bSubsampleY)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::makeSubsampled(colorSpace, bSubsampleX, bSubsampleY);
}

std::uint32_t ColorTransformsFactory::getNumberOfChannels(const std::string& colorSpace)
{
    return imebra::implementation::transforms::colorTransforms::colorTransformsFactory::getNumberOfChannels(colorSpace);
}

Transform* ColorTransformsFactory::getTransform(const std::string& startColorSpace, const std::string& endColorSpace)
{
    std::shared_ptr<imebra::implementation::transforms::colorTransforms::colorTransformsFactory> factory(imebra::implementation::transforms::colorTransforms::colorTransformsFactory::getColorTransformsFactory());
    Transform* transform = new Transform(factory->getTransform(startColorSpace, endColorSpace));
    if(transform->m_pTransform == 0)
    {
        IMEBRA_THROW(ColorTransformsFactoryNoTransformError, "There is no color transform that can convert between the specified color spaces " << startColorSpace << " and " << endColorSpace);
    }
    return transform;
}

}
