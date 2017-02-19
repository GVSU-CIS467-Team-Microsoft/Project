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

/*! \file drawBitmap.cpp
    \brief Implementation of the class DrawBitmap.

*/

#include "../include/imebra/drawBitmap.h"
#include "../implementation/drawBitmapImpl.h"
#include "../include/imebra/image.h"
#include "../include/imebra/transform.h"

namespace imebra
{

DrawBitmap::DrawBitmap():
    m_pDrawBitmap(std::make_shared<implementation::drawBitmap>(std::shared_ptr<implementation::transforms::transform>()))
{
}

DrawBitmap::DrawBitmap(const Transform& transform):
    m_pDrawBitmap(std::make_shared<implementation::drawBitmap>(transform.m_pTransform))
{
}

DrawBitmap::~DrawBitmap()
{
}

size_t DrawBitmap::getBitmap(const Image& image, drawBitmapType_t drawBitmapType, std::uint32_t rowAlignBytes, char* buffer, size_t bufferSize)
{
    return m_pDrawBitmap->getBitmap(image.m_pImage, drawBitmapType, rowAlignBytes, (std::uint8_t*)buffer, bufferSize);
}

ReadWriteMemory* DrawBitmap::getBitmap(const Image& image, drawBitmapType_t drawBitmapType, std::uint32_t rowAlignBytes)
{
    return new ReadWriteMemory(m_pDrawBitmap->getBitmap(image.m_pImage, drawBitmapType, rowAlignBytes));
}

}
