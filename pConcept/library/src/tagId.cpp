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

#include "../include/imebra/dataSet.h"
#include "../implementation/dataSetImpl.h"
#include "../implementation/dataHandlerNumericImpl.h"
#include "../implementation/charsetConversionBaseImpl.h"
#include <typeinfo>
#include <memory>

namespace imebra
{

TagId::TagId(): m_groupId(0), m_groupOrder(0), m_tagId(0)
{
}

TagId::TagId(std::uint16_t groupId, std::uint16_t tagId):
    m_groupId(groupId), m_groupOrder(0), m_tagId(tagId)
{
}

TagId::TagId(std::uint16_t groupId, std::uint32_t groupOrder, std::uint16_t tagId):
    m_groupId(groupId), m_groupOrder(groupOrder), m_tagId(tagId)
{
}

TagId::TagId(tagId_t id):
    m_groupId((std::uint16_t)((std::uint32_t)id / (std::uint32_t)65536)),
    m_groupOrder(0),
    m_tagId((std::uint16_t)((std::uint32_t)id & (std::uint32_t)0x0000ffff))
{
}

TagId::TagId(tagId_t id, std::uint32_t groupOrder):
    m_groupId((std::uint16_t)((std::uint32_t)id / (std::uint32_t)65536)),
    m_groupOrder(groupOrder),
    m_tagId((std::uint16_t)((std::uint32_t)id & (std::uint32_t)0x0000ffff))
{
}

TagId::~TagId()
{
}

std::uint16_t TagId::getGroupId() const
{
    return m_groupId;
}

std::uint32_t TagId::getGroupOrder() const
{
    return m_groupOrder;
}

std::uint16_t TagId::getTagId() const
{
    return m_tagId;
}

}
