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

/*! \file dicomDir.cpp
    \brief Implementation of the classes that parse/create a DICOMDIR
        structure (DicomDir and DicomDirEntry).

*/

#include "../include/imebra/dicomDirEntry.h"
#include "../include/imebra/dataSet.h"
#include "../implementation/dicomDirImpl.h"

namespace imebra
{

DicomDirEntry::DicomDirEntry(std::shared_ptr<imebra::implementation::directoryRecord> pDirectoryRecord): m_pDirectoryRecord(pDirectoryRecord)
{
}

DicomDirEntry::~DicomDirEntry()
{
}

DataSet* DicomDirEntry::getEntryDataSet()
{
    return new DataSet(m_pDirectoryRecord->getRecordDataSet());
}

DicomDirEntry* DicomDirEntry::getNextEntry()
{
    std::shared_ptr<implementation::directoryRecord> pNextEntry(m_pDirectoryRecord->getNextRecord());
    if(pNextEntry == 0)
    {
        return 0;
    }
    return new DicomDirEntry(pNextEntry);
}

DicomDirEntry* DicomDirEntry::getFirstChildEntry()
{
    std::shared_ptr<implementation::directoryRecord> pChildEntry(m_pDirectoryRecord->getFirstChildRecord());
    if(pChildEntry == 0)
    {
        return 0;
    }
    return new DicomDirEntry(pChildEntry);
}
	
void DicomDirEntry::setNextEntry(const DicomDirEntry& nextEntry)
{
    m_pDirectoryRecord->setNextRecord(nextEntry.m_pDirectoryRecord);
}

void DicomDirEntry::setFirstChildEntry(const DicomDirEntry& firstChildEntry)
{
    m_pDirectoryRecord->setFirstChildRecord(firstChildEntry.m_pDirectoryRecord);
}

fileParts_t DicomDirEntry::getFileParts() const
{
    return m_pDirectoryRecord->getFileParts();
}

void DicomDirEntry::setFileParts(const fileParts_t& fileParts)
{
    m_pDirectoryRecord->setFileParts(fileParts);
}

directoryRecordType_t DicomDirEntry::getType() const
{
    return m_pDirectoryRecord->getType();
}

std::string DicomDirEntry::getTypeString() const
{
    return m_pDirectoryRecord->getTypeString();
}


}
