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

/*! \file memory.cpp
    \brief Implementation of the memory manager and the memory class.

*/

#include "memoryImpl.h"
#include "exceptionImpl.h"
#include "../include/imebra/exceptions.h"
#include <string.h>

namespace imebra
{

namespace implementation
{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// memory
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Constructors
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
memory::memory():
    m_pMemoryBuffer(new stringUint8())
{
}

memory::memory(stringUint8* pBuffer):
    m_pMemoryBuffer(pBuffer)
{
}

memory::memory(size_t initialSize):
    m_pMemoryBuffer(memoryPoolGetter::getMemoryPoolGetter().getMemoryPoolLocal().getMemory(initialSize))
{
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Destructor
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
memory::~memory()
{
    memoryPoolGetter::getMemoryPoolGetter().getMemoryPoolLocal().reuseMemory(m_pMemoryBuffer.release());
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Copy the content from another memory object
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memory::copyFrom(const std::shared_ptr<const memory>& sourceMemory)
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0)
	{
		m_pMemoryBuffer.reset(new stringUint8);
	}
	m_pMemoryBuffer->assign(sourceMemory->data(), sourceMemory->size());

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Clear the managed string
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memory::clear()
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() != 0)
	{
		m_pMemoryBuffer->clear();
	}

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Resize the memory buffer
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memory::resize(size_t newSize)
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0)
	{
		m_pMemoryBuffer.reset(new stringUint8((size_t)newSize, (std::uint8_t)0));
	}
	else
	{
	    m_pMemoryBuffer->resize((size_t)newSize, (std::uint8_t)0);
	}

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Reserve memory
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memory::reserve(size_t reserveSize)
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0)
	{
        m_pMemoryBuffer.reset(new stringUint8());
	}
	m_pMemoryBuffer->reserve(reserveSize);

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Return the size of the managed string
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
size_t memory::size() const
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0)
	{
		return 0;
	}
    return m_pMemoryBuffer->size();

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Return a pointer to the data
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
std::uint8_t* memory::data()
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0 || m_pMemoryBuffer->empty())
	{
		return 0;
	}
    return &( (*(m_pMemoryBuffer.get()))[0]);

    IMEBRA_FUNCTION_END();
}

const std::uint8_t* memory::data() const
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0 || m_pMemoryBuffer->empty())
    {
        return 0;
    }
    return &( ( (*m_pMemoryBuffer.get()))[0]);

    IMEBRA_FUNCTION_END();
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Return true if the managed string is empty
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
bool memory::empty() const
{
    IMEBRA_FUNCTION_START();

    return m_pMemoryBuffer.get() == 0 || m_pMemoryBuffer->empty();

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Copy the content of a buffer in the managed string
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memory::assign(const std::uint8_t* pSource, const size_t sourceLength)
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0)
	{
		m_pMemoryBuffer.reset(new stringUint8);
	}
	m_pMemoryBuffer->assign(pSource, sourceLength);

    IMEBRA_FUNCTION_END();
}

void memory::assignRegion(const std::uint8_t* pSource, const size_t sourceLength, const size_t destinationOffset)
{
    IMEBRA_FUNCTION_START();

    if(m_pMemoryBuffer.get() == 0)
    {
        m_pMemoryBuffer.reset(new stringUint8);
    }
    if(m_pMemoryBuffer->size() < destinationOffset + sourceLength)
    {
        IMEBRA_THROW(MemorySizeError, "The memory size is too small to accept the source region");
    }
    ::memcpy(&(m_pMemoryBuffer->at(destinationOffset)), pSource, sourceLength);

    IMEBRA_FUNCTION_END();

}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
//
// memoryPool
//
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Constructor
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
memoryPool::memoryPool(size_t memoryMinSize, size_t poolMaxSize):
    m_minMemoryBlockSize(memoryMinSize), m_maxMemoryUsageSize(poolMaxSize),
    m_actualSize(0), m_firstUsedCell(0), m_firstFreeCell(0)
{
}

memoryPool::~memoryPool()
{
    while(m_actualSize != 0)
    {
        m_actualSize -= m_memorySize[m_firstUsedCell];
        delete m_memoryPointer[m_firstUsedCell++];
        if(m_firstUsedCell == m_memoryPointer.size())
        {
            m_firstUsedCell = 0;
        }
    }
}

void memoryPool::setMinMaxMemory(size_t memoryMinSize, size_t poolMaxSize)
{
    flush();
    m_minMemoryBlockSize = memoryMinSize;
    m_maxMemoryUsageSize = poolMaxSize;

}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Save a memory object to reuse it
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memoryPool::reuseMemory(stringUint8* pString)
{
    IMEBRA_FUNCTION_START();

    if(pString == 0)
    {
        return;
    }
    std::unique_ptr<stringUint8> pBuffer(pString);

	// Check for the memory size. Don't reuse it if the memory
	//  doesn't match the requested parameters
	///////////////////////////////////////////////////////////
    size_t memorySize = pBuffer->size();
    if(memorySize == 0 || memorySize < m_minMemoryBlockSize || memorySize > m_maxMemoryUsageSize)
	{
        return;
	}

	// Store the memory object in the pool
	///////////////////////////////////////////////////////////
	m_memorySize[m_firstFreeCell] = memorySize;
    m_memoryPointer[m_firstFreeCell] = pBuffer.release();
	m_actualSize += memorySize;
    if(++m_firstFreeCell >= m_memorySize.size())
	{
		m_firstFreeCell = 0;
	}

	// Remove old unused memory objects
	///////////////////////////////////////////////////////////
	if(m_firstFreeCell == m_firstUsedCell)
	{
		m_actualSize -= m_memorySize[m_firstUsedCell];
		delete m_memoryPointer[m_firstUsedCell];
        if(++m_firstUsedCell >= m_memorySize.size())
		{
			m_firstUsedCell = 0;
		}
	}

	// Remove old unused memory objects if the total unused
	//  memory is bigger than the specified parameters
	///////////////////////////////////////////////////////////
    while(m_actualSize != 0 && m_actualSize > m_maxMemoryUsageSize)
	{
		m_actualSize -= m_memorySize[m_firstUsedCell];
		delete m_memoryPointer[m_firstUsedCell];
		if(++m_firstUsedCell >= IMEBRA_MEMORY_POOL_SLOTS)
		{
			m_firstUsedCell = 0;
		}
	}

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Return the size of the unused memory
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
size_t memoryPool::getUnusedMemorySize()
{
    return m_actualSize;
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Discard the currently unused memory
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
bool memoryPool::flush()
{
    IMEBRA_FUNCTION_START();

    bool bCleared(m_firstUsedCell != m_firstFreeCell);
	while(m_firstUsedCell != m_firstFreeCell)
	{
		delete m_memoryPointer[m_firstUsedCell];

		m_actualSize -= m_memorySize[m_firstUsedCell];
        if(++m_firstUsedCell >= m_memorySize.size())
		{
			m_firstUsedCell = 0;
		}
	}
    return bCleared;

    IMEBRA_FUNCTION_END();
}


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Look for a memory object to reuse or allocate a new one
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
stringUint8* memoryPool::getMemory(size_t requestedSize)
{
    IMEBRA_FUNCTION_START();

    if(requestedSize < m_minMemoryBlockSize || requestedSize > m_maxMemoryUsageSize)
    {
        return new stringUint8(requestedSize, 0);
    }

	// Look for an object to reuse
	///////////////////////////////////////////////////////////
    for(size_t findCell = m_firstUsedCell; findCell != m_firstFreeCell;)
	{
		if(m_memorySize[findCell] != requestedSize)
		{
            if(++findCell >= m_memorySize.size())
			{
				findCell = 0;
			}
			continue;
		}

		// Memory found
		///////////////////////////////////////////////////////////
        std::unique_ptr<stringUint8> pMemory(m_memoryPointer[findCell]);
		m_actualSize -= m_memorySize[findCell];
		if(findCell == m_firstUsedCell)
		{
			if(++m_firstUsedCell >= IMEBRA_MEMORY_POOL_SLOTS)
			{
				m_firstUsedCell = 0;
			}
            return pMemory.release();
		}

        size_t lastUsedCell = m_firstFreeCell == 0 ? (IMEBRA_MEMORY_POOL_SLOTS - 1) : (m_firstFreeCell - 1);
		if(findCell == lastUsedCell)
		{
			m_firstFreeCell = lastUsedCell;
            return pMemory.release();
		}

		m_memorySize[findCell] = m_memorySize[m_firstUsedCell];
		m_memoryPointer[findCell] = m_memoryPointer[m_firstUsedCell];
		if(++m_firstUsedCell >= IMEBRA_MEMORY_POOL_SLOTS)
		{
			m_firstUsedCell = 0;
		}
        return pMemory.release();
	}

    return new stringUint8(requestedSize, 0);

    IMEBRA_FUNCTION_END();
}


memoryPoolGetter::memoryPoolGetter()
{
    IMEBRA_FUNCTION_START();

    m_oldNewHandler = std::set_new_handler(memoryPoolGetter::newHandler);
#ifdef __APPLE__
    ::pthread_key_create(&m_key, &memoryPoolGetter::deleteMemoryPool);
#endif

    IMEBRA_FUNCTION_END();
}

memoryPoolGetter::~memoryPoolGetter()
{
    std::set_new_handler(m_oldNewHandler);
#ifdef __APPLE__
    ::pthread_key_delete(m_key);
#endif
}

memoryPoolGetter& memoryPoolGetter::getMemoryPoolGetter()
{
    static memoryPoolGetter getter;
    return getter;
}

#ifndef __APPLE__
thread_local std::unique_ptr<memoryPool> memoryPoolGetter::m_pool = std::unique_ptr<memoryPool>();
#endif

memoryPool& memoryPoolGetter::getMemoryPoolLocal()
{
    IMEBRA_FUNCTION_START();

#ifdef __APPLE__
    memoryPool* pPool = (memoryPool*)pthread_getspecific(m_key);
    if(pPool == 0)
    {
        pPool = new memoryPool(IMEBRA_MEMORY_POOL_MIN_SIZE, IMEBRA_MEMORY_POOL_MAX_SIZE);
        pthread_setspecific(m_key, pPool);
    }
    return *pPool;
#else
    if(m_pool.get() == 0)
    {
        m_pool.reset(new memoryPool(IMEBRA_MEMORY_POOL_MIN_SIZE, IMEBRA_MEMORY_POOL_MAX_SIZE));
    }
    return *(m_pool.get());
#endif

    IMEBRA_FUNCTION_END();
}

#ifdef __APPLE__
void memoryPoolGetter::deleteMemoryPool(void* pMemoryPool)
{
    delete (memoryPool*)pMemoryPool;
}
#endif

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
//
//
// Set by std::set_new_handler() as new handler.
//
//
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
void memoryPoolGetter::newHandler()
{
    if(!memoryPoolGetter::getMemoryPoolGetter().getMemoryPoolLocal().flush())
    {
        throw ImebraBadAlloc();
    }
}

} // namespace implementation

} // namespace imebra
