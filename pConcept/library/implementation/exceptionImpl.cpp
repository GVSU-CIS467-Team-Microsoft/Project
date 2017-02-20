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

/*! \file exception.cpp
    \brief Implementation of the exception classes.

*/

#include "exceptionImpl.h"
#include "charsetConversionImpl.h"

namespace imebra
{

namespace implementation
{


///////////////////////////////////////////////////////////
// Return the message info for the current thread
///////////////////////////////////////////////////////////
std::string exceptionsManager::getMessage()
{
	tExceptionInfoList infoList;
	exceptionsManager::getExceptionInfo(&infoList);

    std::string message;
	for(tExceptionInfoList::iterator scanInfo = infoList.begin(); 
		scanInfo != infoList.end(); 
		++scanInfo)
	{
		message += scanInfo->getMessage();
        message += "\n\n";
	}

	return message;
}


///////////////////////////////////////////////////////////
// Return the info objects for the specified thread
///////////////////////////////////////////////////////////
void exceptionsManager::getExceptionInfo(tExceptionInfoList* pList)
{
    for(tExceptionInfoList::iterator scanInformation = m_information.begin();
        scanInformation != m_information.end();
		++scanInformation)
	{
		pList->push_back(*scanInformation);
	}
    m_information.clear();
}


///////////////////////////////////////////////////////////
// Add an info object to the current thread
///////////////////////////////////////////////////////////
void exceptionsManager::startExceptionInfo(const exceptionInfo& info)
{
    m_information.clear();
    m_information.push_back(info);
}


///////////////////////////////////////////////////////////
// Add an info object to the current thread
///////////////////////////////////////////////////////////
void exceptionsManager::addExceptionInfo(const exceptionInfo& info)
{
    m_information.push_back(info);
}


///////////////////////////////////////////////////////////
// Construct the exceptionInfo object
///////////////////////////////////////////////////////////
exceptionInfo::exceptionInfo(const std::string& functionName, const std::string& fileName, const long lineNumber, const std::string& exceptionType, const std::string& exceptionMessage):
	m_functionName(functionName), 
	m_fileName(fileName),
	m_lineNumber(lineNumber),
	m_exceptionType(exceptionType),
	m_exceptionMessage(exceptionMessage)
{}

///////////////////////////////////////////////////////////
// Copy constructor
///////////////////////////////////////////////////////////
exceptionInfo::exceptionInfo(const exceptionInfo& right):
			m_functionName(right.m_functionName), 
			m_fileName(right.m_fileName),
			m_lineNumber(right.m_lineNumber),
			m_exceptionType(right.m_exceptionType),
			m_exceptionMessage(right.m_exceptionMessage)
{}

///////////////////////////////////////////////////////////
// Return the exceptionInfo content in a string
///////////////////////////////////////////////////////////
std::string exceptionInfo::getMessage()
{
    std::ostringstream message;
	message << "[" << m_functionName << "]" << "\n";
    message << " file: " << m_fileName << "  line: " << m_lineNumber << "\n";
    message << " exception type: " << m_exceptionType << "\n";
    message << " exception message: " << m_exceptionMessage << "\n";
	return message.str();
}



exceptionsManagerGetter::exceptionsManagerGetter()
{
    IMEBRA_FUNCTION_START();

#ifdef __APPLE__
    ::pthread_key_create(&m_key, &exceptionsManagerGetter::deleteExceptionsManager);
#endif

    IMEBRA_FUNCTION_END();
}

exceptionsManagerGetter::~exceptionsManagerGetter()
{
#ifdef __APPLE__
    ::pthread_key_delete(m_key);
#endif
}

exceptionsManagerGetter& exceptionsManagerGetter::getExceptionsManagerGetter()
{
    static exceptionsManagerGetter getter;
    return getter;
}

#ifndef __APPLE__
thread_local std::unique_ptr<exceptionsManager> exceptionsManagerGetter::m_pManager = std::unique_ptr<exceptionsManager>();
#endif

exceptionsManager& exceptionsManagerGetter::getExceptionsManager()
{
    IMEBRA_FUNCTION_START();

#ifdef __APPLE__
    exceptionsManager* pManager = (exceptionsManager*)pthread_getspecific(m_key);
    if(pManager == 0)
    {
        pManager = new exceptionsManager();
        pthread_setspecific(m_key, pManager);
    }
    return *pManager;
#else
    if(m_pManager.get() == 0)
    {
        m_pManager.reset(new exceptionsManager());
    }
    return *(m_pManager.get());
#endif

    IMEBRA_FUNCTION_END();
}

#ifdef __APPLE__
void exceptionsManagerGetter::deleteExceptionsManager(void* pManager)
{
    delete (exceptionsManager*)pManager;
}
#endif

} // namespace implementation

} // namespace imebra
