# Install script for directory: /home/ron-patrick/Documents/Capstone/pConcept/library

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/libimebra1" TYPE FILE PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ FILES "/home/ron-patrick/Documents/Capstone/pConcept/artifacts/copyright")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimebra.so.1.0.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimebra.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimebra.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/ron-patrick/Documents/Capstone/pConcept/artifacts/libimebra.so.1.0.0"
    "/home/ron-patrick/Documents/Capstone/pConcept/artifacts/libimebra.so.1"
    "/home/ron-patrick/Documents/Capstone/pConcept/artifacts/libimebra.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimebra.so.1.0.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimebra.so.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libimebra.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Include files")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/imebra" TYPE FILE FILES
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/VOILUT.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/baseStreamInput.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/baseStreamOutput.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/codecFactory.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/colorTransformsFactory.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/dataSet.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/definitions.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/dicomDictionary.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/dicomDir.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/dicomDirEntry.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/drawBitmap.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/exceptions.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/fileStreamInput.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/fileStreamOutput.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/image.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/imebra.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/lut.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/memoryPool.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/memoryStreamInput.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/memoryStreamOutput.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/modalityVOILUT.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/readMemory.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/readWriteMemory.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/readingDataHandler.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/readingDataHandlerNumeric.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/streamReader.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/streamWriter.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/tag.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/tagId.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/tagsEnumeration.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/transform.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/transformHighBit.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/transformsChain.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/writingDataHandler.h"
    "/home/ron-patrick/Documents/Capstone/pConcept/library/include/imebra/writingDataHandlerNumeric.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ron-patrick/Documents/Capstone/pConcept/artifacts/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
