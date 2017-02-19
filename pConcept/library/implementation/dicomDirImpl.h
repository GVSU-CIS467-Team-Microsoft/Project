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

/*! \file dicomDir.h
    \brief Declaration of the classes that parse/create a DICOMDIR
	        structure (dicomDir and directoryRecord).

*/

#if !defined(imebraDicomDir_93F684BF_0024_4bf3_89BA_D98E82A1F44C__INCLUDED_)
#define imebraDicomDir_93F684BF_0024_4bf3_89BA_D98E82A1F44C__INCLUDED_

#include <memory>
#include <string>
#include <list>
#include "../include/imebra/definitions.h"

namespace imebra
{

namespace implementation
{


class dataSet; // Used in this class
class dicomDir;

/// \addtogroup group_dicomdir Dicomdir
///
/// @{

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief directoryRecord represents a single record in
///         a DICOMDIR structure (see dicomDir).
///
/// A new directoryRecord object can be constructed
///  by calling dicomDir::getNewRecord().
///
/// 
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class directoryRecord
{
    friend class dicomDir;

public:
    /// \brief Constructor. A directoryRecord object must be
    ///         connected to a dataSet object, which contains
    ///         the record's information.
    ///
    /// The dataSet object is an item of the sequence tag
    ///  0004,1220
    ///
    /// When a directoryRecord is connected to an empty dataSet
    ///  then the other constructor must be used, which allows
    ///  to define the record type.
    ///
    /// @param pDataSet  the dataSet that must be connected
    ///                   to the directoryRecord
    ///
    ///////////////////////////////////////////////////////////
    directoryRecord(std::shared_ptr<dataSet> pDataSet);

	/// \brief Returns the dataSet that contains the
	///         record's information.
	///
	/// The record's dataSet is embedded in a sequence in the
	///  DICOMDIR dataset.
	///
	/// @return the dataSet that contains the record's 
	///          information
	///
	///////////////////////////////////////////////////////////
    std::shared_ptr<dataSet> getRecordDataSet() const;

	/// \brief Returns the next sibling record.
	///
	/// @return the next sibling record, or 0 if this is the
	///          last record
	///
	///////////////////////////////////////////////////////////
    std::shared_ptr<directoryRecord> getNextRecord() const;

	/// \brief Returns the first child record.
	///
	/// @return the first child record, or 0 if the record 
	///          doesn't have any child
	///
	///////////////////////////////////////////////////////////
    std::shared_ptr<directoryRecord> getFirstChildRecord() const;
	
	/// \brief Sets the next sibling record.
	///
	/// The new sibling record takes the place of the old
	///  next sibling record, if it was already set.
	///
	/// @param pNextRecord    the next sibling record
	///
	///////////////////////////////////////////////////////////
	void setNextRecord(std::shared_ptr<directoryRecord> pNextRecord);

	/// \brief Set the first child record.
	///
	/// The new child record takes the place of the old child
	///  record, if it was already set.
	///
	/// @param pFirstChildRecord the first child record
	///
	///////////////////////////////////////////////////////////
	void setFirstChildRecord(std::shared_ptr<directoryRecord> pFirstChildRecord);

	/// \brief Get the full path to the  file referenced by
	///         the record.
	///
	/// Several calls to this function may have to be made to 
	///  obtain the full file path.
	///
	/// The full file path is formed by a list of folder and
	///  by the file name; for each call to this function
	///  the caller has to specify a zero based part number.
	///
	/// The last valid part contains the file name, while
	///  the parts before the last one contain the folder
	///  in which the file is stored; each folder is a child
	///  of the folder returned by the previous part.
	///
	/// The function returns an empty string if the
	///  specified part doesn't exist.
    ///
    /// See also setFilePart().
	/// 
	/// @param part the part to be returned, 0 based.
	/// @return the part's name (folder or file), or an
	///          empty string if the requested part doesn't
	///          exist
	///
	///////////////////////////////////////////////////////////
    fileParts_t getFileParts() const;

	/// \brief Set the full path to the  file referenced by
	///         the record.
	///
	/// Several calls to this function may have to be made to
	///  set the full file path.
	///
	/// The full file path is formed by a list of folder and
	///  by the file name; for each call to this function
	///  the caller has to specify a zero based part number.
	///
	/// The last valid part contains the file name, while
	///  the parts before the last one contain the folder
	///  in which the file is stored; each folder is a child
	///  of the folder set in the previous part.
	///
    /// For instance, the file /folder/file.dcm is set
    ///  with two calls to setFilePart():
    /// - setFilePart(0, "folder")
    /// - setFilePart(1, "file.dcm")
    ///
    /// See also getFilePart().
	///
	/// @param part the part to be set, 0 based.
    /// @param partName tha value to set for the part
	///
	///////////////////////////////////////////////////////////
    void setFileParts(const fileParts_t& fileParts);

	/// \brief Returns the record's type.
	///
	/// This function calls getTypeString() and convert the
	///  result to an enumerated value.
	///
	/// @return the record's type
	///
	///////////////////////////////////////////////////////////
    directoryRecordType_t getType() const;

	/// \brief Returns a string representing the record's type.
	///
	/// @return the record's type
	///
	///////////////////////////////////////////////////////////
    std::string getTypeString() const;

	/// \brief Sets the record's type.
	///
	/// @param recordType  the record's type
	///
	///////////////////////////////////////////////////////////
    void setType(directoryRecordType_t recordType);

	/// \brief Sets the record's type.
	///
	/// @param recordType  the string representing the record's
	///                     type
	///
	///////////////////////////////////////////////////////////
    void setTypeString(const std::string& recordType);

private:
    void checkCircularReference(directoryRecord* pStartRecord);

	void updateOffsets();

    struct tDirectoryRecordTypeDef
    {
        std::string m_name;
        directoryRecordType_t m_type;
    };

    static const tDirectoryRecordTypeDef* getRecordTypeMap();

	std::shared_ptr<directoryRecord> m_pNextRecord;
	std::shared_ptr<directoryRecord> m_pFirstChildRecord;

    std::shared_ptr<dataSet> m_pDataSet;
};


///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
/// \brief A dicomDir object contains the information about
///         the tree structure of a DICOMDIR dataset.
///
/// The dicomDir class connects to the DICOMDIR dataset
///  specified in the constructor and detects the DICOMDIR
///  tree structure that it contains.
///
/// Each directory record in the DICOMDIR structure is
///  represented by a directoryRecord class.
///
/// The first root directoryRecord can be obtained with
///  a call to getFirstRootRecord().
///
/// Any change made to the dicomDir class is immediately
///  reflected into the connected dataset.
///
/// WARNING: all the directoryRecord allocated by this
///  class are released only when the dicomDir itself
///  is released, even if the directoryRecord are 
///  explicitly released by your application.\n
/// 
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
class dicomDir
{
public:
    dicomDir();

	/// \brief Initializes a dicomDir object and attach it to 
	///         a dataset.
	///
	/// If a dataSet is specified then the constructor parses
	///  it and build an in memory representation of the
	///  directory tree and its records. The first root record
	///  can be retrieved with a call to getFirstRootRecord().
	///
	/// @param pDataSet   the dataSet that contains or will 
	///                    contain the DICOMDIR structure. 
	///                   If a null pointer is passed, then a
	///                    new dataSet is created by this
	///                    constructor
	///
	///////////////////////////////////////////////////////////
	dicomDir(std::shared_ptr<dataSet> pDataSet);

    // Destructor.
    //
    ///////////////////////////////////////////////////////////
    virtual ~dicomDir();

	/// \brief Returns the DICOMDIR dataset.
	///
	/// This is the same dataset specified in the constructor
	///  dicomDir::dicomDir() and returned by buildDataSet().
	///
	/// @return the dataset containing the DICOMDIR
	///          information
	///
	///////////////////////////////////////////////////////////
    std::shared_ptr<dataSet> getDirectoryDataSet() const;

	/// \brief Creates a new directoryRecord and embeds its
	///         dataSet into the DICOMDIR sequence of items.
	///
	/// Once the new directoryRecord has been returned, the
	///  calling application should set the proper record's
	///  values and specify the relationship with other items
	///  by calling setFirstRootRecord() or 
	///  directoryRecord::setNextRecord() or 
    ///  directoryRecord::setFirstChildRecord().
	///
	/// @return a new directoryRecord object belonging to the
	///         DICOMDIR
	///
	///////////////////////////////////////////////////////////
    std::shared_ptr<directoryRecord> getNewRecord(directoryRecordType_t recordType);

	/// \brief Returns the first root record in the DICOMDIR.
	///
	/// Once the first root record has been retrieved, the
	///  application can retrieve sibling root records by
	///  calling directoryRecord::getNextRecord().
	///
	/// @return the first root record in the DICOMDIR.
	///
	///////////////////////////////////////////////////////////
    std::shared_ptr<directoryRecord> getFirstRootRecord() const;

	/// \brief Sets the first root record in the DICOMDIR.
	///
	/// @param pFirstRootRecord a directoryRecord obtained with
	///                          a call to getNewRecord().
	///                         The directoryRecord will be the
	///                          first root record in the 
	///                          directory
	///////////////////////////////////////////////////////////
	void setFirstRootRecord(std::shared_ptr<directoryRecord> pFirstRootRecord);

	/// \brief Updates the dataSet containing the DICOMDIR
	///         with the information contained in the directory
	///         records.
	///
	/// Before building the DICOMDIR dataSet remember to set
	///  the following values in the dataSet (call
	///  getDirectoryDataSet() to get the dataSet):
	/// - tag 0x2,0x3: Media Storage SOP instance UID
	/// - tag 0x2,0x12: Implementation class UID
	/// - tag 0x2,0x13: Implementation version name
	/// - tag 0x2,0x16: Source application Entity Title
	///
	/// Please note that if any data is added to the directory
	///  or any of the directoryItem objects after a call to
	///  buildDataSet() then the dataset has to be rebuilt once
	///  more with another call to buildDataSet().
	///
	/// buildDataSet() doesn't return a new dataSet but 
	///  modifies the dataSet specified in the constructor.
	///
	/// @return a pointer to the updated dataSet
	///
	///////////////////////////////////////////////////////////
	std::shared_ptr<dataSet> buildDataSet();

protected:
	std::shared_ptr<dataSet> m_pDataSet;

	std::shared_ptr<directoryRecord> m_pFirstRootRecord;

	typedef std::list<std::shared_ptr<directoryRecord> > tRecordsList;
	tRecordsList m_recordsList;
};



/// @}

} // namespace implementation

} // namespace imebra

#endif // !defined(imebraDicomDir_93F684BF_0024_4bf3_89BA_D98E82A1F44C__INCLUDED_)
