/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.8
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.imebra;

public class imebraJNI {
  public final static native long new_FileParts__SWIG_0();
  public final static native long new_FileParts__SWIG_1(long jarg1);
  public final static native long FileParts_size(long jarg1, FileParts jarg1_);
  public final static native long FileParts_capacity(long jarg1, FileParts jarg1_);
  public final static native void FileParts_reserve(long jarg1, FileParts jarg1_, long jarg2);
  public final static native boolean FileParts_isEmpty(long jarg1, FileParts jarg1_);
  public final static native void FileParts_clear(long jarg1, FileParts jarg1_);
  public final static native void FileParts_add(long jarg1, FileParts jarg1_, String jarg2);
  public final static native String FileParts_get(long jarg1, FileParts jarg1_, int jarg2);
  public final static native void FileParts_set(long jarg1, FileParts jarg1_, int jarg2, String jarg3);
  public final static native void delete_FileParts(long jarg1);
  public final static native long new_Groups__SWIG_0();
  public final static native long new_Groups__SWIG_1(long jarg1);
  public final static native long Groups_size(long jarg1, Groups jarg1_);
  public final static native long Groups_capacity(long jarg1, Groups jarg1_);
  public final static native void Groups_reserve(long jarg1, Groups jarg1_, long jarg2);
  public final static native boolean Groups_isEmpty(long jarg1, Groups jarg1_);
  public final static native void Groups_clear(long jarg1, Groups jarg1_);
  public final static native void Groups_add(long jarg1, Groups jarg1_, int jarg2);
  public final static native int Groups_get(long jarg1, Groups jarg1_, int jarg2);
  public final static native void Groups_set(long jarg1, Groups jarg1_, int jarg2, int jarg3);
  public final static native void delete_Groups(long jarg1);
  public final static native long new_TagsIds__SWIG_0();
  public final static native long new_TagsIds__SWIG_1(long jarg1);
  public final static native long TagsIds_size(long jarg1, TagsIds jarg1_);
  public final static native long TagsIds_capacity(long jarg1, TagsIds jarg1_);
  public final static native void TagsIds_reserve(long jarg1, TagsIds jarg1_, long jarg2);
  public final static native boolean TagsIds_isEmpty(long jarg1, TagsIds jarg1_);
  public final static native void TagsIds_clear(long jarg1, TagsIds jarg1_);
  public final static native void TagsIds_add(long jarg1, TagsIds jarg1_, long jarg2, TagId jarg2_);
  public final static native long TagsIds_get(long jarg1, TagsIds jarg1_, int jarg2);
  public final static native void TagsIds_set(long jarg1, TagsIds jarg1_, int jarg2, long jarg3, TagId jarg3_);
  public final static native void delete_TagsIds(long jarg1);
  public final static native long new_VOIs__SWIG_0();
  public final static native long new_VOIs__SWIG_1(long jarg1);
  public final static native long VOIs_size(long jarg1, VOIs jarg1_);
  public final static native long VOIs_capacity(long jarg1, VOIs jarg1_);
  public final static native void VOIs_reserve(long jarg1, VOIs jarg1_, long jarg2);
  public final static native boolean VOIs_isEmpty(long jarg1, VOIs jarg1_);
  public final static native void VOIs_clear(long jarg1, VOIs jarg1_);
  public final static native void VOIs_add(long jarg1, VOIs jarg1_, long jarg2, VOIDescription jarg2_);
  public final static native long VOIs_get(long jarg1, VOIs jarg1_, int jarg2);
  public final static native void VOIs_set(long jarg1, VOIs jarg1_, int jarg2, long jarg3, VOIDescription jarg3_);
  public final static native void delete_VOIs(long jarg1);
  public final static native long new_TagId__SWIG_0();
  public final static native long new_TagId__SWIG_1(int jarg1, int jarg2);
  public final static native long new_TagId__SWIG_2(int jarg1, long jarg2, int jarg3);
  public final static native void delete_TagId(long jarg1);
  public final static native int TagId_getGroupId(long jarg1, TagId jarg1_);
  public final static native long TagId_getGroupOrder(long jarg1, TagId jarg1_);
  public final static native int TagId_getTagId(long jarg1, TagId jarg1_);
  public final static native int ageUnit_t_days_get();
  public final static native int ageUnit_t_weeks_get();
  public final static native int ageUnit_t_months_get();
  public final static native int ageUnit_t_years_get();
  public final static native int imageQuality_t_veryHigh_get();
  public final static native int imageQuality_t_high_get();
  public final static native int imageQuality_t_aboveMedium_get();
  public final static native int imageQuality_t_medium_get();
  public final static native int imageQuality_t_belowMedium_get();
  public final static native int imageQuality_t_low_get();
  public final static native int imageQuality_t_veryLow_get();
  public final static native int bitDepth_t_depthU8_get();
  public final static native int bitDepth_t_depthS8_get();
  public final static native int bitDepth_t_depthU16_get();
  public final static native int bitDepth_t_depthS16_get();
  public final static native int bitDepth_t_depthU32_get();
  public final static native int bitDepth_t_depthS32_get();
  public final static native int tagVR_t_AE_get();
  public final static native int tagVR_t_AS_get();
  public final static native int tagVR_t_AT_get();
  public final static native int tagVR_t_CS_get();
  public final static native int tagVR_t_DA_get();
  public final static native int tagVR_t_DS_get();
  public final static native int tagVR_t_DT_get();
  public final static native int tagVR_t_FL_get();
  public final static native int tagVR_t_FD_get();
  public final static native int tagVR_t_IS_get();
  public final static native int tagVR_t_LO_get();
  public final static native int tagVR_t_LT_get();
  public final static native int tagVR_t_OB_get();
  public final static native int tagVR_t_SB_get();
  public final static native int tagVR_t_OD_get();
  public final static native int tagVR_t_OF_get();
  public final static native int tagVR_t_OL_get();
  public final static native int tagVR_t_OW_get();
  public final static native int tagVR_t_PN_get();
  public final static native int tagVR_t_SH_get();
  public final static native int tagVR_t_SL_get();
  public final static native int tagVR_t_SQ_get();
  public final static native int tagVR_t_SS_get();
  public final static native int tagVR_t_ST_get();
  public final static native int tagVR_t_TM_get();
  public final static native int tagVR_t_UC_get();
  public final static native int tagVR_t_UI_get();
  public final static native int tagVR_t_UL_get();
  public final static native int tagVR_t_UN_get();
  public final static native int tagVR_t_UR_get();
  public final static native int tagVR_t_US_get();
  public final static native int tagVR_t_UT_get();
  public final static native int drawBitmapType_t_drawBitmapRGB_get();
  public final static native int drawBitmapType_t_drawBitmapBGR_get();
  public final static native int drawBitmapType_t_drawBitmapRGBA_get();
  public final static native int drawBitmapType_t_drawBitmapBGRA_get();
  public final static native long new_Age(long jarg1, int jarg2);
  public final static native double Age_years(long jarg1, Age jarg1_);
  public final static native void Age_age_set(long jarg1, Age jarg1_, long jarg2);
  public final static native long Age_age_get(long jarg1, Age jarg1_);
  public final static native void Age_units_set(long jarg1, Age jarg1_, int jarg2);
  public final static native int Age_units_get(long jarg1, Age jarg1_);
  public final static native void delete_Age(long jarg1);
  public final static native long new_Date(long jarg1, long jarg2, long jarg3, long jarg4, long jarg5, long jarg6, long jarg7, int jarg8, int jarg9);
  public final static native void Date_year_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_year_get(long jarg1, Date jarg1_);
  public final static native void Date_month_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_month_get(long jarg1, Date jarg1_);
  public final static native void Date_day_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_day_get(long jarg1, Date jarg1_);
  public final static native void Date_hour_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_hour_get(long jarg1, Date jarg1_);
  public final static native void Date_minutes_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_minutes_get(long jarg1, Date jarg1_);
  public final static native void Date_seconds_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_seconds_get(long jarg1, Date jarg1_);
  public final static native void Date_nanoseconds_set(long jarg1, Date jarg1_, long jarg2);
  public final static native long Date_nanoseconds_get(long jarg1, Date jarg1_);
  public final static native void Date_offsetHours_set(long jarg1, Date jarg1_, int jarg2);
  public final static native int Date_offsetHours_get(long jarg1, Date jarg1_);
  public final static native void Date_offsetMinutes_set(long jarg1, Date jarg1_, int jarg2);
  public final static native int Date_offsetMinutes_get(long jarg1, Date jarg1_);
  public final static native void delete_Date(long jarg1);
  public final static native void VOIDescription_center_set(long jarg1, VOIDescription jarg1_, double jarg2);
  public final static native double VOIDescription_center_get(long jarg1, VOIDescription jarg1_);
  public final static native void VOIDescription_width_set(long jarg1, VOIDescription jarg1_, double jarg2);
  public final static native double VOIDescription_width_get(long jarg1, VOIDescription jarg1_);
  public final static native void VOIDescription_description_set(long jarg1, VOIDescription jarg1_, String jarg2);
  public final static native String VOIDescription_description_get(long jarg1, VOIDescription jarg1_);
  public final static native long new_VOIDescription();
  public final static native void delete_VOIDescription(long jarg1);
  public final static native long new_ReadMemory__SWIG_0();
  public final static native long new_ReadMemory__SWIG_1(byte[] jarg1);
  public final static native void delete_ReadMemory(long jarg1);
  public final static native long ReadMemory_size(long jarg1, ReadMemory jarg1_);
  public final static native long ReadMemory_data(long jarg1, ReadMemory jarg1_, byte[] jarg2);
  public final static native void ReadMemory_regionData(long jarg1, ReadMemory jarg1_, byte[] jarg2, long jarg4);
  public final static native boolean ReadMemory_empty(long jarg1, ReadMemory jarg1_);
  public final static native long new_ReadWriteMemory__SWIG_0();
  public final static native long new_ReadWriteMemory__SWIG_1(long jarg1);
  public final static native long new_ReadWriteMemory__SWIG_2(long jarg1, ReadMemory jarg1_);
  public final static native long new_ReadWriteMemory__SWIG_3(byte[] jarg1);
  public final static native void delete_ReadWriteMemory(long jarg1);
  public final static native void ReadWriteMemory_copyFrom(long jarg1, ReadWriteMemory jarg1_, long jarg2, ReadMemory jarg2_);
  public final static native void ReadWriteMemory_clear(long jarg1, ReadWriteMemory jarg1_);
  public final static native void ReadWriteMemory_resize(long jarg1, ReadWriteMemory jarg1_, long jarg2);
  public final static native void ReadWriteMemory_reserve(long jarg1, ReadWriteMemory jarg1_, long jarg2);
  public final static native void ReadWriteMemory_assign(long jarg1, ReadWriteMemory jarg1_, byte[] jarg2);
  public final static native void ReadWriteMemory_assignRegion(long jarg1, ReadWriteMemory jarg1_, byte[] jarg2, long jarg4);
  public final static native void MemoryPool_flush();
  public final static native long MemoryPool_getUnusedMemorySize();
  public final static native void MemoryPool_setMemoryPoolSize(long jarg1, long jarg2);
  public final static native long new_MemoryPool();
  public final static native void delete_MemoryPool(long jarg1);
  public final static native void delete_BaseStreamInput(long jarg1);
  public final static native void delete_BaseStreamOutput(long jarg1);
  public final static native long new_StreamReader__SWIG_0(long jarg1, BaseStreamInput jarg1_);
  public final static native long new_StreamReader__SWIG_1(long jarg1, BaseStreamInput jarg1_, long jarg2, long jarg3);
  public final static native void delete_StreamReader(long jarg1);
  public final static native long new_StreamWriter__SWIG_0(long jarg1, BaseStreamOutput jarg1_);
  public final static native long new_StreamWriter__SWIG_1(long jarg1, BaseStreamOutput jarg1_, long jarg2, long jarg3);
  public final static native void delete_StreamWriter(long jarg1);
  public final static native void delete_ReadingDataHandler(long jarg1);
  public final static native long ReadingDataHandler_getSize(long jarg1, ReadingDataHandler jarg1_);
  public final static native int ReadingDataHandler_getDataType(long jarg1, ReadingDataHandler jarg1_);
  public final static native int ReadingDataHandler_getSignedLong(long jarg1, ReadingDataHandler jarg1_, long jarg2);
  public final static native long ReadingDataHandler_getUnsignedLong(long jarg1, ReadingDataHandler jarg1_, long jarg2);
  public final static native double ReadingDataHandler_getDouble(long jarg1, ReadingDataHandler jarg1_, long jarg2);
  public final static native String ReadingDataHandler_getString(long jarg1, ReadingDataHandler jarg1_, long jarg2);
  public final static native long ReadingDataHandler_getDate(long jarg1, ReadingDataHandler jarg1_, long jarg2);
  public final static native long ReadingDataHandler_getAge(long jarg1, ReadingDataHandler jarg1_, long jarg2);
  public final static native void delete_ReadingDataHandlerNumeric(long jarg1);
  public final static native long ReadingDataHandlerNumeric_getMemory(long jarg1, ReadingDataHandlerNumeric jarg1_);
  public final static native long ReadingDataHandlerNumeric_data(long jarg1, ReadingDataHandlerNumeric jarg1_, byte[] jarg2);
  public final static native long ReadingDataHandlerNumeric_getUnitSize(long jarg1, ReadingDataHandlerNumeric jarg1_);
  public final static native boolean ReadingDataHandlerNumeric_isSigned(long jarg1, ReadingDataHandlerNumeric jarg1_);
  public final static native boolean ReadingDataHandlerNumeric_isFloat(long jarg1, ReadingDataHandlerNumeric jarg1_);
  public final static native void ReadingDataHandlerNumeric_copyTo(long jarg1, ReadingDataHandlerNumeric jarg1_, long jarg2, WritingDataHandlerNumeric jarg2_);
  public final static native void delete_WritingDataHandler(long jarg1);
  public final static native void WritingDataHandler_setSize(long jarg1, WritingDataHandler jarg1_, long jarg2);
  public final static native long WritingDataHandler_getSize(long jarg1, WritingDataHandler jarg1_);
  public final static native int WritingDataHandler_getDataType(long jarg1, WritingDataHandler jarg1_);
  public final static native void WritingDataHandler_setSignedLong(long jarg1, WritingDataHandler jarg1_, long jarg2, int jarg3);
  public final static native void WritingDataHandler_setUnsignedLong(long jarg1, WritingDataHandler jarg1_, long jarg2, long jarg3);
  public final static native void WritingDataHandler_setDouble(long jarg1, WritingDataHandler jarg1_, long jarg2, double jarg3);
  public final static native void WritingDataHandler_setString(long jarg1, WritingDataHandler jarg1_, long jarg2, String jarg3);
  public final static native void WritingDataHandler_setDate(long jarg1, WritingDataHandler jarg1_, long jarg2, long jarg3, Date jarg3_);
  public final static native void WritingDataHandler_setAge(long jarg1, WritingDataHandler jarg1_, long jarg2, long jarg3, Age jarg3_);
  public final static native void delete_WritingDataHandlerNumeric(long jarg1);
  public final static native long WritingDataHandlerNumeric_getMemory(long jarg1, WritingDataHandlerNumeric jarg1_);
  public final static native void WritingDataHandlerNumeric_assign(long jarg1, WritingDataHandlerNumeric jarg1_, byte[] jarg2);
  public final static native long WritingDataHandlerNumeric_data(long jarg1, WritingDataHandlerNumeric jarg1_, byte[] jarg2);
  public final static native long WritingDataHandlerNumeric_getUnitSize(long jarg1, WritingDataHandlerNumeric jarg1_);
  public final static native boolean WritingDataHandlerNumeric_isSigned(long jarg1, WritingDataHandlerNumeric jarg1_);
  public final static native boolean WritingDataHandlerNumeric_isFloat(long jarg1, WritingDataHandlerNumeric jarg1_);
  public final static native void WritingDataHandlerNumeric_copyFrom(long jarg1, WritingDataHandlerNumeric jarg1_, long jarg2, ReadingDataHandlerNumeric jarg2_);
  public final static native void delete_LUT(long jarg1);
  public final static native String LUT_getDescription(long jarg1, LUT jarg1_);
  public final static native long LUT_getReadingDataHandler(long jarg1, LUT jarg1_);
  public final static native long LUT_getBits(long jarg1, LUT jarg1_);
  public final static native long LUT_getSize(long jarg1, LUT jarg1_);
  public final static native int LUT_getFirstMapped(long jarg1, LUT jarg1_);
  public final static native long LUT_getMappedValue(long jarg1, LUT jarg1_, int jarg2);
  public final static native long new_Image(long jarg1, long jarg2, int jarg3, String jarg4, long jarg5);
  public final static native void delete_Image(long jarg1);
  public final static native double Image_getWidthMm(long jarg1, Image jarg1_);
  public final static native double Image_getHeightMm(long jarg1, Image jarg1_);
  public final static native void Image_setSizeMm(long jarg1, Image jarg1_, double jarg2, double jarg3);
  public final static native long Image_getWidth(long jarg1, Image jarg1_);
  public final static native long Image_getHeight(long jarg1, Image jarg1_);
  public final static native long Image_getReadingDataHandler(long jarg1, Image jarg1_);
  public final static native long Image_getWritingDataHandler(long jarg1, Image jarg1_);
  public final static native String Image_getColorSpace(long jarg1, Image jarg1_);
  public final static native long Image_getChannelsNumber(long jarg1, Image jarg1_);
  public final static native int Image_getDepth(long jarg1, Image jarg1_);
  public final static native long Image_getHighBit(long jarg1, Image jarg1_);
  public final static native void delete_Tag(long jarg1);
  public final static native long Tag_getBuffersCount(long jarg1, Tag jarg1_);
  public final static native boolean Tag_bufferExists(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getBufferSize(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getReadingDataHandler(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getWritingDataHandler(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getReadingDataHandlerNumeric(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getReadingDataHandlerRaw(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getWritingDataHandlerNumeric(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getWritingDataHandlerRaw(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getStreamReader(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getStreamWriter(long jarg1, Tag jarg1_, long jarg2);
  public final static native long Tag_getSequenceItem(long jarg1, Tag jarg1_, long jarg2);
  public final static native boolean Tag_sequenceItemExists(long jarg1, Tag jarg1_, long jarg2);
  public final static native void Tag_setSequenceItem(long jarg1, Tag jarg1_, long jarg2, long jarg3, DataSet jarg3_);
  public final static native void Tag_appendSequenceItem(long jarg1, Tag jarg1_, long jarg2, DataSet jarg2_);
  public final static native int Tag_getDataType(long jarg1, Tag jarg1_);
  public final static native long new_DataSet__SWIG_0();
  public final static native long new_DataSet__SWIG_1(String jarg1);
  public final static native long new_DataSet__SWIG_2(String jarg1, long jarg2, FileParts jarg2_);
  public final static native void delete_DataSet(long jarg1);
  public final static native long DataSet_getTags(long jarg1, DataSet jarg1_);
  public final static native long DataSet_getTag(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_);
  public final static native long DataSet_getTagCreate__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, int jarg3);
  public final static native long DataSet_getTagCreate__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_);
  public final static native long DataSet_getImage(long jarg1, DataSet jarg1_, long jarg2);
  public final static native long DataSet_getImageApplyModalityTransform(long jarg1, DataSet jarg1_, long jarg2);
  public final static native void DataSet_setImage(long jarg1, DataSet jarg1_, long jarg2, long jarg3, Image jarg3_, int jarg4);
  public final static native long DataSet_getVOIs(long jarg1, DataSet jarg1_);
  public final static native long DataSet_getSequenceItem(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native void DataSet_setSequenceItem(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, long jarg4, DataSet jarg4_);
  public final static native long DataSet_getLUT(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getReadingDataHandler(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getWritingDataHandler__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, int jarg4);
  public final static native long DataSet_getWritingDataHandler__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getReadingDataHandlerNumeric(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getReadingDataHandlerRaw(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getWritingDataHandlerNumeric__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, int jarg4);
  public final static native long DataSet_getWritingDataHandlerNumeric__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getWritingDataHandlerRaw__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, int jarg4);
  public final static native long DataSet_getWritingDataHandlerRaw__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native boolean DataSet_bufferExists(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native int DataSet_getSignedLong__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native int DataSet_getSignedLong__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, int jarg4);
  public final static native void DataSet_setSignedLong__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, int jarg3, int jarg4);
  public final static native void DataSet_setSignedLong__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, int jarg3);
  public final static native long DataSet_getUnsignedLong__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getUnsignedLong__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, long jarg4);
  public final static native void DataSet_setUnsignedLong__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, int jarg4);
  public final static native void DataSet_setUnsignedLong__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native double DataSet_getDouble__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native double DataSet_getDouble__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, double jarg4);
  public final static native void DataSet_setDouble__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, double jarg3, int jarg4);
  public final static native void DataSet_setDouble__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, double jarg3);
  public final static native String DataSet_getString__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native String DataSet_getString__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, String jarg4);
  public final static native void DataSet_setString__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, String jarg3, int jarg4);
  public final static native void DataSet_setString__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, String jarg3);
  public final static native long DataSet_getAge__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getAge__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, long jarg4, Age jarg4_);
  public final static native void DataSet_setAge(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, Age jarg3_);
  public final static native long DataSet_getDate__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3);
  public final static native long DataSet_getDate__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, long jarg4, Date jarg4_);
  public final static native void DataSet_setDate__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, Date jarg3_, int jarg4);
  public final static native void DataSet_setDate__SWIG_1(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_, long jarg3, Date jarg3_);
  public final static native int DataSet_getDataType(long jarg1, DataSet jarg1_, long jarg2, TagId jarg2_);
  public final static native long CodecFactory_load__SWIG_0(long jarg1, StreamReader jarg1_, long jarg2);
  public final static native long CodecFactory_load__SWIG_1(long jarg1, StreamReader jarg1_);
  public final static native long CodecFactory_load__SWIG_2(String jarg1, long jarg2);
  public final static native long CodecFactory_load__SWIG_3(String jarg1);
  public final static native void CodecFactory_saveImage(long jarg1, StreamWriter jarg1_, long jarg2, Image jarg2_, String jarg3, int jarg4, int jarg5, long jarg6, boolean jarg7, boolean jarg8, boolean jarg9, boolean jarg10);
  public final static native void CodecFactory_save__SWIG_0(long jarg1, DataSet jarg1_, long jarg2, StreamWriter jarg2_, int jarg3);
  public final static native void CodecFactory_save__SWIG_1(long jarg1, DataSet jarg1_, String jarg2, int jarg3);
  public final static native void CodecFactory_setMaximumImageSize(long jarg1, long jarg2);
  public final static native long new_CodecFactory();
  public final static native void delete_CodecFactory(long jarg1);
  public final static native void delete_Transform(long jarg1);
  public final static native boolean Transform_isEmpty(long jarg1, Transform jarg1_);
  public final static native long Transform_allocateOutputImage(long jarg1, Transform jarg1_, long jarg2, Image jarg2_, long jarg3, long jarg4);
  public final static native void Transform_runTransform(long jarg1, Transform jarg1_, long jarg2, Image jarg2_, long jarg3, long jarg4, long jarg5, long jarg6, long jarg7, Image jarg7_, long jarg8, long jarg9);
  public final static native long new_TransformHighBit();
  public final static native void delete_TransformHighBit(long jarg1);
  public final static native long new_TransformsChain();
  public final static native void delete_TransformsChain(long jarg1);
  public final static native void TransformsChain_addTransform(long jarg1, TransformsChain jarg1_, long jarg2, Transform jarg2_);
  public final static native long new_ModalityVOILUT(long jarg1, DataSet jarg1_);
  public final static native void delete_ModalityVOILUT(long jarg1);
  public final static native long new_VOILUT();
  public final static native void delete_VOILUT(long jarg1);
  public final static native void VOILUT_applyOptimalVOI(long jarg1, VOILUT jarg1_, long jarg2, Image jarg2_, long jarg3, long jarg4, long jarg5, long jarg6);
  public final static native void VOILUT_setCenterWidth(long jarg1, VOILUT jarg1_, double jarg2, double jarg3);
  public final static native void VOILUT_setLUT(long jarg1, VOILUT jarg1_, long jarg2, LUT jarg2_);
  public final static native double VOILUT_getCenter(long jarg1, VOILUT jarg1_);
  public final static native double VOILUT_getWidth(long jarg1, VOILUT jarg1_);
  public final static native String ColorTransformsFactory_normalizeColorSpace(String jarg1);
  public final static native boolean ColorTransformsFactory_isMonochrome(String jarg1);
  public final static native boolean ColorTransformsFactory_isSubsampledX(String jarg1);
  public final static native boolean ColorTransformsFactory_isSubsampledY(String jarg1);
  public final static native boolean ColorTransformsFactory_canSubsample(String jarg1);
  public final static native String ColorTransformsFactory_makeSubsampled(String jarg1, boolean jarg2, boolean jarg3);
  public final static native long ColorTransformsFactory_getNumberOfChannels(String jarg1);
  public final static native long ColorTransformsFactory_getTransform(String jarg1, String jarg2);
  public final static native long new_ColorTransformsFactory();
  public final static native void delete_ColorTransformsFactory(long jarg1);
  public final static native void delete_DicomDirEntry(long jarg1);
  public final static native long DicomDirEntry_getEntryDataSet(long jarg1, DicomDirEntry jarg1_);
  public final static native long DicomDirEntry_getNextEntry(long jarg1, DicomDirEntry jarg1_);
  public final static native long DicomDirEntry_getFirstChildEntry(long jarg1, DicomDirEntry jarg1_);
  public final static native void DicomDirEntry_setNextEntry(long jarg1, DicomDirEntry jarg1_, long jarg2, DicomDirEntry jarg2_);
  public final static native void DicomDirEntry_setFirstChildEntry(long jarg1, DicomDirEntry jarg1_, long jarg2, DicomDirEntry jarg2_);
  public final static native long DicomDirEntry_getFileParts(long jarg1, DicomDirEntry jarg1_);
  public final static native void DicomDirEntry_setFileParts(long jarg1, DicomDirEntry jarg1_, long jarg2, FileParts jarg2_);
  public final static native int DicomDirEntry_getType(long jarg1, DicomDirEntry jarg1_);
  public final static native String DicomDirEntry_getTypeString(long jarg1, DicomDirEntry jarg1_);
  public final static native long new_DicomDir__SWIG_0();
  public final static native long new_DicomDir__SWIG_1(long jarg1, DataSet jarg1_);
  public final static native void delete_DicomDir(long jarg1);
  public final static native long DicomDir_getNewEntry(long jarg1, DicomDir jarg1_, int jarg2);
  public final static native long DicomDir_getFirstRootEntry(long jarg1, DicomDir jarg1_);
  public final static native void DicomDir_setFirstRootEntry(long jarg1, DicomDir jarg1_, long jarg2, DicomDirEntry jarg2_);
  public final static native long DicomDir_updateDataSet(long jarg1, DicomDir jarg1_);
  public final static native String DicomDictionary_getUnicodeTagName(long jarg1, TagId jarg1_);
  public final static native String DicomDictionary_getTagName(long jarg1, TagId jarg1_);
  public final static native int DicomDictionary_getTagType(long jarg1, TagId jarg1_);
  public final static native long DicomDictionary_getWordSize(int jarg1);
  public final static native long DicomDictionary_getMaxSize(int jarg1);
  public final static native long new_DicomDictionary();
  public final static native void delete_DicomDictionary(long jarg1);
  public final static native long new_DrawBitmap__SWIG_0();
  public final static native long new_DrawBitmap__SWIG_1(long jarg1, Transform jarg1_);
  public final static native void delete_DrawBitmap(long jarg1);
  public final static native long DrawBitmap_getBitmap__SWIG_0(long jarg1, DrawBitmap jarg1_, long jarg2, Image jarg2_, int jarg3, long jarg4, byte[] jarg5);
  public final static native long DrawBitmap_getBitmap__SWIG_1(long jarg1, DrawBitmap jarg1_, long jarg2, Image jarg2_, int jarg3, long jarg4);
  public final static native long new_FileStreamInput(String jarg1);
  public final static native void delete_FileStreamInput(long jarg1);
  public final static native long new_FileStreamOutput(String jarg1);
  public final static native void delete_FileStreamOutput(long jarg1);
  public final static native long new_MemoryStreamInput__SWIG_0(long jarg1, ReadMemory jarg1_);
  public final static native long new_MemoryStreamInput__SWIG_1(long jarg1, ReadWriteMemory jarg1_);
  public final static native void delete_MemoryStreamInput(long jarg1);
  public final static native long new_MemoryStreamOutput(long jarg1, ReadWriteMemory jarg1_);
  public final static native void delete_MemoryStreamOutput(long jarg1);
  public final static native long ReadWriteMemory_SWIGUpcast(long jarg1);
  public final static native long ReadingDataHandlerNumeric_SWIGUpcast(long jarg1);
  public final static native long WritingDataHandlerNumeric_SWIGUpcast(long jarg1);
  public final static native long TransformHighBit_SWIGUpcast(long jarg1);
  public final static native long TransformsChain_SWIGUpcast(long jarg1);
  public final static native long ModalityVOILUT_SWIGUpcast(long jarg1);
  public final static native long VOILUT_SWIGUpcast(long jarg1);
  public final static native long FileStreamInput_SWIGUpcast(long jarg1);
  public final static native long FileStreamOutput_SWIGUpcast(long jarg1);
  public final static native long MemoryStreamInput_SWIGUpcast(long jarg1);
  public final static native long MemoryStreamOutput_SWIGUpcast(long jarg1);
}
