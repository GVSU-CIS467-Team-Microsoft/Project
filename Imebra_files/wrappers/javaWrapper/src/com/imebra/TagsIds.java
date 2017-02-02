/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.8
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.imebra;

public class TagsIds {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected TagsIds(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(TagsIds obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        imebraJNI.delete_TagsIds(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public TagsIds() {
    this(imebraJNI.new_TagsIds__SWIG_0(), true);
  }

  public TagsIds(long n) {
    this(imebraJNI.new_TagsIds__SWIG_1(n), true);
  }

  public long size() {
    return imebraJNI.TagsIds_size(swigCPtr, this);
  }

  public long capacity() {
    return imebraJNI.TagsIds_capacity(swigCPtr, this);
  }

  public void reserve(long n) {
    imebraJNI.TagsIds_reserve(swigCPtr, this, n);
  }

  public boolean isEmpty() {
    return imebraJNI.TagsIds_isEmpty(swigCPtr, this);
  }

  public void clear() {
    imebraJNI.TagsIds_clear(swigCPtr, this);
  }

  public void add(TagId x) {
    imebraJNI.TagsIds_add(swigCPtr, this, TagId.getCPtr(x), x);
  }

  public TagId get(int i) {
    return new TagId(imebraJNI.TagsIds_get(swigCPtr, this, i), false);
  }

  public void set(int i, TagId val) {
    imebraJNI.TagsIds_set(swigCPtr, this, i, TagId.getCPtr(val), val);
  }

}
