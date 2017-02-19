#ifdef JAVA

#include "onload.h"

JavaVM* m_javaVM = 0;


extern "C"
{

    //------------------------------------------------------------------------
    JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* aVm, void* aReserved)
    {
        // cache java VM
        m_javaVM = aVm;

        return JNI_VERSION_1_6;
    }

} // extern "C"


JavaVM* get_imebra_javaVM()
{
    return m_javaVM;
}

#endif
