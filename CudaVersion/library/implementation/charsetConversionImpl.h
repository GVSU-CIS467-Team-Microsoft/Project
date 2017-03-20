#if !defined(imebraCharsetConversion_3146DA5A_5276_4804_B9AB_A3D54C6B123A__INCLUDED_)
#define imebraCharsetConversion_3146DA5A_5276_4804_B9AB_A3D54C6B123A__INCLUDED_

#include "configurationImpl.h"
#include "charsetConversionIconvImpl.h"
#include "charsetConversionICUImpl.h"
#include "charsetConversionWindowsImpl.h"
#include "charsetConversionJavaImpl.h"
#include "charsetsListImpl.h"

#include <string>

namespace imebra
{

namespace implementation
{

class dicomConversion
{
public:
    static std::string convertFromUnicode(const std::wstring& unicodeString, charsetsList::tCharsetsList* pCharsets);
    static std::wstring convertToUnicode(const std::string& value, const charsetsList::tCharsetsList& charsets);
};

}

}

#endif // imebraCharsetConversion_3146DA5A_5276_4804_B9AB_A3D54C6B123A__INCLUDED_

