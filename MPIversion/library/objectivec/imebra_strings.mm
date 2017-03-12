#import "../include/imebra/objectivec/imebra_strings.h"
#import <Foundation/NSString.h>

namespace imebra
{

std::string NSStringToString ( NSString* str )
{
    return std::string([str UTF8String]);
}

NSString* stringToNSString ( const std::string& str )
{
    return [ [ NSString alloc] initWithUTF8String: str.c_str()];
}

} // namespace imebra
