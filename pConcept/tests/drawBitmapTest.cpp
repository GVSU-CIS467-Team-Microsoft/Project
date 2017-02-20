#include <imebra/imebra.h>
#include "buildImageForTest.h"
#include <gtest/gtest.h>

namespace imebra
{

namespace tests
{

TEST(drawBitmapTest, testDrawBitmapRGB)
{
	for(int monochrome(0); monochrome != 2; ++monochrome)
	{
        for(int highBit = 15; highBit != 2; --highBit)
		{
            std::unique_ptr<Image> testImage(buildImageForTest(
                    401,
                    301,
                    bitDepth_t::depthU32,
                    highBit,
                    401,
                    301,
                    monochrome == 1 ? "MONOCHROME2" :
                                      "RGB",
					50));

            DataSet testDataSet("1.2.840.10008.1.2");
            testDataSet.setImage(0, *testImage, imageQuality_t::high);

            DrawBitmap testDraw;
            std::unique_ptr<ReadWriteMemory> bitmapBuffer(testDraw.getBitmap(*testImage, drawBitmapType_t::drawBitmapRGB, 1));
            size_t bufferSize;
            char* pBuffer(bitmapBuffer->data(&bufferSize));

            std::unique_ptr<ReadingDataHandler> imageHandler(testImage->getReadingDataHandler());

            std::uint32_t red, green, blue;
            size_t index(0);
            for(std::uint32_t scanY = 0; scanY != testImage->getHeight(); ++scanY)
			{
                for(std::uint32_t scanX = 0; scanX != testImage->getWidth(); ++scanX)
				{
                    std::uint8_t displayRed(*pBuffer++);
                    std::uint8_t displayGreen(*pBuffer++);
                    std::uint8_t displayBlue(*pBuffer++);

                    if(monochrome)
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                        }
                        green = blue = red;
                    }
                    else
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        green = imageHandler->getUnsignedLong(index++);
                        blue = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                            green >>= (highBit - 7);
                            blue >>= (highBit - 7);
                        }
                    }

                    if(highBit < 7)
                    {
                        displayRed >>= (7 - highBit);
                        displayGreen >>= (7 - highBit);
                        displayBlue >>= (7 - highBit);
                    }
                    EXPECT_EQ(red, displayRed);
                    EXPECT_EQ(green, displayGreen);
                    EXPECT_EQ(blue, displayBlue);
				}

			}
		}
	}
}


TEST(drawBitmapTest, testDrawBitmapBGR)
{
    for(int monochrome(0); monochrome != 2; ++monochrome)
    {
        for(int highBit = 15; highBit != 2; --highBit)
        {
            std::unique_ptr<Image> testImage(buildImageForTest(
                    401,
                    301,
                    bitDepth_t::depthU32,
                    highBit,
                    401,
                    301,
                    monochrome == 1 ? "MONOCHROME2" :
                                      "RGB",
                    50));

            DataSet testDataSet("1.2.840.10008.1.2");
            testDataSet.setImage(0, *testImage, imageQuality_t::high);

            DrawBitmap testDraw;
            std::unique_ptr<ReadWriteMemory> bitmapBuffer(testDraw.getBitmap(*testImage, drawBitmapType_t::drawBitmapBGR, 4));
            size_t bufferSize;
            char* pBuffer(bitmapBuffer->data(&bufferSize));

            std::unique_ptr<ReadingDataHandler> imageHandler(testImage->getReadingDataHandler());

            std::uint32_t red, green, blue;
            size_t index(0);
            for(int scanY = 0; scanY != testImage->getHeight(); ++scanY)
            {
                for(int scanX = 0; scanX != testImage->getWidth(); ++scanX)
                {
                    std::uint8_t displayBlue(*pBuffer++);
                    std::uint8_t displayGreen(*pBuffer++);
                    std::uint8_t displayRed(*pBuffer++);

                    if(monochrome)
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                        }
                        green = blue = red;
                    }
                    else
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        green = imageHandler->getUnsignedLong(index++);
                        blue = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                            green >>= (highBit - 7);
                            blue >>= (highBit - 7);
                        }
                    }

                    if(highBit < 7)
                    {
                        displayRed >>= (7 - highBit);
                        displayGreen >>= (7 - highBit);
                        displayBlue >>= (7 - highBit);
                    }
                    EXPECT_EQ(red, displayRed);
                    EXPECT_EQ(green, displayGreen);
                    EXPECT_EQ(blue, displayBlue);
                }
                pBuffer++; // 4 bytes alignment

            }
        }
    }
}

TEST(drawBitmapTest, testDrawBitmapRGBA)
{
    for(int monochrome(0); monochrome != 2; ++monochrome)
    {
        for(int highBit = 15; highBit != 2; --highBit)
        {
            std::unique_ptr<Image> testImage(buildImageForTest(
                    401,
                    301,
                    bitDepth_t::depthU32,
                    highBit,
                    401,
                    301,
                    monochrome == 1 ? "MONOCHROME2" :
                                      "RGB",
                    50));

            DataSet testDataSet("1.2.840.10008.1.2");
            testDataSet.setImage(0, *testImage, imageQuality_t::high);

            DrawBitmap testDraw;
            std::unique_ptr<ReadWriteMemory> bitmapBuffer(testDraw.getBitmap(*testImage, drawBitmapType_t::drawBitmapRGBA, 1));
            size_t bufferSize;
            char* pBuffer(bitmapBuffer->data(&bufferSize));

            std::unique_ptr<ReadingDataHandler> imageHandler(testImage->getReadingDataHandler());

            std::uint32_t red, green, blue;
            size_t index(0);
            for(int scanY = 0; scanY != testImage->getHeight(); ++scanY)
            {
                for(int scanX = 0; scanX != testImage->getWidth(); ++scanX)
                {
                    std::uint8_t displayRed(*pBuffer++);
                    std::uint8_t displayGreen(*pBuffer++);
                    std::uint8_t displayBlue(*pBuffer++);
                    std::uint8_t displayAlpha(*pBuffer++);

                    if(monochrome)
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                        }
                        green = blue = red;
                    }
                    else
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        green = imageHandler->getUnsignedLong(index++);
                        blue = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                            green >>= (highBit - 7);
                            blue >>= (highBit - 7);
                        }
                    }

                    if(highBit < 7)
                    {
                        displayRed >>= (7 - highBit);
                        displayGreen >>= (7 - highBit);
                        displayBlue >>= (7 - highBit);
                    }
                    EXPECT_EQ(red, displayRed);
                    EXPECT_EQ(green, displayGreen);
                    EXPECT_EQ(blue, displayBlue);
                    EXPECT_EQ(255, displayAlpha);
                }
            }
        }
    }
}

TEST(drawBitmapTest, testDrawBitmapBGRA)
{
    for(int monochrome(0); monochrome != 2; ++monochrome)
    {
        for(int highBit = 15; highBit != 2; --highBit)
        {
            std::unique_ptr<Image> testImage(buildImageForTest(
                    401,
                    301,
                    bitDepth_t::depthU32,
                    highBit,
                    401,
                    301,
                    monochrome == 1 ? "MONOCHROME2" :
                                      "RGB",
                    50));

            DataSet testDataSet("1.2.840.10008.1.2");
            testDataSet.setImage(0, *testImage, imageQuality_t::high);

            DrawBitmap testDraw;
            std::unique_ptr<ReadWriteMemory> bitmapBuffer(testDraw.getBitmap(*testImage, drawBitmapType_t::drawBitmapBGRA, 4));
            size_t bufferSize;
            char* pBuffer(bitmapBuffer->data(&bufferSize));

            std::unique_ptr<ReadingDataHandler> imageHandler(testImage->getReadingDataHandler());

            std::uint32_t red, green, blue;
            size_t index(0);
            for(int scanY = 0; scanY != testImage->getHeight(); ++scanY)
            {
                for(int scanX = 0; scanX != testImage->getWidth(); ++scanX)
                {
                    std::uint8_t displayBlue(*pBuffer++);
                    std::uint8_t displayGreen(*pBuffer++);
                    std::uint8_t displayRed(*pBuffer++);
                    std::uint8_t displayAlpha(*pBuffer++);

                    if(monochrome)
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                        }
                        green = blue = red;
                    }
                    else
                    {
                        red = imageHandler->getUnsignedLong(index++);
                        green = imageHandler->getUnsignedLong(index++);
                        blue = imageHandler->getUnsignedLong(index++);
                        if(highBit > 7)
                        {
                            red >>= (highBit - 7);
                            green >>= (highBit - 7);
                            blue >>= (highBit - 7);
                        }
                    }

                    if(highBit < 7)
                    {
                        displayRed >>= (7 - highBit);
                        displayGreen >>= (7 - highBit);
                        displayBlue >>= (7 - highBit);
                    }
                    EXPECT_EQ(red, displayRed);
                    EXPECT_EQ(green, displayGreen);
                    EXPECT_EQ(blue, displayBlue);
                    EXPECT_EQ(255, displayAlpha);
                }
            }
        }
    }
}

} // namespace tests

} // namespace imebra
