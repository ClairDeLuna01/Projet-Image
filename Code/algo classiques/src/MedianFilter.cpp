#include "MedianFilter.hpp"

#include <algorithm>
#include <vector>

Image MedianFilter::apply(const Image &img)
{
    Image out = Image(img.getWidth(), img.getHeight(), img.getChannels());

    for (int y = 0; y < img.getHeight(); y++)
    {
        for (int x = 0; x < img.getWidth(); x++)
        {
            for (int c = 0; c < img.getChannels(); c++)
            {
                std::vector<unsigned char> values;
                for (int j = -kernelSize / 2; j <= kernelSize / 2; j++)
                {
                    for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
                    {
                        values.push_back(img.getClampedPixel(x + i, y + j, c));
                    }
                }
                std::sort(values.begin(), values.end());
                out(x, y, c) = values[values.size() / 2];
            }
        }
    }

    return out;
}