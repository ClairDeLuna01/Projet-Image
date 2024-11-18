#include "NonlocalMeansFilter.hpp"
#include "extern/progressBar.hpp"

#include <cmath>
#include <unordered_map>
#include <vector>

float computePatchValueMemoize(const Image &img, int x, int y, int c, int patchSize)
{
    static std::unordered_map<int, float> memoize;
    int key = (y * img.getWidth() + x) * img.getChannels() + c;
    if (memoize.find(key) != memoize.end())
    {
        return memoize[key];
    }
    float total = 0;
    for (int di = -patchSize / 2; di < patchSize / 2; di++)
    {
        for (int dj = -patchSize / 2; dj < patchSize / 2; dj++)
        {
            total += img.getClampedPixel(x + di, y + dj, c);
        }
    }
    float mean = total / (patchSize * patchSize);
    memoize[key] = mean;
    return mean;
}

float NonlocalMeansFilter::computeWeight(const Image &img, int x, int y, int i, int j, int c)
{
    float meanp = computePatchValueMemoize(img, x, y, c, patchSize);
    float meanq = computePatchValueMemoize(img, i, j, c, patchSize);

    float weight = std::exp(-(abs((meanp * meanp) - (meanq * meanq)) / (h * h)));

    return weight;
}

Image NonlocalMeansFilter::apply(const Image &img)
{
    Image out = Image(img.getWidth(), img.getHeight());

    ProgressBar<int> progressBar(0, img.getHeight() * img.getWidth(), -1, ProgressBarFill::BLOCK);

    for (int y = 0; y < img.getHeight(); y++)
    {
        for (int x = 0; x < img.getWidth(); x++)
        {
            for (int c = 0; c < img.getChannels(); c++)
            {
                float totalWeight = 0;
                float newValue = 0;
                for (int j = 0; j < img.getHeight(); j++)
                {
                    for (int i = 0; i < img.getWidth(); i++)
                    {
                        float weight = computeWeight(img, x, y, i, j, c);
                        totalWeight += weight;
                        newValue += weight * img(i, j, c);
                    }
                }
                // std::cout << totalWeight << std::endl;
                out.setPixel(x, y, c, newValue / totalWeight);
            }
            progressBar++;
        }
    }

    return out;
}