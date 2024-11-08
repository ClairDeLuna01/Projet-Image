#include "BilateralFilter.hpp"
#include "utils.hpp"

#include <cmath>

Image BilateralFilter::apply(const Image &img)
{
    // On crée une image temporaire pour stocker le résultat
    Image out(img.getWidth(), img.getHeight(), img.getChannels());

    // On parcourt chaque pixel de l'image
    for (int x = 0; x < img.getHeight(); x++)
    {
        for (int y = 0; y < img.getWidth(); y++)
        {
            for (int c = 0; c < img.getChannels(); c++)
            {
                if (x < kernelSize / 2 || x >= img.getHeight() - kernelSize / 2 || y < kernelSize / 2 ||
                    y >= img.getWidth() - kernelSize / 2)
                {
                    out.setPixel(x, y, c, img.getClampedPixel(x, y, c));
                }
                float sum = 0;
                float weightSum = 0;

                // On parcourt les pixels du voisinage
                for (int j = -kernelSize / 2; j <= kernelSize / 2; j++)
                {
                    for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
                    {
                        float nx = x + i;
                        float ny = y + j;

                        float p = img.getClampedPixel(x, y, c);
                        float q = img.getClampedPixel(nx, ny, c);

                        float gd = std::exp(-((x - nx) * (x - nx) + (y - ny) * (y - ny)) / (2.0f * sigmaD * sigmaD));
                        float gr = std::exp(-((p - q) * (p - q)) / (2.0f * sigmaR * sigmaR));
                        float w = gd * gr;

                        sum += q * w;
                        weightSum += w;
                    }
                }

                // On normalise le résultat
                out.setPixel(x, y, c, utils::clamp(sum / weightSum, 0.0f, 255.0f));
            }
        }
    }

    return out;
}