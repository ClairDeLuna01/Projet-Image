#include "GaussianFilter.hpp"
#include <cmath>
#include <iostream>

Image GaussianFilter::apply(const Image &img)
{
    // On crée une image temporaire pour stocker le résultat
    Image out(img.getWidth(), img.getHeight());

    // On applique le filtre sur chaque pixel
    for (int y = 0; y < img.getHeight(); y++)
    {
        for (int x = 0; x < img.getWidth(); x++)
        {
            for (int c = 0; c < img.getChannels(); c++)
            {
                float sum = 0;
                float weightSum = 0;

                // On parcourt les pixels du voisinage
                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        // On calcule le poids du pixel
                        float weight = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                        weightSum += weight;

                        // On ajoute le poids multiplié par la valeur du pixel
                        sum += img.getClampedPixel(x + i, y + j, c) * weight;
                    }
                }

                // On normalise le résultat
                out.setPixel(x, y, c, sum / weightSum);
            }
        }
    }

    return out;
}