#include "BilinearFilter.hpp"
#include <iostream>

Image BilinearFilter::apply(const Image &img)
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
                // Coordonnées du pixel dans l'image d'entrée
                float srcX = x + 0.5f;
                float srcY = y + 0.5f;

                // Coordonnées du pixel dans l'image de sortie
                float dstX = srcX * img.getWidth() / out.getWidth();
                float dstY = srcY * img.getHeight() / out.getHeight();

                // Coordonnées des pixels voisins
                int x0 = static_cast<int>(dstX);
                int y0 = static_cast<int>(dstY);
                int x1 = std::min(x0 + 1, out.getWidth() - 1);
                int y1 = std::min(y0 + 1, out.getHeight() - 1);

                // Coefficients de pondération
                float dx = dstX - x0;
                float dy = dstY - y0;

                // Interpolation bilinéaire
                float value = (1 - dx) * (1 - dy) * img.getClampedPixel(x0, y0, c) +
                              dx * (1 - dy) * img.getClampedPixel(x1, y0, c) +
                              (1 - dx) * dy * img.getClampedPixel(x0, y1, c) + dx * dy * img.getClampedPixel(x1, y1, c);

                // On assigne la valeur interpolée au pixel de l'image de sortie
                out.setPixel(x, y, c, static_cast<unsigned char>(value));
            }
        }
    }

    return out;
}