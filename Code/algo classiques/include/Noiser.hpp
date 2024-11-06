#include <random>
#include <algorithm>
#include "Image.hpp"

class Noiser {
public:
    // Applique un bruit gaussien avec un écart-type sigma
    static void applyGaussianNoise(Image& img, double sigma) {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, sigma);

        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                for (int c = 0; c < img.getChannels(); ++c) {
                    int pixelValue = img.getClampedPixel(x, y, c) + static_cast<int>(distribution(generator));
                    img.setPixel(x, y, c, static_cast<unsigned char>(std::clamp(pixelValue, 0, 255)));
                }
            }
        }
    }

    // Applique un bruit de Poisson
    static void applyPoissonNoise(Image& img) {
        std::default_random_engine generator;
        std::poisson_distribution<int> distribution(30); // Paramètre de Poisson ajustable

        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                for (int c = 0; c < img.getChannels(); ++c) {
                    int pixelValue = img.getClampedPixel(x, y, c) + distribution(generator);
                    img.setPixel(x, y, c, static_cast<unsigned char>(std::clamp(pixelValue, 0, 255)));
                }
            }
        }
    }

    // Applique un bruit de Sel et Poivre avec une probabilité p pour chaque pixel
    static void applySaltAndPepperNoise(Image& img, double p) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                if (distribution(generator) < p) {
                    unsigned char value = (distribution(generator) < 0.5) ? 0 : 255;
                    for (int c = 0; c < img.getChannels(); ++c) {
                        img.setPixel(x, y, c, value);
                    }
                }
            }
        }
    }

    // Applique un bruit de Speckle avec un écart-type sigma
    static void applySpeckleNoise(Image& img, double sigma) {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, sigma);

        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                for (int c = 0; c < img.getChannels(); ++c) {
                    int pixelValue = img.getClampedPixel(x, y, c) * (1 + distribution(generator));
                    img.setPixel(x, y, c, static_cast<unsigned char>(std::clamp(pixelValue, 0, 255)));
                }
            }
        }
    }
};
