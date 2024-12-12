#pragma once

#include "Image.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

enum class FilterType
{
    BILATERAL,
    GAUSSIAN,
    NONLOCAL_MEANS,
    MEDIAN,
    FFDNET_PRETRAINED,
    FFDNET,
    INVALID
};

enum class NoiseType
{
    GAUSSIAN,
    POISSON,
    SALT_PEPPER,
    SPECKLE
};

namespace utils
{
inline float MSE(const Image &img1, const Image &img2)
{
    if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() ||
        img1.getChannels() != img2.getChannels())
    {
        std::cerr << "Erreur : Les images doivent avoir les mêmes dimensions et le même nombre de canaux" << std::endl;
        return -1;
    }

    float sum = 0;
    for (int y = 0; y < img1.getHeight(); y++)
    {
        for (int x = 0; x < img1.getWidth(); x++)
        {
            for (int c = 0; c < img1.getChannels(); c++)
            {
                float diff = img1.getClampedPixel(x, y, c) - img2.getClampedPixel(x, y, c);
                sum += diff * diff;
            }
        }
    }

    return sum / (img1.getWidth() * img1.getHeight() * img1.getChannels());
}

inline float PSNR(const Image &img1, const Image &img2)
{
    float mse = MSE(img1, img2);
    if (mse < 0)
    {
        return -1;
    }

    return 10 * log10(255 * 255 / mse);
}

inline float SSIM(const Image &img1, const Image &img2)
{
    if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight() ||
        img1.getChannels() != img2.getChannels())
    {
        std::cerr << "Erreur : Les images doivent avoir les mêmes dimensions et le même nombre de canaux" << std::endl;
        return -1;
    }

    float c1 = 6.5025f, c2 = 58.5225f;
    float ssim = 0;

    for (int c = 0; c < img1.getChannels(); c++)
    {
        float mu_x = 0, mu_y = 0;
        float mu_x_sq = 0, mu_y_sq = 0, mu_x_mu_y = 0;
        float sigma_x = 0, sigma_y = 0, sigma_xy = 0;

        for (int y = 0; y < img1.getHeight(); y++)
        {
            for (int x = 0; x < img1.getWidth(); x++)
            {
                float x_val = img1.getClampedPixel(x, y, c);
                float y_val = img2.getClampedPixel(x, y, c);

                mu_x += x_val;
                mu_y += y_val;
                mu_x_sq += x_val * x_val;
                mu_y_sq += y_val * y_val;
                mu_x_mu_y += x_val * y_val;
            }
        }

        mu_x /= img1.getWidth() * img1.getHeight();
        mu_y /= img1.getWidth() * img1.getHeight();
        mu_x_sq /= img1.getWidth() * img1.getHeight();
        mu_y_sq /= img1.getWidth() * img1.getHeight();
        mu_x_mu_y /= img1.getWidth() * img1.getHeight();

        sigma_x = mu_x_sq - mu_x * mu_x;
        sigma_y = mu_y_sq - mu_y * mu_y;
        sigma_xy = mu_x_mu_y - mu_x * mu_y;

        float numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2);
        float denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2);

        ssim += numerator / denominator;
    }

    return ssim / img1.getChannels();
}

inline std::vector<std::string> getFilesInDirectory(const std::string &directory)
{
    std::vector<std::string> files;

    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        files.push_back(entry.path().string());
    }

    return files;
}

inline std::string basename(const std::string &path)
{
    size_t lastSlash = path.find_last_of('/');
    if (lastSlash == std::string::npos)
    {
        return path;
    }
    return path.substr(lastSlash + 1);
}

inline std::string basenameWithoutExt(const std::string &path)
{
    // Trouver la position du dernier '/' pour isoler le nom de fichier
    size_t lastSlash = path.find_last_of('/');
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

    // Trouver la position du dernier '.' pour retirer l'extension
    size_t lastDot = filename.find_last_of('.');
    if (lastDot == std::string::npos)
    {
        return filename; // Pas d'extension trouvée
    }

    return filename.substr(0, lastDot); // Retourne le nom sans extension
}

inline std::string dirname(const std::string &path)
{
    size_t lastSlash = path.find_last_of('/');
    if (lastSlash == std::string::npos)
    {
        return "";
    }
    return path.substr(0, lastSlash);
}

inline std::string noiseTypeToString(NoiseType noiseType)
{
    switch (noiseType)
    {
    case NoiseType::GAUSSIAN:
        return "gaussian";
    case NoiseType::POISSON:
        return "poisson";
    case NoiseType::SALT_PEPPER:
        return "salt_pepper";
    case NoiseType::SPECKLE:
        return "speckle";
    default:
        return "unknown"; // Cas par défaut pour les valeurs non prises en charge
    }
}

inline bool fileExists(const std::string &path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

inline bool createFolder(const std::string &path)
{
    if (path.empty() || std::filesystem::exists(path))
    {
        return true;
    }
    return std::filesystem::create_directory(path);
}

inline float getLuminance(unsigned char r, unsigned char g, unsigned char b)
{
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

inline std::string filterTypeToString(FilterType filterType)
{
    switch (filterType)
    {
    case FilterType::BILATERAL:
        return "bilateral";
    case FilterType::GAUSSIAN:
        return "gaussian";
    case FilterType::NONLOCAL_MEANS:
        return "nonlocal_means";
    case FilterType::MEDIAN:
        return "median";
    case FilterType::FFDNET_PRETRAINED:
        return "ffdnet_pretrained";
    default:
        return "unknown"; // Cas par défaut pour les valeurs non prises en charge
    }
}

template <typename T> inline T clamp(T value, T min, T max)
{
    return std::max(min, std::min(value, max));
}

} // namespace utils
